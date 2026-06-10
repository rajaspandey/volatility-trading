"""MLAugmentedStrangleStrategy: wraps WeeklyStrangleStrategy with ML gate.

S1: skip trade if P(y1) >= c1 (bad trade predicted)
S2: S1 + double quantity if P(y2) > 0.5 and margin > 25%
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from src.strategies.strangle import WeeklyStrangleStrategy
from src.options.position import OptionPosition
from src.ml.features import build_features, feature_names
from src.ml.labeler import build_labeled_dataset
from src.ml.classifier import SharpeClassifier

logger = logging.getLogger(__name__)


class MLAugmentedStrangleStrategy:
    """ML-augmented strangle (ss=0.1) with S1 and S2 variants.

    variant: 's1' or 's2'
    Cold start: ML unavailable until start + 2 years; base strangle trades normally.
    """

    def __init__(
        self,
        variant: str = "s1",
        ml_params: dict | None = None,
        strangle_size: float = 0.1,
    ):
        self.variant = variant.lower()
        self._base = WeeklyStrangleStrategy(strangle_size)
        self._clf = SharpeClassifier(ml_params)
        self._daily: pd.DataFrame | None = None
        self._feature_cols = feature_names()
        self._features: pd.DataFrame | None = None

    def set_daily_data(self, daily: pd.DataFrame) -> None:
        """Inject daily data for feature construction (called once before backtest)."""
        self._daily = daily
        self._features = None   # will be built lazily

    def _ensure_features(self, equity_curve: pd.DataFrame) -> None:
        """Build features from daily data + current equity curve."""
        if self._daily is None:
            return
        self._features = build_features(self._daily, equity_curve)

    def on_open(self, ts, portfolio, universe, surface, pricer) -> List[OptionPosition]:
        """Open strangle, optionally gated or sized by ML signal."""
        base_positions = self._base.on_open(ts, portfolio, universe, surface, pricer)
        if not base_positions:
            return []

        # Try ML gate if model is trained
        if not self._clf.is_trained or self._features is None:
            return base_positions

        ts_key = pd.Timestamp(ts).normalize()
        if ts_key not in self._features.index:
            return base_positions

        x = self._features.loc[ts_key, [c for c in self._feature_cols if c in self._features.columns]]
        p_y1, p_y2 = self._clf.predict_proba(x)

        self._clf.update_c1(p_y1)

        # S1: skip if P(y1) >= c1
        if np.isnan(p_y1):
            return base_positions

        if p_y1 >= self._clf.c1:
            logger.debug("%s ML-S1 skip trade (P(y1)=%.3f >= c1=%.3f)", ts, p_y1, self._clf.c1)
            return []

        # S2: double if P(y2) > 0.5 and margin > 25%
        if self.variant == "s2" and not np.isnan(p_y2) and p_y2 > 0.5:
            equity = portfolio.mark_to_market(ts, universe, surface, pricer)
            margin = (equity - portfolio.initial_capital) / portfolio.initial_capital
            if margin > 0.25:
                doubled = []
                for pos in base_positions:
                    from dataclasses import replace
                    doubled.append(replace(pos, quantity=pos.quantity * 2))
                logger.debug("%s ML-S2 double position (P(y2)=%.3f)", ts, p_y2)
                return doubled

        return base_positions

    def maybe_retrain(self, ts: pd.Timestamp, trade_log: pd.DataFrame) -> None:
        """Retrain classifier if enough time has passed."""
        if self._features is None or trade_log.empty:
            return
        labeled = build_labeled_dataset(trade_log, self._features)
        self._clf.maybe_retrain(ts, labeled, self._feature_cols)

    def on_risk_exit(self, ts) -> None:
        self._base.on_risk_exit(ts)
