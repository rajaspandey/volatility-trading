"""SharpeClassifier: Random Forest with walk-forward retraining."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SharpeClassifier:
    """Walk-forward Random Forest classifier for trade filtering.

    Retrains every `retrain_freq_months` months on a trailing 2-year window.
    Predicts P(y1) = probability of a bad trade and P(y2) = probability of a good trade.

    Attributes
    ----------
    c1 : float
        Dynamic threshold for S1 strategy; adjusted daily to target a trade
        frequency between target_freq_lo and target_freq_hi.
    """

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.n_estimators  = p.get("n_estimators", 300)
        self.max_depth     = p.get("max_depth", 6)
        self.min_samples_leaf = p.get("min_samples_leaf", 50)
        self.max_features  = p.get("max_features", 0.25)
        self.retrain_freq  = p.get("retrain_freq_months", 4)
        self.train_window  = p.get("train_window_years", 2)
        self.target_freq_lo = p.get("target_trade_freq_lo", 0.70)
        self.target_freq_hi = p.get("target_trade_freq_hi", 0.85)

        self._clf_y1: RandomForestClassifier | None = None
        self._clf_y2: RandomForestClassifier | None = None
        self._scaler: StandardScaler = StandardScaler()
        self._last_retrain: pd.Timestamp | None = None
        self.c1: float = 0.5    # dynamic threshold for S1
        self._recent_probs: list[float] = []

    def _build_clf(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y1: pd.Series, y2: pd.Series) -> None:
        """Train both classifiers on labeled data."""
        if len(X) < 50:
            logger.debug("Skipping fit: only %d samples", len(X))
            return
        X_scaled = self._scaler.fit_transform(X.fillna(0))
        self._clf_y1 = self._build_clf()
        self._clf_y2 = self._build_clf()
        self._clf_y1.fit(X_scaled, y1)
        self._clf_y2.fit(X_scaled, y2)
        logger.info("Retrained classifiers on %d samples", len(X))

    def predict_proba(self, x: pd.Series) -> Tuple[float, float]:
        """Return (P(y1), P(y2)) for a single feature row.

        Returns (nan, nan) if model not yet trained.
        """
        if self._clf_y1 is None or self._clf_y2 is None:
            return float("nan"), float("nan")
        x_arr = self._scaler.transform(x.fillna(0).values.reshape(1, -1))

        def _safe_proba(clf, arr) -> float:
            proba = clf.predict_proba(arr)[0]
            # If classifier saw only one class, proba is 1-column; return 0.0
            if len(proba) == 1:
                return 0.0 if clf.classes_[0] == 0 else 1.0
            return float(proba[1])

        p_y1 = _safe_proba(self._clf_y1, x_arr)
        p_y2 = _safe_proba(self._clf_y2, x_arr)
        return p_y1, p_y2

    def maybe_retrain(
        self,
        ts: pd.Timestamp,
        labeled_dataset: pd.DataFrame,
        feature_cols: list[str],
    ) -> bool:
        """Retrain if enough time has passed since last retrain.

        labeled_dataset: output of build_labeled_dataset (has feature + y1/y2 cols).
        Returns True if retrain occurred.
        """
        if self._last_retrain is not None:
            months_since = (ts.year - self._last_retrain.year) * 12 + (ts.month - self._last_retrain.month)
            if months_since < self.retrain_freq:
                return False

        # Training window: trailing 2 years
        cutoff = ts - pd.DateOffset(years=self.train_window)
        window = labeled_dataset[
            (labeled_dataset["ts_open"] >= cutoff) &
            (labeled_dataset["ts_open"] <  ts)
        ]

        available_cols = [c for c in feature_cols if c in window.columns]
        if window.empty or len(available_cols) == 0:
            return False

        X = window[available_cols]
        y1 = window["y1"]
        y2 = window["y2"]

        self.fit(X, y1, y2)
        self._last_retrain = ts
        return True

    def update_c1(self, p_y1: float) -> None:
        """Track recent P(y1) predictions to adjust threshold c1.

        c1 is adjusted to maintain trade frequency in [target_freq_lo, target_freq_hi].
        """
        if not np.isnan(p_y1):
            self._recent_probs.append(p_y1)
            if len(self._recent_probs) > 63:   # ~3 months
                self._recent_probs.pop(0)

        if len(self._recent_probs) >= 20:
            trade_freq = np.mean([p < self.c1 for p in self._recent_probs])
            if trade_freq < self.target_freq_lo:
                self.c1 = min(self.c1 + 0.01, 0.95)
            elif trade_freq > self.target_freq_hi:
                self.c1 = max(self.c1 - 0.01, 0.05)

    @property
    def is_trained(self) -> bool:
        return self._clf_y1 is not None
