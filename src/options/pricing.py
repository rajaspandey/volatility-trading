"""OptionPricer: Black-Scholes price and Greeks implemented with scipy.

py_vollib_vectorized has numba incompatibilities on Python 3.13.
This implementation uses scipy.stats.norm and numpy vectorization,
which is fast enough for our ~50K BS calls/year.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Minimum time-to-expiry to avoid T=0 singularity (half a trading day)
MIN_T = 0.5 / 252
MIN_SIGMA = 1e-6


def _safe_t(T):
    return np.maximum(np.asarray(T, dtype=float), MIN_T)


def _d1d2(S, K, T, r, sigma):
    """Compute d1 and d2 for Black-Scholes."""
    sigma = np.maximum(sigma, MIN_SIGMA)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def _bs_price(flag, S, K, T, r, sigma):
    """Black-Scholes price for European call ('c') or put ('p')."""
    T = _safe_t(T)
    d1, d2 = _d1d2(S, K, T, r, sigma)
    disc = np.exp(-r * T)
    if np.ndim(flag) == 0:
        # scalar flag
        if flag == "c":
            return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
        else:
            return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        # array of flags
        flag = np.asarray(flag)
        call_val = S * norm.cdf(d1) - K * disc * norm.cdf(d2)
        put_val  = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
        return np.where(flag == "c", call_val, put_val)


def _bs_delta(flag, S, K, T, r, sigma):
    T = _safe_t(T)
    d1, _ = _d1d2(S, K, T, r, sigma)
    if np.ndim(flag) == 0:
        if flag == "c":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1.0
    else:
        flag = np.asarray(flag)
        return np.where(flag == "c", norm.cdf(d1), norm.cdf(d1) - 1.0)


def _bs_gamma(S, K, T, r, sigma):
    T = _safe_t(T)
    d1, _ = _d1d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * np.maximum(sigma, MIN_SIGMA) * np.sqrt(T))


def _bs_vega(S, K, T, r, sigma):
    """Vega per 1-unit change in sigma (same for call and put)."""
    T = _safe_t(T)
    d1, _ = _d1d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def _bs_theta(flag, S, K, T, r, sigma):
    """Theta per calendar day (annualized / 365)."""
    T = _safe_t(T)
    d1, d2 = _d1d2(S, K, T, r, sigma)
    disc = np.exp(-r * T)
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if np.ndim(flag) == 0:
        if flag == "c":
            return (term1 - r * K * disc * norm.cdf(d2)) / 365.0
        else:
            return (term1 + r * K * disc * norm.cdf(-d2)) / 365.0
    else:
        flag = np.asarray(flag)
        call_t = (term1 - r * K * disc * norm.cdf(d2)) / 365.0
        put_t  = (term1 + r * K * disc * norm.cdf(-d2)) / 365.0
        return np.where(flag == "c", call_t, put_t)


class OptionPricer:
    """Vectorized Black-Scholes pricer using scipy/numpy.

    All methods accept scalars or numpy arrays.
    flag = 'c' (call) or 'p' (put).
    T = time to expiry in years (trading_days / 252).
    sigma = annualized implied volatility as decimal (0.20 = 20%).
    """

    def price(self, flag, S, K, T, r, sigma):
        out = _bs_price(flag, S, K, T, r, sigma)
        return float(out) if np.ndim(out) == 0 else np.asarray(out, dtype=float)

    def delta(self, flag, S, K, T, r, sigma):
        out = _bs_delta(flag, S, K, T, r, sigma)
        return float(out) if np.ndim(out) == 0 else np.asarray(out, dtype=float)

    def gamma(self, S, K, T, r, sigma):
        out = _bs_gamma(S, K, T, r, sigma)
        return float(out) if np.ndim(out) == 0 else np.asarray(out, dtype=float)

    def theta(self, flag, S, K, T, r, sigma):
        out = _bs_theta(flag, S, K, T, r, sigma)
        return float(out) if np.ndim(out) == 0 else np.asarray(out, dtype=float)

    def vega(self, S, K, T, r, sigma):
        out = _bs_vega(S, K, T, r, sigma)
        return float(out) if np.ndim(out) == 0 else np.asarray(out, dtype=float)

    def implied_vol(self, flag: str, S: float, K: float, T: float, r: float, price: float) -> float:
        """Newton-Brent IV solver for scalar inputs."""
        T = float(_safe_t(T))
        intrinsic = max(0.0, (S - K) if flag == "c" else (K - S))
        if price <= intrinsic + 1e-8:
            return float("nan")
        try:
            def objective(sigma):
                return _bs_price(flag, S, K, T, r, sigma) - price
            return brentq(objective, 1e-6, 10.0, xtol=1e-6, maxiter=100)
        except Exception:
            return float("nan")

    def price_and_greeks(self, flag: str, S: float, K: float, T: float, r: float, sigma: float) -> dict:
        """All Greeks in one call (scalar inputs)."""
        return {
            "price": self.price(flag, S, K, T, r, sigma),
            "delta": self.delta(flag, S, K, T, r, sigma),
            "gamma": self.gamma(S, K, T, r, sigma),
            "theta": self.theta(flag, S, K, T, r, sigma),
            "vega":  self.vega(S, K, T, r, sigma),
        }

    def batch_price_and_delta(self, flags, S_arr, K_arr, T_arr, r_arr, sigma_arr):
        """Vectorized price + delta for arrays of positions (MTM loop)."""
        T_arr = _safe_t(np.asarray(T_arr, dtype=float))
        flags = np.asarray(flags)
        S_arr = np.asarray(S_arr, dtype=float)
        K_arr = np.asarray(K_arr, dtype=float)
        r_arr = np.asarray(r_arr, dtype=float)
        sigma_arr = np.asarray(sigma_arr, dtype=float)
        return (
            np.asarray(_bs_price(flags, S_arr, K_arr, T_arr, r_arr, sigma_arr), dtype=float),
            np.asarray(_bs_delta(flags, S_arr, K_arr, T_arr, r_arr, sigma_arr), dtype=float),
        )

    def batch_gamma(self, S_arr, K_arr, T_arr, r_arr, sigma_arr):
        T_arr = _safe_t(np.asarray(T_arr, dtype=float))
        return np.asarray(_bs_gamma(
            np.asarray(S_arr, dtype=float),
            np.asarray(K_arr, dtype=float),
            T_arr,
            np.asarray(r_arr, dtype=float),
            np.asarray(sigma_arr, dtype=float),
        ), dtype=float)
