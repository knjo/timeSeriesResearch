"""Small shared statistics for the defensive-long-universe study (no framework edits)."""

from __future__ import annotations

import math

import numpy as np

ANNUALIZATION = 252.0
HAC_LAG = 5


def hac_mean_test(values: np.ndarray, lag: int = HAC_LAG) -> dict[str, float]:
    """Equal-weight mean with Bartlett Newey-West standard error over the given lag."""

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return {"n": float(n), "mean": float(np.mean(x)) if n else math.nan, "se": math.nan, "t": math.nan, "p": math.nan}
    mean = float(np.mean(x))
    e = x - mean
    gamma0 = float(np.dot(e, e) / n)
    var = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        var += 2.0 * w * float(np.dot(e[k:], e[:-k]) / n)
    se = math.sqrt(max(var, 0.0) / n)
    t = mean / se if se > 0 else math.nan
    p = 2.0 * (1.0 - _norm_cdf(abs(t))) if math.isfinite(t) else math.nan
    return {"n": float(n), "mean": mean, "se": se, "t": t, "p": p}


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def calendar_sharpe(daily_pnl: np.ndarray) -> float:
    x = np.asarray(daily_pnl, dtype=float)
    if x.size < 2:
        return math.nan
    sd = float(np.std(x, ddof=1))
    if sd <= np.finfo(float).eps:
        return math.nan
    return float(np.mean(x) / sd * math.sqrt(ANNUALIZATION))


def max_drawdown(daily_pnl: np.ndarray) -> float:
    x = np.asarray(daily_pnl, dtype=float)
    path = np.concatenate(([0.0], np.cumsum(x)))
    return float(np.min(path - np.maximum.accumulate(path)))
