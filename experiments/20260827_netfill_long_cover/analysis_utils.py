"""Pure statistical helpers for the retrospective long-cover study.

The functions in this module deliberately do no file I/O and do not select a
candidate or a portfolio weight.  In particular, the DSR helper is an
approximation when cross-trial Sharpe dispersion is not supplied; see
``deflated_sharpe_ratio`` for the exact fallback assumption.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats


DEFAULT_ANNUALIZATION = 252.0
DEFAULT_BLOCK_LENGTH = 5
EULER_MASCHERONI = float(np.euler_gamma)


def _finite_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _positive_fraction(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must be finite and inside (0, 1]")
    return result


def calendar_sharpe(
    daily_pnl: Sequence[float] | np.ndarray,
    *,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> float:
    """Return sample calendar Sharpe after zero-PnL sessions are already filled.

    The input is PnL rather than return, so this is a scale-invariant daily PnL
    Sharpe.  It is undefined for fewer than two observations or zero variance.
    """

    values = _finite_vector(daily_pnl, name="daily_pnl")
    annualization = float(annualization)
    if not math.isfinite(annualization) or annualization <= 0.0:
        raise ValueError("annualization must be finite and positive")
    if values.size < 2:
        return math.nan
    dispersion = float(np.std(values, ddof=1))
    if dispersion <= np.finfo(float).eps:
        return math.nan
    return float(np.mean(values) / dispersion * math.sqrt(annualization))


def max_drawdown(daily_pnl: Sequence[float] | np.ndarray) -> float:
    """Return absolute maximum drawdown, including the initial zero capital mark."""

    values = _finite_vector(daily_pnl, name="daily_pnl")
    cumulative = np.cumsum(values, dtype=float)
    path = np.concatenate((np.array([0.0]), cumulative))
    high_water = np.maximum.accumulate(path)
    return float(np.min(path - high_water))


def historical_expected_shortfall(
    daily_pnl: Sequence[float] | np.ndarray,
    *,
    tail_fraction: float = 0.05,
) -> float:
    """Return the average of the worst ``ceil(tail_fraction * N)`` PnL days."""

    values = _finite_vector(daily_pnl, name="daily_pnl")
    tail_fraction = _positive_fraction(tail_fraction, name="tail_fraction")
    tail_count = max(1, math.ceil(tail_fraction * values.size))
    return float(np.mean(np.sort(values)[:tail_count]))


def calendar_pnl_metrics(
    daily_pnl: Sequence[float] | np.ndarray,
    *,
    annualization: float = DEFAULT_ANNUALIZATION,
    expected_shortfall_fraction: float = 0.05,
) -> dict[str, float | int]:
    """Summarize a zero-filled calendar PnL vector without changing its scale."""

    values = _finite_vector(daily_pnl, name="daily_pnl")
    return {
        "calendar_days": int(values.size),
        "total_pnl": float(np.sum(values)),
        "mean_daily_pnl": float(np.mean(values)),
        "calendar_sharpe": calendar_sharpe(
            values,
            annualization=annualization,
        ),
        "max_drawdown": max_drawdown(values),
        "expected_shortfall_5pct": historical_expected_shortfall(
            values,
            tail_fraction=expected_shortfall_fraction,
        ),
    }


def _circular_block_bootstrap_means(
    centered_matrix: np.ndarray,
    *,
    block_length: int,
    repetitions: int,
    seed: int,
    batch_size: int = 1024,
) -> np.ndarray:
    """Draw synchronous circular blocks and return one mean per trial and draw."""

    n_periods, n_trials = centered_matrix.shape
    effective_block = min(block_length, n_periods)
    blocks_needed = math.ceil(n_periods / effective_block)
    rng = np.random.default_rng(int(seed))
    starts = rng.integers(
        0,
        n_periods,
        size=(repetitions, blocks_needed),
        dtype=np.int64,
    )
    offsets = np.arange(effective_block, dtype=np.int64)
    result = np.empty((repetitions, n_trials), dtype=float)
    for begin in range(0, repetitions, batch_size):
        end = min(begin + batch_size, repetitions)
        indices = (
            (starts[begin:end, :, None] + offsets[None, None, :]) % n_periods
        ).reshape(end - begin, -1)[:, :n_periods]
        result[begin:end, :] = np.mean(centered_matrix[indices, :], axis=1)
    return result


def moving_block_max_t(
    trial_daily_pnl: Mapping[str, Sequence[float] | np.ndarray],
    *,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    repetitions: int = 10_000,
    seed: int = 20260827,
    family_alpha: float = 0.05,
) -> dict[str, Any]:
    """One-sided synchronous circular moving-block max-t family test.

    Each trial is centered under its own zero-mean null.  The bootstrap standard
    error of its mean is used as a fixed studentizer, and the maximum t statistic
    across estimable trials supplies a single-step familywise adjusted p-value.
    Trials with no time-series variation are retained as non-estimable with
    p-value 1 instead of being silently dropped from the ledger.
    """

    if not trial_daily_pnl:
        raise ValueError("trial_daily_pnl must contain at least one trial")
    if int(block_length) != block_length or block_length < 1:
        raise ValueError("block_length must be a positive integer")
    if int(repetitions) != repetitions or repetitions < 2:
        raise ValueError("repetitions must be an integer of at least two")
    family_alpha = float(family_alpha)
    if not math.isfinite(family_alpha) or not 0.0 < family_alpha < 1.0:
        raise ValueError("family_alpha must be finite and inside (0, 1)")

    trial_ids = sorted(trial_daily_pnl)
    vectors = [
        _finite_vector(trial_daily_pnl[trial_id], name=f"trial {trial_id!r}")
        for trial_id in trial_ids
    ]
    lengths = {len(vector) for vector in vectors}
    if len(lengths) != 1:
        raise ValueError("all trial daily PnL arrays must have equal length")
    n_periods = lengths.pop()
    if n_periods < 2:
        raise ValueError("at least two calendar periods are required")

    matrix = np.column_stack(vectors)
    point_means = np.mean(matrix, axis=0)
    centered = matrix - point_means
    bootstrap_means = _circular_block_bootstrap_means(
        centered,
        block_length=int(block_length),
        repetitions=int(repetitions),
        seed=int(seed),
    )
    bootstrap_se = np.std(bootstrap_means, axis=0, ddof=1)
    estimable = np.isfinite(bootstrap_se) & (
        bootstrap_se > np.finfo(float).eps
    )

    null_t = np.full(bootstrap_means.shape, np.nan, dtype=float)
    observed_t = np.full(len(trial_ids), np.nan, dtype=float)
    null_t[:, estimable] = (
        bootstrap_means[:, estimable] / bootstrap_se[estimable]
    )
    observed_t[estimable] = point_means[estimable] / bootstrap_se[estimable]

    if estimable.any():
        max_null_t = np.max(null_t[:, estimable], axis=1)
        critical_value = float(np.quantile(max_null_t, 1.0 - family_alpha))
    else:
        max_null_t = np.zeros(int(repetitions), dtype=float)
        critical_value = math.nan

    rows: dict[str, dict[str, float | bool | int]] = {}
    for column, trial_id in enumerate(trial_ids):
        if estimable[column]:
            own_null = null_t[:, column]
            statistic = float(observed_t[column])
            raw_p = float(
                (1 + np.count_nonzero(own_null >= statistic))
                / (int(repetitions) + 1)
            )
            adjusted_p = float(
                (1 + np.count_nonzero(max_null_t >= statistic))
                / (int(repetitions) + 1)
            )
            simultaneous_lower_bound = float(
                point_means[column] - critical_value * bootstrap_se[column]
            )
        else:
            statistic = math.nan
            raw_p = 1.0
            adjusted_p = 1.0
            simultaneous_lower_bound = math.nan
        rows[trial_id] = {
            "calendar_days": int(n_periods),
            "mean_daily_pnl": float(point_means[column]),
            "bootstrap_se_mean": float(bootstrap_se[column]),
            "observed_t": statistic,
            "raw_one_sided_p": raw_p,
            "max_t_adjusted_p": adjusted_p,
            "simultaneous_lower_bound": simultaneous_lower_bound,
            "estimable": bool(estimable[column]),
        }

    return {
        "method": "circular_moving_block_single_step_max_t_one_sided",
        "alternative": "positive_mean_daily_pnl",
        "calendar_days": int(n_periods),
        "trial_count": len(trial_ids),
        "block_length_requested": int(block_length),
        "block_length_effective": min(int(block_length), n_periods),
        "bootstrap_repetitions": int(repetitions),
        "seed": int(seed),
        "family_alpha": family_alpha,
        "max_t_critical_value": critical_value,
        "trials": rows,
    }


def _sample_sharpe_moments(
    daily_pnl: Sequence[float] | np.ndarray,
    *,
    annualization: float,
) -> dict[str, float | int]:
    values = _finite_vector(daily_pnl, name="daily_pnl")
    if values.size < 4:
        raise ValueError("at least four observations are required for Sharpe moments")
    annualization = float(annualization)
    if not math.isfinite(annualization) or annualization <= 0.0:
        raise ValueError("annualization must be finite and positive")
    dispersion = float(np.std(values, ddof=1))
    if dispersion <= np.finfo(float).eps:
        raise ValueError("daily_pnl must have positive sample variance")
    daily_sharpe = float(np.mean(values) / dispersion)
    skewness = float(stats.skew(values, bias=False))
    pearson_kurtosis = float(stats.kurtosis(values, fisher=False, bias=False))
    if not math.isfinite(skewness) or not math.isfinite(pearson_kurtosis):
        raise ValueError("skewness and kurtosis must be estimable")
    return {
        "observations": int(values.size),
        "daily_sharpe": daily_sharpe,
        "annualized_sharpe": daily_sharpe * math.sqrt(annualization),
        "sample_skewness": skewness,
        "sample_pearson_kurtosis": pearson_kurtosis,
    }


def probabilistic_sharpe_ratio(
    daily_pnl: Sequence[float] | np.ndarray,
    *,
    benchmark_annualized_sharpe: float = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> dict[str, float | int | str]:
    """Bailey-style PSR with sample skewness and Pearson kurtosis adjustment.

    ``benchmark_annualized_sharpe`` is converted to the same daily frequency as
    the observed Sharpe before evaluating the asymptotic normal probability.
    """

    moments = _sample_sharpe_moments(
        daily_pnl,
        annualization=annualization,
    )
    benchmark_annualized_sharpe = float(benchmark_annualized_sharpe)
    if not math.isfinite(benchmark_annualized_sharpe):
        raise ValueError("benchmark_annualized_sharpe must be finite")
    benchmark_daily = benchmark_annualized_sharpe / math.sqrt(annualization)
    observed_daily = float(moments["daily_sharpe"])
    skewness = float(moments["sample_skewness"])
    kurtosis = float(moments["sample_pearson_kurtosis"])
    variance_term = (
        1.0
        - skewness * observed_daily
        + ((kurtosis - 1.0) / 4.0) * observed_daily**2
    )
    if not math.isfinite(variance_term) or variance_term <= 0.0:
        raise ValueError("Bailey Sharpe variance adjustment is not positive")
    observations = int(moments["observations"])
    standard_error_daily = math.sqrt(variance_term / (observations - 1))
    z_statistic = (observed_daily - benchmark_daily) / standard_error_daily
    probability = float(stats.norm.cdf(z_statistic))
    return {
        **moments,
        "benchmark_daily_sharpe": benchmark_daily,
        "benchmark_annualized_sharpe": benchmark_annualized_sharpe,
        "sharpe_standard_error_daily": standard_error_daily,
        "psr_z": z_statistic,
        "psr_probability": probability,
        "method": "bailey_probabilistic_sharpe_skew_kurtosis_adjusted",
    }


def expected_maximum_sharpe(
    *,
    trial_count: int,
    trial_sharpe_std_annualized: float,
) -> float:
    """Approximate expected best annualized Sharpe among independent trials."""

    if int(trial_count) != trial_count or trial_count < 1:
        raise ValueError("trial_count must be a positive integer")
    trial_sharpe_std_annualized = float(trial_sharpe_std_annualized)
    if (
        not math.isfinite(trial_sharpe_std_annualized)
        or trial_sharpe_std_annualized < 0.0
    ):
        raise ValueError("trial_sharpe_std_annualized must be finite and non-negative")
    if trial_count == 1 or trial_sharpe_std_annualized == 0.0:
        return 0.0
    z_one = float(stats.norm.ppf(1.0 - 1.0 / trial_count))
    z_e = float(stats.norm.ppf(1.0 - 1.0 / (trial_count * math.e)))
    return float(
        trial_sharpe_std_annualized
        * ((1.0 - EULER_MASCHERONI) * z_one + EULER_MASCHERONI * z_e)
    )


def deflated_sharpe_ratio(
    daily_pnl: Sequence[float] | np.ndarray,
    *,
    trial_count: int,
    trial_sharpe_std_annualized: float | None = None,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> dict[str, float | int | str | bool]:
    """Return an approximate DSR probability for a stated trial count.

    Bailey's expected-maximum-Sharpe benchmark ideally uses the observed
    cross-trial standard deviation of annualized Sharpe estimates.  If it is not
    supplied, this helper transparently substitutes the iid zero-Sharpe sampling
    standard error ``sqrt(annualization / (N_days - 1))``.  That fallback makes
    this a screening diagnostic, not a full reconstruction of historical trial
    dispersion or dependence.
    """

    moments = _sample_sharpe_moments(
        daily_pnl,
        annualization=annualization,
    )
    if int(trial_count) != trial_count or trial_count < 1:
        raise ValueError("trial_count must be a positive integer")
    used_fallback = trial_sharpe_std_annualized is None
    if used_fallback:
        trial_sharpe_std_annualized = math.sqrt(
            float(annualization) / (int(moments["observations"]) - 1)
        )
    benchmark = expected_maximum_sharpe(
        trial_count=int(trial_count),
        trial_sharpe_std_annualized=float(trial_sharpe_std_annualized),
    )
    psr = probabilistic_sharpe_ratio(
        daily_pnl,
        benchmark_annualized_sharpe=benchmark,
        annualization=annualization,
    )
    return {
        **psr,
        "trial_count": int(trial_count),
        "trial_sharpe_std_annualized": float(trial_sharpe_std_annualized),
        "trial_sharpe_std_fallback_used": used_fallback,
        "dsr_benchmark_annualized_sharpe": benchmark,
        "dsr_probability": float(psr["psr_probability"]),
        "method": "approximate_bailey_deflated_sharpe",
    }


def deflated_sharpe_sensitivity(
    daily_pnl: Sequence[float] | np.ndarray,
    *,
    trial_counts: Sequence[int] = (6,),
    trial_sharpe_std_annualized: float | None = None,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> dict[int, dict[str, float | int | str | bool]]:
    """Evaluate the same immutable PnL path at six or any stated trial counts."""

    if not trial_counts:
        raise ValueError("trial_counts must not be empty")
    results: dict[int, dict[str, float | int | str | bool]] = {}
    for raw_count in trial_counts:
        if int(raw_count) != raw_count or raw_count < 1:
            raise ValueError("each trial count must be a positive integer")
        count = int(raw_count)
        if count in results:
            raise ValueError("trial_counts must be unique")
        results[count] = deflated_sharpe_ratio(
            daily_pnl,
            trial_count=count,
            trial_sharpe_std_annualized=trial_sharpe_std_annualized,
            annualization=annualization,
        )
    return results


def _position_total(
    position: float | Sequence[float] | np.ndarray,
    *,
    name: str,
) -> float:
    values = np.asarray(position, dtype=float)
    if values.ndim > 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty scalar or one-dimensional array")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(f"{name} must contain only finite non-negative values")
    return float(np.sum(values))


def position_only_normalization(
    short_position: float | Sequence[float] | np.ndarray,
    long_position: float | Sequence[float] | np.ndarray,
) -> float:
    """Scale long PnL to the short sleeve's aggregate position, without PnL input."""

    short_total = _position_total(short_position, name="short_position")
    long_total = _position_total(long_position, name="long_position")
    if short_total <= 0.0:
        raise ValueError("short_position aggregate must be positive")
    if long_total <= 0.0:
        raise ValueError("long_position aggregate must be positive")
    return short_total / long_total


def _safe_correlation(left: np.ndarray, right: np.ndarray, *, rank: bool) -> float:
    if left.size < 2:
        return math.nan
    if (
        np.std(left, ddof=1) <= np.finfo(float).eps
        or np.std(right, ddof=1) <= np.finfo(float).eps
    ):
        return math.nan
    if rank:
        return float(stats.spearmanr(left, right).statistic)
    return float(np.corrcoef(left, right)[0, 1])


def _conditional_beta(
    dependent: np.ndarray,
    independent: np.ndarray,
    condition: np.ndarray,
) -> float:
    """Return OLS slope with an intercept on the requested conditional days."""

    left = dependent[condition]
    right = independent[condition]
    if left.size < 2:
        return math.nan
    variance = float(np.var(right, ddof=1))
    if variance <= np.finfo(float).eps:
        return math.nan
    covariance = float(np.cov(left, right, ddof=1)[0, 1])
    return covariance / variance


def _fixed_multiplier_frontier(
    short: np.ndarray,
    long: np.ndarray,
    multipliers: Sequence[float],
    *,
    multiplier_field: str,
    annualization: float,
    expected_shortfall_fraction: float,
    short_metrics: Mapping[str, float | int],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for multiplier in multipliers:
        combined = short + multiplier * long
        metrics = calendar_pnl_metrics(
            combined,
            annualization=annualization,
            expected_shortfall_fraction=expected_shortfall_fraction,
        )
        rows.append(
            {
                multiplier_field: multiplier,
                **metrics,
                "incremental_total_pnl_vs_short": float(
                    metrics["total_pnl"] - short_metrics["total_pnl"]
                ),
                "calendar_sharpe_delta_vs_short": float(
                    metrics["calendar_sharpe"] - short_metrics["calendar_sharpe"]
                ),
                "max_drawdown_delta_vs_short": float(
                    metrics["max_drawdown"] - short_metrics["max_drawdown"]
                ),
                "expected_shortfall_5pct_delta_vs_short": float(
                    metrics["expected_shortfall_5pct"]
                    - short_metrics["expected_shortfall_5pct"]
                ),
            }
        )
    return rows


def cover_portfolio_metrics(
    short_daily_pnl: Sequence[float] | np.ndarray,
    long_daily_pnl: Sequence[float] | np.ndarray,
    *,
    short_position: float | Sequence[float] | np.ndarray,
    long_position: float | Sequence[float] | np.ndarray,
    native_long_multipliers: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    normalized_long_multipliers: Sequence[float] | None = None,
    annualization: float = DEFAULT_ANNUALIZATION,
    worst_short_fraction: float = 0.10,
    expected_shortfall_fraction: float = 0.05,
) -> dict[str, Any]:
    """Evaluate native-cash and secondary position-normalized cover frontiers.

    The short calendar is authoritative: callers must first join and zero-fill the
    long sleeve to the exact same dates.  This function checks equal array length
    but intentionally has no date-join or multiplier-selection behavior.  The
    primary frontier is exactly ``short + h * long`` in native cash PnL.  Aggregate
    entry position is turnover-like rather than concurrent capital, so its
    normalized frontier is explicitly secondary and descriptive.
    """

    short = _finite_vector(short_daily_pnl, name="short_daily_pnl")
    long = _finite_vector(long_daily_pnl, name="long_daily_pnl")
    if short.size != long.size:
        raise ValueError("short_daily_pnl and long_daily_pnl must have equal length")
    worst_short_fraction = _positive_fraction(
        worst_short_fraction,
        name="worst_short_fraction",
    )
    _positive_fraction(
        expected_shortfall_fraction,
        name="expected_shortfall_fraction",
    )
    if not native_long_multipliers:
        raise ValueError("native_long_multipliers must not be empty")
    native_multipliers = [float(value) for value in native_long_multipliers]
    if any(
        not math.isfinite(value) or value < 0.0 for value in native_multipliers
    ):
        raise ValueError("native_long_multipliers must be finite and non-negative")
    if len(set(native_multipliers)) != len(native_multipliers):
        raise ValueError("native_long_multipliers must be unique")
    normalized_source = (
        native_long_multipliers
        if normalized_long_multipliers is None
        else normalized_long_multipliers
    )
    if not normalized_source:
        raise ValueError("normalized_long_multipliers must not be empty")
    normalized_multipliers = [float(value) for value in normalized_source]
    if any(
        not math.isfinite(value) or value < 0.0
        for value in normalized_multipliers
    ):
        raise ValueError(
            "normalized_long_multipliers must be finite and non-negative"
        )
    if len(set(normalized_multipliers)) != len(normalized_multipliers):
        raise ValueError("normalized_long_multipliers must be unique")

    short_position_total = _position_total(short_position, name="short_position")
    long_position_total = _position_total(long_position, name="long_position")
    normalization = position_only_normalization(
        short_position_total,
        long_position_total,
    )
    normalized_long = long * normalization
    short_metrics = calendar_pnl_metrics(
        short,
        annualization=annualization,
        expected_shortfall_fraction=expected_shortfall_fraction,
    )
    raw_long_metrics = calendar_pnl_metrics(
        long,
        annualization=annualization,
        expected_shortfall_fraction=expected_shortfall_fraction,
    )
    normalized_long_metrics = calendar_pnl_metrics(
        normalized_long,
        annualization=annualization,
        expected_shortfall_fraction=expected_shortfall_fraction,
    )

    short_loss = short < 0.0
    worst_count = max(1, math.ceil(worst_short_fraction * short.size))
    worst_indices = np.argsort(short, kind="stable")[:worst_count]
    mean_on_loss_days = (
        float(np.mean(normalized_long[short_loss])) if short_loss.any() else math.nan
    )
    mean_on_worst_days = float(np.mean(normalized_long[worst_indices]))
    native_diagnostics = {
        "pearson_net_daily_pnl": _safe_correlation(short, long, rank=False),
        "spearman_net_daily_pnl": _safe_correlation(short, long, rank=True),
        "downside_beta_long_on_short_loss_days": _conditional_beta(
            long,
            short,
            short_loss,
        ),
        "short_loss_days": int(np.count_nonzero(short_loss)),
        "mean_native_long_pnl_on_short_loss_days": (
            float(np.mean(long[short_loss])) if short_loss.any() else math.nan
        ),
        "median_native_long_pnl_on_short_loss_days": (
            float(np.median(long[short_loss])) if short_loss.any() else math.nan
        ),
        "native_long_positive_rate_on_short_loss_days": (
            float(np.mean(long[short_loss] > 0.0)) if short_loss.any() else math.nan
        ),
        "worst_short_fraction": worst_short_fraction,
        "worst_short_days": int(worst_count),
        "mean_native_long_pnl_on_worst_short_days": float(
            np.mean(long[worst_indices])
        ),
        "median_native_long_pnl_on_worst_short_days": float(
            np.median(long[worst_indices])
        ),
        "native_long_positive_rate_on_worst_short_days": float(
            np.mean(long[worst_indices] > 0.0)
        ),
    }
    normalized_diagnostics = {
        "downside_beta_long_on_short_loss_days": _conditional_beta(
            normalized_long,
            short,
            short_loss,
        ),
        "mean_long_pnl_on_short_loss_days": mean_on_loss_days,
        "mean_long_pnl_on_worst_short_days": mean_on_worst_days,
    }

    native_frontier = _fixed_multiplier_frontier(
        short,
        long,
        native_multipliers,
        multiplier_field="native_long_multiplier_h",
        annualization=annualization,
        expected_shortfall_fraction=expected_shortfall_fraction,
        short_metrics=short_metrics,
    )
    normalized_frontier = _fixed_multiplier_frontier(
        short,
        normalized_long,
        normalized_multipliers,
        multiplier_field="normalized_long_multiplier_h",
        annualization=annualization,
        expected_shortfall_fraction=expected_shortfall_fraction,
        short_metrics=short_metrics,
    )
    for row in normalized_frontier:
        row["descriptive_position_equivalent"] = (
            float(row["normalized_long_multiplier_h"]) * short_position_total
        )

    return {
        "capital_normalization_method": (
            "short_aggregate_position_divided_by_long_aggregate_position"
        ),
        "short_position_total": short_position_total,
        "long_position_total": long_position_total,
        "long_position_normalization_scale": normalization,
        "primary_frontier_definition": "short_pnl_plus_h_times_native_long_pnl",
        "position_normalization_role": "secondary_descriptive_only",
        "position_measure_limitation": (
            "aggregate entry position is cumulative turnover-like position, not "
            "concurrent capital"
        ),
        "short": short_metrics,
        "long_raw": raw_long_metrics,
        "long_normalized_to_short_position": normalized_long_metrics,
        "native_cover_diagnostics": native_diagnostics,
        "position_normalized_cover_diagnostics": normalized_diagnostics,
        "native_cash_frontier": native_frontier,
        "position_normalized_frontier": normalized_frontier,
    }
