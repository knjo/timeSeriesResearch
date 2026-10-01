"""Report inference statistics from a FillLots_atLow daily sweep.

The input is the ``daily_stats.parquet`` produced by ``run_sweep.py``.  This
script intentionally consumes only additive daily statistics, so reporting is
independent of event construction and does not need to reload market data.

Inference conventions:

* pooled delta-z weights rows within the active event days;
* equal-day statistics give every active session one observation;
* Newey-West uses a Bartlett kernel and at most five observation lags;
* an optional runner-provided matched stock-day estimate is tested separately;
* an optional stock-by-5-minute matched estimate is a separate test family;
* the raw p-value is the asymptotic two-sided p-value for the HAC t-statistic;
* Benjamini-Hochberg correction is across full-period candidate raw p-values.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


DEFAULT_HAC_LAG = 5
PERIODS: tuple[tuple[str, str | None], ...] = (
    ("full", None),
    ("2025", "2025"),
    ("2026", "2026"),
)
STUDY_COLUMNS = (
    "contract_id",
    "family_id",
    "family_version",
    "label_profile_id",
    "label_profile_hash",
    "pool_id",
    "pool_version",
    "control_mode",
    "control_sample_size",
)
OPTIONAL_STUDY_COLUMNS = ("stock_time_bucket", "primary_estimand")
REQUIRED_COLUMNS = {
    "Date",
    "candidate_id",
    "eligible_pool_rows",
    "condition_n",
    "condition_n_z",
    "condition_sum_z",
    "control_n",
    "control_n_z",
    "control_sum_z",
    *STUDY_COLUMNS,
}
COUNT_COLUMNS = (
    "eligible_pool_rows",
    "condition_n",
    "condition_n_z",
    "control_n",
    "control_n_z",
)
SUM_COLUMNS = ("condition_sum_z", "control_sum_z")
STOCK_DAY_EQUAL_COLUMN = "stock_day_equal_delta_z"
STOCK_TIME_EQUAL_COLUMN = "stock_time_equal_delta_z"


class SweepReportError(ValueError):
    """Raised when daily sweep statistics violate the report contract."""


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Return the regularized incomplete beta using a continued fraction."""

    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1]")
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    def continued_fraction(left: float, right: float, value: float) -> float:
        max_iterations = 300
        epsilon = 3.0e-14
        floor = 1.0e-300
        qab = left + right
        qap = left + 1.0
        qam = left - 1.0
        c = 1.0
        d = 1.0 - qab * value / qap
        if abs(d) < floor:
            d = floor
        d = 1.0 / d
        result = d
        for iteration in range(1, max_iterations + 1):
            doubled = 2 * iteration
            numerator = (
                iteration
                * (right - iteration)
                * value
                / ((qam + doubled) * (left + doubled))
            )
            d = 1.0 + numerator * d
            if abs(d) < floor:
                d = floor
            c = 1.0 + numerator / c
            if abs(c) < floor:
                c = floor
            d = 1.0 / d
            result *= d * c

            numerator = -(
                (left + iteration)
                * (qab + iteration)
                * value
                / ((left + doubled) * (qap + doubled))
            )
            d = 1.0 + numerator * d
            if abs(d) < floor:
                d = floor
            c = 1.0 + numerator / c
            if abs(c) < floor:
                c = floor
            d = 1.0 / d
            change = d * c
            result *= change
            if abs(change - 1.0) <= epsilon:
                return result
        raise ArithmeticError("incomplete beta continued fraction did not converge")

    log_scale = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    scale = math.exp(log_scale)
    if x < (a + 1.0) / (a + b + 2.0):
        result = scale * continued_fraction(a, b, x) / a
    else:
        result = 1.0 - scale * continued_fraction(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, result))


def student_t_two_sided_p(t_stat: float | None, degrees: int) -> float | None:
    """Return an exact two-sided Student t p-value without SciPy."""

    if t_stat is None or degrees < 1 or not math.isfinite(t_stat):
        return None
    scaled = degrees / (degrees + t_stat * t_stat)
    return _regularized_incomplete_beta(scaled, degrees / 2.0, 0.5)


def normal_two_sided_p(statistic: float | None) -> float | None:
    """Return an asymptotic two-sided standard-normal p-value."""

    if statistic is None or not math.isfinite(statistic):
        return None
    return math.erfc(abs(statistic) / math.sqrt(2.0))


def equal_weight_stats(values: list[float]) -> dict[str, float | int | None]:
    """Compute mean, standard error, t-statistic, and Student t p-value."""

    count = len(values)
    if count == 0:
        return {"n": 0, "mean": None, "se": None, "t": None, "p": None}
    mean = math.fsum(values) / count
    if count < 2:
        return {"n": count, "mean": mean, "se": None, "t": None, "p": None}
    squared_error = math.fsum((value - mean) ** 2 for value in values)
    variance = squared_error / (count - 1)
    if variance <= 1.0e-30:
        return {"n": count, "mean": mean, "se": None, "t": None, "p": None}
    standard_error = math.sqrt(variance / count)
    t_stat = mean / standard_error
    return {
        "n": count,
        "mean": mean,
        "se": standard_error,
        "t": t_stat,
        "p": student_t_two_sided_p(t_stat, count - 1),
    }


def newey_west_mean_stats(
    values: list[float], requested_lag: int = DEFAULT_HAC_LAG
) -> dict[str, float | int | None]:
    """Compute a Bartlett-kernel Newey-West standard error for a mean."""

    if requested_lag < 0:
        raise ValueError("requested_lag cannot be negative")
    count = len(values)
    effective_lag = min(requested_lag, max(0, count - 1))
    if count == 0:
        return {
            "lag": effective_lag,
            "long_run_variance": None,
            "se": None,
            "t": None,
            "p": None,
        }
    mean = math.fsum(values) / count
    residuals = [value - mean for value in values]
    long_run_variance = math.fsum(value * value for value in residuals) / count
    for lag in range(1, effective_lag + 1):
        covariance = math.fsum(
            residuals[index] * residuals[index - lag]
            for index in range(lag, count)
        ) / count
        weight = 1.0 - lag / (effective_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 1.0e-30:
        return {
            "lag": effective_lag,
            "long_run_variance": max(0.0, long_run_variance),
            "se": None,
            "t": None,
            "p": None,
        }
    standard_error = math.sqrt(long_run_variance / count)
    t_stat = mean / standard_error
    return {
        "lag": effective_lag,
        "long_run_variance": long_run_variance,
        "se": standard_error,
        "t": t_stat,
        "p": normal_two_sided_p(t_stat),
    }


def benjamini_hochberg(
    p_values: dict[str, float | None],
) -> dict[str, float | None]:
    """Adjust finite p-values with the Benjamini-Hochberg step-up method."""

    adjusted = {key: None for key in p_values}
    ranked = sorted(
        (value, key)
        for key, value in p_values.items()
        if value is not None and math.isfinite(value)
    )
    total = len(ranked)
    running = 1.0
    for reverse_index in range(total - 1, -1, -1):
        p_value, key = ranked[reverse_index]
        rank = reverse_index + 1
        running = min(running, p_value * total / rank)
        adjusted[key] = min(1.0, max(0.0, running))
    return adjusted


def _validate_daily_stats(daily: pl.DataFrame) -> pl.DataFrame:
    if daily.is_empty():
        raise SweepReportError("daily statistics are empty")
    missing = sorted(REQUIRED_COLUMNS - set(daily.columns))
    if missing:
        raise SweepReportError(f"daily statistics missing columns: {missing}")

    has_stock_equal = STOCK_DAY_EQUAL_COLUMN in daily.columns
    has_stock_time_equal = STOCK_TIME_EQUAL_COLUMN in daily.columns
    has_stock_time_bucket = "stock_time_bucket" in daily.columns
    if has_stock_time_equal != has_stock_time_bucket:
        missing_pair = (
            "stock_time_bucket"
            if has_stock_time_equal
            else STOCK_TIME_EQUAL_COLUMN
        )
        raise SweepReportError(
            f"stock-time estimand identity missing paired column {missing_pair}"
        )

    # Validate count semantics before casting.  Polars intentionally truncates
    # finite floats when casting to Int64, which would otherwise let a damaged
    # artifact such as condition_n=1.5 silently pass as one row.
    count_probe = daily.select(
        *[
            pl.col(column).cast(pl.Float64, strict=True).alias(column)
            for column in COUNT_COLUMNS
        ]
    )
    for column in COUNT_COLUMNS:
        invalid = count_probe.filter(
            pl.col(column).is_null()
            | ~pl.col(column).is_finite()
            | (pl.col(column) < 0)
            | (pl.col(column) != pl.col(column).floor())
        )
        if not invalid.is_empty():
            raise SweepReportError(
                f"{column} must contain nonnegative integers"
            )

    normalized = daily.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("candidate_id").cast(pl.String),
        *[pl.col(column).cast(pl.Int64) for column in COUNT_COLUMNS],
        *[pl.col(column).cast(pl.Float64) for column in SUM_COLUMNS],
        *(
            [pl.col(STOCK_DAY_EQUAL_COLUMN).cast(pl.Float64)]
            if has_stock_equal
            else []
        ),
        *(
            [pl.col(STOCK_TIME_EQUAL_COLUMN).cast(pl.Float64)]
            if has_stock_time_equal
            else []
        ),
    )
    for column in ("Date", "candidate_id", *STUDY_COLUMNS):
        if normalized[column].null_count():
            raise SweepReportError(f"{column} contains null values")
    for column in STUDY_COLUMNS:
        if normalized[column].n_unique() != 1:
            raise SweepReportError(f"input mixes multiple {column} values")
    if has_stock_time_equal:
        if normalized["stock_time_bucket"].null_count():
            raise SweepReportError("stock_time_bucket contains null values")
        if normalized["stock_time_bucket"].n_unique() != 1:
            raise SweepReportError(
                "input mixes multiple stock_time_bucket values"
            )

    dates = normalized["Date"].unique().to_list()
    for date in dates:
        try:
            parsed = datetime.strptime(date, "%Y%m%d")
        except ValueError as exc:
            raise SweepReportError(f"invalid Date value: {date}") from exc
        if parsed.strftime("%Y%m%d") != date:
            raise SweepReportError(f"invalid Date value: {date}")

    duplicates = (
        normalized.group_by("candidate_id", "Date")
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicates.is_empty():
        raise SweepReportError(
            "duplicate candidate/date rows: "
            f"{duplicates.head(5).to_dicts()}"
        )
    for column in COUNT_COLUMNS:
        invalid = normalized.filter(
            pl.col(column).is_null() | (pl.col(column) < 0)
        )
        if not invalid.is_empty():
            raise SweepReportError(f"{column} must contain nonnegative integers")
    for column in SUM_COLUMNS:
        invalid = normalized.filter(
            pl.col(column).is_null() | ~pl.col(column).is_finite()
        )
        if not invalid.is_empty():
            raise SweepReportError(f"{column} must contain finite values")
    if has_stock_equal:
        invalid = normalized.filter(
            pl.col(STOCK_DAY_EQUAL_COLUMN).is_not_null()
            & ~pl.col(STOCK_DAY_EQUAL_COLUMN).is_finite()
        )
        if not invalid.is_empty():
            raise SweepReportError(
                f"{STOCK_DAY_EQUAL_COLUMN} must contain finite or null values"
            )
        invalid_active_stock_mean = normalized.filter(
            (pl.col("condition_n_z") > 0)
            & (pl.col("control_n_z") > 0)
            & pl.col(STOCK_DAY_EQUAL_COLUMN).is_null()
        )
        if not invalid_active_stock_mean.is_empty():
            raise SweepReportError(
                f"active rows require {STOCK_DAY_EQUAL_COLUMN}"
            )
    if has_stock_time_equal:
        invalid = normalized.filter(
            pl.col(STOCK_TIME_EQUAL_COLUMN).is_not_null()
            & ~pl.col(STOCK_TIME_EQUAL_COLUMN).is_finite()
        )
        if not invalid.is_empty():
            raise SweepReportError(
                f"{STOCK_TIME_EQUAL_COLUMN} must contain finite or null values"
            )
        invalid_active_stock_time_mean = normalized.filter(
            (pl.col("condition_n_z") > 0)
            & (pl.col("control_n_z") > 0)
            & pl.col(STOCK_TIME_EQUAL_COLUMN).is_null()
        )
        if not invalid_active_stock_time_mean.is_empty():
            raise SweepReportError(
                f"active rows require {STOCK_TIME_EQUAL_COLUMN}"
            )
    invalid_counts = normalized.filter(
        (pl.col("condition_n_z") > pl.col("condition_n"))
        | (pl.col("control_n_z") > pl.col("control_n"))
    )
    if not invalid_counts.is_empty():
        raise SweepReportError("z-valid row counts exceed cohort row counts")
    invalid_empty_sums = normalized.filter(
        (
            (pl.col("condition_n_z") == 0)
            & (pl.col("condition_sum_z").abs() > 1.0e-12)
        )
        | (
            (pl.col("control_n_z") == 0)
            & (pl.col("control_sum_z").abs() > 1.0e-12)
        )
    )
    if not invalid_empty_sums.is_empty():
        raise SweepReportError("zero-count cohorts must have a zero z sum")

    for date, group in normalized.group_by("Date", maintain_order=True):
        control_n_values = group["control_n_z"].unique().to_list()
        if len(control_n_values) != 1:
            raise SweepReportError(f"{date[0]} has inconsistent shared control counts")
        control_sums = group["control_sum_z"].to_list()
        reference = control_sums[0]
        if any(
            not math.isclose(value, reference, rel_tol=1.0e-12, abs_tol=1.0e-12)
            for value in control_sums[1:]
        ):
            raise SweepReportError(f"{date[0]} has inconsistent shared control sums")

    return normalized.sort("Date", "candidate_id")


def _enrich_daily(daily: pl.DataFrame) -> pl.DataFrame:
    active = (pl.col("condition_n_z") > 0) & (pl.col("control_n_z") > 0)
    has_stock_equal = STOCK_DAY_EQUAL_COLUMN in daily.columns
    stock_delta = (
        pl.when(active).then(pl.col(STOCK_DAY_EQUAL_COLUMN)).otherwise(None)
        if has_stock_equal
        else pl.lit(None, dtype=pl.Float64)
    )
    has_stock_time_equal = STOCK_TIME_EQUAL_COLUMN in daily.columns
    stock_time_delta = (
        pl.when(active).then(pl.col(STOCK_TIME_EQUAL_COLUMN)).otherwise(None)
        if has_stock_time_equal
        else pl.lit(None, dtype=pl.Float64)
    )
    return daily.with_columns(
        active.alias("active"),
        pl.lit(has_stock_equal).alias("stock_day_equal_available"),
        pl.lit(has_stock_time_equal).alias("stock_time_equal_available"),
        pl.col("Date").str.slice(0, 4).alias("year"),
        pl.col("Date").str.slice(0, 6).alias("month"),
        pl.when(active)
        .then(
            pl.col("condition_sum_z") / pl.col("condition_n_z")
            - pl.col("control_sum_z") / pl.col("control_n_z")
        )
        .otherwise(None)
        .alias("daily_delta_z"),
        stock_delta.alias("daily_stock_equal_delta_z"),
        stock_time_delta.alias("daily_stock_time_equal_delta_z"),
    )


def _pooled_stats(active: pl.DataFrame) -> dict[str, float | int | None]:
    if active.is_empty():
        return {
            "event_rows": 0,
            "control_rows": 0,
            "condition_z_pooled": None,
            "control_z_pooled": None,
            "delta_z_pooled": None,
        }
    condition_count = int(active["condition_n_z"].sum())
    control_count = int(active["control_n_z"].sum())
    if condition_count == 0 or control_count == 0:
        raise SweepReportError("active rows cannot have zero pooled counts")
    condition_mean = float(active["condition_sum_z"].sum()) / condition_count
    control_mean = float(active["control_sum_z"].sum()) / control_count
    return {
        "event_rows": condition_count,
        "control_rows": control_count,
        "condition_z_pooled": condition_mean,
        "control_z_pooled": control_mean,
        "delta_z_pooled": condition_mean - control_mean,
    }


def _values(frame: pl.DataFrame, column: str) -> list[float]:
    return [float(value) for value in frame[column].drop_nulls().to_list()]


def _study_metadata(daily: pl.DataFrame) -> dict[str, Any]:
    metadata = {column: daily[column][0] for column in STUDY_COLUMNS}
    metadata.update(
        {
            column: daily[column][0] if column in daily.columns else None
            for column in OPTIONAL_STUDY_COLUMNS
        }
    )
    return metadata


def build_monthly_stats(
    daily: pl.DataFrame,
    *,
    hac_lag: int = DEFAULT_HAC_LAG,
) -> pl.DataFrame:
    """Build a complete candidate-by-observed-month statistics table."""

    validated = _validate_daily_stats(daily)
    enriched = _enrich_daily(validated)
    metadata = _study_metadata(enriched)
    candidates = sorted(enriched["candidate_id"].unique().to_list())
    months = sorted(enriched["month"].unique().to_list())
    session_dates_by_month = {
        month: set(
            enriched.filter(pl.col("month") == month)["Date"].unique().to_list()
        )
        for month in months
    }
    rows: list[dict[str, Any]] = []
    for candidate_id in candidates:
        candidate = enriched.filter(pl.col("candidate_id") == candidate_id)
        for month in months:
            subset = candidate.filter(pl.col("month") == month)
            active = subset.filter(pl.col("active"))
            values = _values(active, "daily_delta_z")
            equal = equal_weight_stats(values)
            hac = newey_west_mean_stats(values, hac_lag)
            pooled = _pooled_stats(active)
            sessions = len(session_dates_by_month[month])
            rows.append(
                {
                    **metadata,
                    "candidate_id": candidate_id,
                    "month": month,
                    "sessions": sessions,
                    "reported_days": len(subset),
                    "active_days": len(active),
                    "reporting_coverage": len(subset) / sessions,
                    "session_coverage": len(active) / sessions,
                    **pooled,
                    "equal_day_delta_z_mean": equal["mean"],
                    "equal_day_se": equal["se"],
                    "equal_day_t": equal["t"],
                    "equal_day_p_two_sided": equal["p"],
                    "hac_lag_effective": hac["lag"],
                    "hac_long_run_variance": hac["long_run_variance"],
                    "hac_se": hac["se"],
                    "hac_t": hac["t"],
                    "hac_p_two_sided": hac["p"],
                    "positive_month": (
                        equal["mean"] > 0.0 if equal["mean"] is not None else None
                    ),
                }
            )
    return pl.DataFrame(
        rows,
        schema_overrides={
            "stock_time_bucket": pl.String,
            "condition_z_pooled": pl.Float64,
            "control_z_pooled": pl.Float64,
            "delta_z_pooled": pl.Float64,
            "equal_day_delta_z_mean": pl.Float64,
            "equal_day_se": pl.Float64,
            "equal_day_t": pl.Float64,
            "equal_day_p_two_sided": pl.Float64,
            "hac_long_run_variance": pl.Float64,
            "hac_se": pl.Float64,
            "hac_t": pl.Float64,
            "hac_p_two_sided": pl.Float64,
            "positive_month": pl.Boolean,
        },
    ).sort("candidate_id", "month")


def _period_dates(all_dates: set[str], year: str | None) -> set[str]:
    if year is None:
        return all_dates
    return {date for date in all_dates if date.startswith(year)}


def build_period_summary(
    daily: pl.DataFrame,
    monthly: pl.DataFrame | None = None,
    *,
    hac_lag: int = DEFAULT_HAC_LAG,
) -> pl.DataFrame:
    """Build full, 2025, and 2026 summaries and full-period BH q-values."""

    validated = _validate_daily_stats(daily)
    enriched = _enrich_daily(validated)
    if monthly is None:
        monthly = build_monthly_stats(validated, hac_lag=hac_lag)
    metadata = _study_metadata(enriched)
    stock_equal_available = STOCK_DAY_EQUAL_COLUMN in validated.columns
    stock_time_equal_available = STOCK_TIME_EQUAL_COLUMN in validated.columns
    candidates = sorted(enriched["candidate_id"].unique().to_list())
    all_dates = set(enriched["Date"].unique().to_list())
    all_months = set(enriched["month"].unique().to_list())
    rows: list[dict[str, Any]] = []
    for candidate_id in candidates:
        candidate = enriched.filter(pl.col("candidate_id") == candidate_id)
        candidate_monthly = monthly.filter(pl.col("candidate_id") == candidate_id)
        for period, year in PERIODS:
            session_dates = _period_dates(all_dates, year)
            period_months = (
                all_months
                if year is None
                else {month for month in all_months if month.startswith(year)}
            )
            subset = (
                candidate
                if year is None
                else candidate.filter(pl.col("year") == year)
            )
            active = subset.filter(pl.col("active"))
            values = _values(active, "daily_delta_z")
            equal = equal_weight_stats(values)
            hac = newey_west_mean_stats(values, hac_lag)
            stock_values = (
                _values(subset, "daily_stock_equal_delta_z")
                if stock_equal_available
                else []
            )
            stock_equal = equal_weight_stats(stock_values)
            stock_hac = newey_west_mean_stats(stock_values, hac_lag)
            stock_time_values = (
                _values(subset, "daily_stock_time_equal_delta_z")
                if stock_time_equal_available
                else []
            )
            stock_time_equal = equal_weight_stats(stock_time_values)
            stock_time_hac = newey_west_mean_stats(
                stock_time_values,
                hac_lag,
            )
            pooled = _pooled_stats(active)
            month_subset = candidate_monthly.filter(
                pl.col("month").is_in(sorted(period_months))
            )
            active_months = month_subset.filter(pl.col("active_days") > 0)
            month_values = _values(active_months, "equal_day_delta_z_mean")
            month_stats = equal_weight_stats(month_values)
            positive_months = sum(value > 0.0 for value in month_values)
            sessions = len(session_dates)
            reported_days = len(subset)
            active_days = len(active)
            rows.append(
                {
                    **metadata,
                    "candidate_id": candidate_id,
                    "period": period,
                    "sessions": sessions,
                    "reported_days": reported_days,
                    "active_days": active_days,
                    "reporting_coverage": (
                        reported_days / sessions if sessions else None
                    ),
                    "session_coverage": active_days / sessions if sessions else None,
                    **pooled,
                    "events_per_active_day": (
                        pooled["event_rows"] / active_days if active_days else None
                    ),
                    "equal_day_delta_z_mean": equal["mean"],
                    "equal_day_se": equal["se"],
                    "equal_day_t": equal["t"],
                    "equal_day_p_two_sided": equal["p"],
                    "hac_lag_requested": hac_lag,
                    "hac_lag_effective": hac["lag"],
                    "hac_long_run_variance": hac["long_run_variance"],
                    "hac_se": hac["se"],
                    "hac_t": hac["t"],
                    "hac_p_two_sided": hac["p"],
                    "raw_p_two_sided": hac["p"],
                    "stock_day_equal_available": stock_equal_available,
                    "stock_day_equal_days": (
                        len(stock_values) if stock_equal_available else None
                    ),
                    "stock_day_equal_delta_z_mean": stock_equal["mean"],
                    "stock_day_equal_se": stock_equal["se"],
                    "stock_day_equal_t": stock_equal["t"],
                    "stock_day_equal_p_two_sided": stock_equal["p"],
                    "stock_day_equal_hac_lag_effective": (
                        stock_hac["lag"] if stock_equal_available else None
                    ),
                    "stock_day_equal_hac_long_run_variance": stock_hac[
                        "long_run_variance"
                    ],
                    "stock_day_equal_hac_se": stock_hac["se"],
                    "stock_day_equal_hac_t": stock_hac["t"],
                    "stock_day_equal_hac_p_two_sided": stock_hac["p"],
                    "stock_day_equal_bh_q_full": None,
                    "stock_time_equal_available": stock_time_equal_available,
                    "stock_time_equal_days": (
                        len(stock_time_values)
                        if stock_time_equal_available
                        else None
                    ),
                    "stock_time_equal_delta_z_mean": stock_time_equal["mean"],
                    "stock_time_equal_se": stock_time_equal["se"],
                    "stock_time_equal_t": stock_time_equal["t"],
                    "stock_time_equal_p_two_sided": stock_time_equal["p"],
                    "stock_time_equal_hac_lag_effective": (
                        stock_time_hac["lag"]
                        if stock_time_equal_available
                        else None
                    ),
                    "stock_time_equal_hac_long_run_variance": stock_time_hac[
                        "long_run_variance"
                    ],
                    "stock_time_equal_hac_se": stock_time_hac["se"],
                    "stock_time_equal_hac_t": stock_time_hac["t"],
                    "stock_time_equal_hac_p_two_sided": stock_time_hac["p"],
                    "stock_time_equal_bh_q_full": None,
                    "calendar_months": len(period_months),
                    "active_months": len(active_months),
                    "monthly_delta_z_mean": month_stats["mean"],
                    "monthly_se": month_stats["se"],
                    "monthly_t": month_stats["t"],
                    "monthly_p_two_sided": month_stats["p"],
                    "positive_months": positive_months,
                    "positive_month_ratio": (
                        positive_months / len(active_months)
                        if len(active_months)
                        else None
                    ),
                    "bh_q_full": None,
                }
            )

    full_p_values = {
        row["candidate_id"]: row["raw_p_two_sided"]
        for row in rows
        if row["period"] == "full"
    }
    full_q_values = benjamini_hochberg(full_p_values)
    for row in rows:
        if row["period"] == "full":
            row["bh_q_full"] = full_q_values[row["candidate_id"]]

    full_stock_p_values = {
        row["candidate_id"]: row["stock_day_equal_hac_p_two_sided"]
        for row in rows
        if row["period"] == "full"
    }
    full_stock_q_values = benjamini_hochberg(full_stock_p_values)
    for row in rows:
        if row["period"] == "full":
            row["stock_day_equal_bh_q_full"] = full_stock_q_values[
                row["candidate_id"]
            ]

    full_stock_time_p_values = {
        row["candidate_id"]: row["stock_time_equal_hac_p_two_sided"]
        for row in rows
        if row["period"] == "full"
    }
    full_stock_time_q_values = benjamini_hochberg(
        full_stock_time_p_values
    )
    for row in rows:
        if row["period"] == "full":
            row["stock_time_equal_bh_q_full"] = full_stock_time_q_values[
                row["candidate_id"]
            ]

    summary = pl.DataFrame(
        rows,
        schema_overrides={
            "stock_time_bucket": pl.String,
            "reporting_coverage": pl.Float64,
            "session_coverage": pl.Float64,
            "condition_z_pooled": pl.Float64,
            "control_z_pooled": pl.Float64,
            "delta_z_pooled": pl.Float64,
            "events_per_active_day": pl.Float64,
            "equal_day_delta_z_mean": pl.Float64,
            "equal_day_se": pl.Float64,
            "equal_day_t": pl.Float64,
            "equal_day_p_two_sided": pl.Float64,
            "hac_long_run_variance": pl.Float64,
            "hac_se": pl.Float64,
            "hac_t": pl.Float64,
            "hac_p_two_sided": pl.Float64,
            "raw_p_two_sided": pl.Float64,
            "stock_day_equal_available": pl.Boolean,
            "stock_day_equal_days": pl.Int64,
            "stock_day_equal_delta_z_mean": pl.Float64,
            "stock_day_equal_se": pl.Float64,
            "stock_day_equal_t": pl.Float64,
            "stock_day_equal_p_two_sided": pl.Float64,
            "stock_day_equal_hac_lag_effective": pl.Int64,
            "stock_day_equal_hac_long_run_variance": pl.Float64,
            "stock_day_equal_hac_se": pl.Float64,
            "stock_day_equal_hac_t": pl.Float64,
            "stock_day_equal_hac_p_two_sided": pl.Float64,
            "stock_day_equal_bh_q_full": pl.Float64,
            "stock_time_equal_available": pl.Boolean,
            "stock_time_equal_days": pl.Int64,
            "stock_time_equal_delta_z_mean": pl.Float64,
            "stock_time_equal_se": pl.Float64,
            "stock_time_equal_t": pl.Float64,
            "stock_time_equal_p_two_sided": pl.Float64,
            "stock_time_equal_hac_lag_effective": pl.Int64,
            "stock_time_equal_hac_long_run_variance": pl.Float64,
            "stock_time_equal_hac_se": pl.Float64,
            "stock_time_equal_hac_t": pl.Float64,
            "stock_time_equal_hac_p_two_sided": pl.Float64,
            "stock_time_equal_bh_q_full": pl.Float64,
            "monthly_delta_z_mean": pl.Float64,
            "monthly_se": pl.Float64,
            "monthly_t": pl.Float64,
            "monthly_p_two_sided": pl.Float64,
            "positive_month_ratio": pl.Float64,
            "bh_q_full": pl.Float64,
        },
    )
    period_order = pl.col("period").replace_strict(
        {"full": 0, "2025": 1, "2026": 2},
        return_dtype=pl.Int8,
    )
    return summary.with_columns(period_order.alias("_period_order")).sort(
        "_period_order", "candidate_id"
    ).drop("_period_order")


def build_report(
    daily: pl.DataFrame,
    *,
    hac_lag: int = DEFAULT_HAC_LAG,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return period and complete monthly reports for one daily sweep."""

    if hac_lag < 0:
        raise ValueError("hac_lag cannot be negative")
    validated = _validate_daily_stats(daily)
    monthly = build_monthly_stats(validated, hac_lag=hac_lag)
    summary = build_period_summary(validated, monthly, hac_lag=hac_lag)
    return summary, monthly


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        frame.write_parquet(temp_path, compression="zstd")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(
            payload,
            handle,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_report(
    daily: pl.DataFrame,
    *,
    input_paths: tuple[Path, ...],
    output_dir: Path,
    hac_lag: int = DEFAULT_HAC_LAG,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build and atomically persist parquet and JSON report artifacts."""

    summary, monthly = build_report(daily, hac_lag=hac_lag)
    _atomic_write_parquet(summary, output_dir / "period_summary.parquet")
    _atomic_write_parquet(monthly, output_dir / "monthly_stats.parquet")
    validated = _validate_daily_stats(daily)
    resolved_sources = [str(path.resolve()) for path in input_paths]
    _atomic_write_json(
        {
            "contract": "filllots_at_low_bias_report/v2",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": resolved_sources[0] if len(resolved_sources) == 1 else None,
            "sources": resolved_sources,
            "input_rows": len(validated),
            "study": _study_metadata(validated),
            "candidate_count": validated["candidate_id"].n_unique(),
            "session_count": validated["Date"].n_unique(),
            "inference": {
                "equal_day_p": "two-sided Student t",
                "hac_kernel": "Bartlett",
                "hac_lag_requested": hac_lag,
                "hac_lag_unit": "ordered active observations after null removal",
                "raw_p": "two-sided asymptotic normal p for HAC t",
                "bh_families": (
                    "three separate full-period candidate families: pooled/equal-day, "
                    "stock-day equal, and stock-time equal"
                ),
                "stock_day_equal": (
                    "runner-provided equal-weight mean across active QuoteCode "
                    "of within-QuoteCode condition mean z minus the same-QuoteCode "
                    "full-pool mean z; reported with Student t and HAC"
                ),
                "stock_time_equal": (
                    "runner-provided condition-minus-full-pool mean z within "
                    "QuoteCode and fixed time bucket; active buckets are equal "
                    "weighted within QuoteCode, then active QuoteCode are equal "
                    "weighted; reported with Student t and HAC"
                ),
            },
            "period_summary": summary.to_dicts(),
            "monthly_stats": monthly.to_dicts(),
        },
        output_dir / "report.json",
    )
    return summary, monthly


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report inference statistics from daily FillLots_atLow sweeps"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="one or more daily_stats.parquet paths or artifact directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: REPORT_INPUT_PARENT/report",
    )
    parser.add_argument("--hac-lag", type=int, default=DEFAULT_HAC_LAG)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.hac_lag < 0:
        parser.error("--hac-lag cannot be negative")
    if len(args.inputs) > 1 and args.output_dir is None:
        parser.error("--output-dir is required with multiple inputs")
    return args


def _resolve_input_paths(inputs: list[Path]) -> tuple[Path, ...]:
    paths = tuple(
        path / "daily_stats.parquet" if path.is_dir() else path
        for path in inputs
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return paths


def _read_daily_inputs(input_paths: tuple[Path, ...]) -> pl.DataFrame:
    frames = [pl.read_parquet(path) for path in input_paths]
    return frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_paths = _resolve_input_paths(args.inputs)
    output_dir = args.output_dir or input_paths[0].parent / "report"
    daily = _read_daily_inputs(input_paths)
    summary, monthly = write_report(
        daily,
        input_paths=input_paths,
        output_dir=output_dir,
        hac_lag=args.hac_lag,
    )
    if not args.quiet:
        display_columns = [
            "candidate_id",
            "period",
            "active_days",
            "session_coverage",
            "delta_z_pooled",
            "equal_day_delta_z_mean",
            "equal_day_t",
            "hac_t",
            "raw_p_two_sided",
            "bh_q_full",
            "stock_day_equal_delta_z_mean",
            "stock_day_equal_hac_t",
            "stock_day_equal_bh_q_full",
            "stock_time_equal_delta_z_mean",
            "stock_time_equal_hac_t",
            "stock_time_equal_bh_q_full",
            "positive_month_ratio",
        ]
        with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=5):
            print(summary.select(display_columns))
        print(f"Monthly rows: {len(monthly)}")
        print(f"Report directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
