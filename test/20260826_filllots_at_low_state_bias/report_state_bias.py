"""Report long/short bias for the unmodified FillLots_atLow sign states.

This adapter consumes the additive daily artifacts written by ``run_sweep.py``
with ``--candidate-registry state``.  It does not reload market data or reshape
the event.  The primary estimand remains condition minus the complete shared
eligible pool; same-stock and same-stock-time estimates are retained only as
attribution diagnostics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl

from src.research.timeSeries.framework.profiles.benchmark import (
    LONG_TAKER_TP109_V1,
)
from src.research.timeSeries.framework.profiles.filllots_at_low import (
    FILLLOTS_AT_LOW_STATE_FAMILY_ID,
    FILLLOTS_AT_LOW_STATE_FAMILY_VERSION,
    STATE_CANDIDATE_IDS,
)
from src.research.timeSeries.framework.profiles.negfill import (
    SHORT_TAKER_STOP108_V1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[5]
FRAMEWORK_STUDY_DIR = Path(__file__).resolve().parents[1] / (
    "20260825_filllots_at_low_framework"
)
REPORTER_PATH = FRAMEWORK_STUDY_DIR / "report_sweep.py"
STATE_SWEEP_CONTRACT_ID = "filllots_at_low_state_bias_sweep/v1"
REPORT_CONTRACT_ID = "filllots_at_low_state_bias_report/v1"
STATE_REGISTRY_ID = "state"
EXPECTED_LABELS = {
    SHORT_TAKER_STOP108_V1.profile_id: (
        "short",
        SHORT_TAKER_STOP108_V1.fingerprint,
    ),
    LONG_TAKER_TP109_V1.profile_id: (
        "long",
        LONG_TAKER_TP109_V1.fingerprint,
    ),
}
PERIODS: tuple[tuple[str, str | None], ...] = (
    ("full", None),
    ("2025", "2025"),
    ("2026", "2026"),
)
HAC_SENSITIVITY_LAGS = (0, 5, 10, 20)


class StateBiasReportError(ValueError):
    """Raised when state-recovery inputs mix incompatible study contracts."""


def _load_reporter() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "filllots_at_low_sweep_report_for_state_bias",
        REPORTER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load report module: {REPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPORTER = _load_reporter()


def _resolve_daily_path(path: Path) -> Path:
    resolved = path / "daily_stats.parquet" if path.is_dir() else path
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def _manifest_path(path: Path) -> Path:
    directory = path if path.is_dir() else path.parent
    manifest = directory / "run_manifest.json"
    if not manifest.exists():
        raise StateBiasReportError(
            f"input is missing its run manifest: {manifest}"
        )
    return manifest


def _load_and_validate_manifest(
    path: Path,
    daily: pl.DataFrame,
) -> dict[str, Any]:
    manifest_path = _manifest_path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateBiasReportError(
            f"cannot read run manifest {manifest_path}: {exc}"
        ) from exc
    expected = {
        "contract": STATE_SWEEP_CONTRACT_ID,
        "candidate_registry": STATE_REGISTRY_ID,
        "family_id": FILLLOTS_AT_LOW_STATE_FAMILY_ID,
        "family_version": FILLLOTS_AT_LOW_STATE_FAMILY_VERSION,
        "primary_estimand": "shared_pool",
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            raise StateBiasReportError(
                f"manifest {field}={payload.get(field)!r}, "
                f"expected {expected_value!r}"
            )
    if tuple(sorted(payload.get("candidate_ids", []))) != tuple(
        sorted(STATE_CANDIDATE_IDS)
    ):
        raise StateBiasReportError("manifest candidate set is not the state registry")
    failures = payload.get("failures")
    if failures != []:
        raise StateBiasReportError(
            f"manifest contains failures or omits the failure audit: {failures!r}"
        )
    sessions = payload.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise StateBiasReportError("manifest sessions must be a non-empty list")
    if len(sessions) != len(set(sessions)):
        raise StateBiasReportError("manifest contains duplicate sessions")
    actual_dates = set(daily["Date"].cast(pl.String).unique().to_list())
    if actual_dates != set(sessions):
        raise StateBiasReportError(
            "daily output does not cover every requested manifest session"
        )
    identities = payload.get("source_identities")
    if not isinstance(identities, dict) or set(identities) != set(sessions):
        raise StateBiasReportError(
            "manifest source identities do not cover every requested session"
        )
    daily_manifest_pairs = {
        "label_profile_id": "label_profile_id",
        "label_profile_hash": "label_profile_hash",
        "pool_id": "pool_id",
        "pool_version": "pool_version",
    }
    for manifest_field, daily_column in daily_manifest_pairs.items():
        if payload.get(manifest_field) != _one_value(daily, daily_column):
            raise StateBiasReportError(
                f"manifest {manifest_field} disagrees with daily output"
            )
    return payload


def _one_value(frame: pl.DataFrame, column: str) -> Any:
    if column not in frame.columns:
        raise StateBiasReportError(f"input missing metadata column {column}")
    values = frame[column].unique().to_list()
    if len(values) != 1:
        raise StateBiasReportError(f"input mixes {column}: {values}")
    return values[0]


def _validate_direction_input(daily: pl.DataFrame) -> str:
    # Reuse the inference module's complete daily-stat validation first.
    REPORTER.build_report(daily)
    expected = {
        "contract_id": STATE_SWEEP_CONTRACT_ID,
        "family_id": FILLLOTS_AT_LOW_STATE_FAMILY_ID,
        "family_version": FILLLOTS_AT_LOW_STATE_FAMILY_VERSION,
        "control_mode": "full",
        "control_sample_size": 0,
        "primary_estimand": "shared_pool",
    }
    for column, expected_value in expected.items():
        actual = _one_value(daily, column)
        if actual != expected_value:
            raise StateBiasReportError(
                f"{column}={actual!r}, expected {expected_value!r}"
            )
    candidates = tuple(sorted(daily["candidate_id"].unique().to_list()))
    if candidates != tuple(sorted(STATE_CANDIDATE_IDS)):
        raise StateBiasReportError(
            f"candidate set={candidates}, expected {STATE_CANDIDATE_IDS}"
        )
    label_profile_id = str(_one_value(daily, "label_profile_id"))
    if label_profile_id not in EXPECTED_LABELS:
        raise StateBiasReportError(f"unexpected label profile {label_profile_id}")
    direction, expected_hash = EXPECTED_LABELS[label_profile_id]
    actual_hash = str(_one_value(daily, "label_profile_hash"))
    if actual_hash != expected_hash:
        raise StateBiasReportError(
            f"{label_profile_id} hash={actual_hash}, expected {expected_hash}"
        )
    required = {
        "eligible_pool_rows",
        "condition_n",
        "condition_n_z",
        "condition_unique_quotes",
        "selection_rate",
        "condition_sum_raw_bp",
        "condition_sum_residual_bp",
        "condition_sum_z2",
        "control_n",
        "control_n_z",
        "control_sum_raw_bp",
        "control_sum_residual_bp",
        "control_sum_z2",
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise StateBiasReportError(f"input is missing state-report columns {missing}")
    for column in (
        "eligible_pool_rows",
        "condition_n",
        "condition_n_z",
        "condition_unique_quotes",
        "control_n",
        "control_n_z",
    ):
        invalid = daily.filter(pl.col(column).is_null() | (pl.col(column) < 0))
        if not invalid.is_empty():
            raise StateBiasReportError(f"{column} must be nonnegative")
    for column in (
        "condition_sum_raw_bp",
        "condition_sum_residual_bp",
        "condition_sum_z2",
        "control_sum_raw_bp",
        "control_sum_residual_bp",
        "control_sum_z2",
    ):
        invalid = daily.filter(
            pl.col(column).is_null() | ~pl.col(column).is_finite()
        )
        if not invalid.is_empty():
            raise StateBiasReportError(f"{column} must be finite")
    negative_second_moment = daily.filter(
        (pl.col("condition_sum_z2") < -1.0e-12)
        | (pl.col("control_sum_z2") < -1.0e-12)
    )
    if not negative_second_moment.is_empty():
        raise StateBiasReportError("sum_z2 cannot be negative")
    invalid_full_control = daily.filter(
        (pl.col("control_n") != pl.col("eligible_pool_rows"))
        | (pl.col("condition_n") > pl.col("eligible_pool_rows"))
        | (pl.col("condition_unique_quotes") > pl.col("condition_n"))
    )
    if not invalid_full_control.is_empty():
        raise StateBiasReportError("full-pool cohort counts are inconsistent")
    invalid_rate = daily.filter(
        pl.when(pl.col("eligible_pool_rows") > 0)
        .then(
            pl.col("selection_rate").is_null()
            | ~pl.col("selection_rate").is_finite()
            | (
                (
                    pl.col("selection_rate")
                    - pl.col("condition_n") / pl.col("eligible_pool_rows")
                ).abs()
                > 1.0e-12
            )
        )
        # The runner's legacy zero-denominator representation is NaN.  Treat
        # null/NaN as the only valid empty-pool values; finite/inf would imply
        # a fabricated selection rate.
        .otherwise(
            pl.col("selection_rate").is_not_null()
            & ~pl.col("selection_rate").is_nan()
        )
    )
    if not invalid_rate.is_empty():
        raise StateBiasReportError("selection_rate disagrees with cohort counts")
    return direction


def _finite_values(frame: pl.DataFrame, column: str) -> list[float]:
    return [
        float(value)
        for value in frame[column].drop_nulls().to_list()
        if math.isfinite(float(value))
    ]


def _pooled_mean(
    frame: pl.DataFrame,
    prefix: str,
    value: str,
) -> float | None:
    count = int(frame[f"{prefix}_n_z"].sum())
    if count <= 0:
        return None
    return float(frame[f"{prefix}_sum_{value}"].sum()) / count


def _pooled_z_std(frame: pl.DataFrame, prefix: str) -> float | None:
    count = int(frame[f"{prefix}_n_z"].sum())
    mean = _pooled_mean(frame, prefix, "z")
    if count <= 0 or mean is None:
        return None
    second_moment = float(frame[f"{prefix}_sum_z2"].sum()) / count
    return math.sqrt(max(0.0, second_moment - mean * mean))


def _mean(frame: pl.DataFrame, column: str) -> float | None:
    value = frame[column].mean()
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _median(frame: pl.DataFrame, column: str) -> float | None:
    value = frame[column].median()
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    return float(
        pl.Series("value", values).quantile(
            probability,
            interpolation="linear",
        )
    )


def _distribution_rows(
    daily: pl.DataFrame,
    *,
    direction: str,
    hac_lag: int,
) -> pl.DataFrame:
    active = (pl.col("condition_n_z") > 0) & (pl.col("control_n_z") > 0)
    enriched = daily.with_columns(
        pl.col("Date").str.slice(0, 4).alias("year"),
        pl.col("Date").str.slice(0, 6).alias("month"),
        pl.when(active)
        .then(
            pl.col("condition_sum_z") / pl.col("condition_n_z")
            - pl.col("control_sum_z") / pl.col("control_n_z")
        )
        .alias("daily_delta_z"),
        pl.when(active)
        .then(
            pl.col("condition_sum_raw_bp") / pl.col("condition_n_z")
            - pl.col("control_sum_raw_bp") / pl.col("control_n_z")
        )
        .alias("daily_delta_raw_bp"),
        pl.when(active)
        .then(
            pl.col("condition_sum_residual_bp") / pl.col("condition_n_z")
            - pl.col("control_sum_residual_bp") / pl.col("control_n_z")
        )
        .alias("daily_delta_residual_bp"),
    )
    metadata_columns = (
        "contract_id",
        "family_id",
        "family_version",
        "label_profile_id",
        "label_profile_hash",
        "pool_id",
        "pool_version",
        "control_mode",
        "control_sample_size",
        "primary_estimand",
        "stock_time_bucket",
    )
    metadata = {
        column: _one_value(enriched, column) for column in metadata_columns
    }
    rows: list[dict[str, Any]] = []
    for candidate_id in STATE_CANDIDATE_IDS:
        candidate = enriched.filter(
            pl.col("candidate_id") == candidate_id
        ).sort("Date")
        for period, year in PERIODS:
            subset = (
                candidate
                if year is None
                else candidate.filter(pl.col("year") == year)
            )
            active_subset = subset.filter(
                (pl.col("condition_n_z") > 0)
                & (pl.col("control_n_z") > 0)
            ).sort("Date")
            z_values = _finite_values(active_subset, "daily_delta_z")
            raw_values = _finite_values(
                active_subset,
                "daily_delta_raw_bp",
            )
            residual_values = _finite_values(
                active_subset,
                "daily_delta_residual_bp",
            )
            z_hac = REPORTER.newey_west_mean_stats(z_values, hac_lag)
            raw_hac = REPORTER.newey_west_mean_stats(raw_values, hac_lag)
            residual_hac = REPORTER.newey_west_mean_stats(
                residual_values,
                hac_lag,
            )
            monthly = active_subset.group_by("month").agg(
                pl.col("daily_delta_z").mean().alias("monthly_delta_z")
            )
            # Match the base reporter: control rows from dates where the
            # condition is inactive are not part of this candidate estimand.
            condition_z = _pooled_mean(active_subset, "condition", "z")
            control_z = _pooled_mean(active_subset, "control", "z")
            condition_raw = _pooled_mean(
                active_subset,
                "condition",
                "raw_bp",
            )
            control_raw = _pooled_mean(
                active_subset,
                "control",
                "raw_bp",
            )
            condition_residual = _pooled_mean(
                active_subset,
                "condition",
                "residual_bp",
            )
            control_residual = _pooled_mean(
                active_subset,
                "control",
                "residual_bp",
            )
            rows.append(
                {
                    **metadata,
                    "direction": direction,
                    "candidate_id": candidate_id,
                    "period": period,
                    "sessions": len(subset),
                    "active_days": len(z_values),
                    "event_rows": int(active_subset["condition_n_z"].sum()),
                    "events_per_day": _mean(
                        active_subset,
                        "condition_n_z",
                    ),
                    "stocks_per_day": _mean(
                        active_subset,
                        "condition_unique_quotes",
                    ),
                    "selection_rate_mean": _mean(
                        active_subset,
                        "selection_rate",
                    ),
                    "selection_rate_median": _median(
                        active_subset,
                        "selection_rate",
                    ),
                    "condition_z_pooled": condition_z,
                    "condition_z_std_pooled": _pooled_z_std(
                        active_subset,
                        "condition",
                    ),
                    "control_z_pooled": control_z,
                    "control_z_std_pooled": _pooled_z_std(
                        active_subset,
                        "control",
                    ),
                    "delta_z_pooled": _difference(condition_z, control_z),
                    "daily_delta_z_mean": _mean(
                        active_subset,
                        "daily_delta_z",
                    ),
                    "daily_delta_z_std": (
                        float(active_subset["daily_delta_z"].std())
                        if len(active_subset) > 1
                        else None
                    ),
                    "daily_delta_z_q05": _quantile(z_values, 0.05),
                    "daily_delta_z_q25": _quantile(z_values, 0.25),
                    "daily_delta_z_median": _quantile(z_values, 0.50),
                    "daily_delta_z_q75": _quantile(z_values, 0.75),
                    "daily_delta_z_q95": _quantile(z_values, 0.95),
                    "daily_delta_z_hac_lag": z_hac["lag"],
                    "daily_delta_z_hac_t": z_hac["t"],
                    "daily_delta_z_hac_p": z_hac["p"],
                    "positive_days": sum(value > 0.0 for value in z_values),
                    "positive_months": int(
                        (monthly["monthly_delta_z"] > 0.0).sum()
                    ),
                    "months": len(monthly),
                    "condition_raw_bp_pooled": condition_raw,
                    "control_raw_bp_pooled": control_raw,
                    "delta_raw_bp_pooled": _difference(
                        condition_raw,
                        control_raw,
                    ),
                    "daily_delta_raw_bp_mean": _mean(
                        active_subset,
                        "daily_delta_raw_bp",
                    ),
                    "daily_delta_raw_bp_hac_t": raw_hac["t"],
                    "daily_delta_raw_bp_hac_p": raw_hac["p"],
                    "condition_residual_bp_pooled": condition_residual,
                    "control_residual_bp_pooled": control_residual,
                    "delta_residual_bp_pooled": _difference(
                        condition_residual,
                        control_residual,
                    ),
                    "daily_delta_residual_bp_mean": _mean(
                        active_subset,
                        "daily_delta_residual_bp",
                    ),
                    "daily_delta_residual_bp_hac_t": residual_hac["t"],
                    "daily_delta_residual_bp_hac_p": residual_hac["p"],
                }
            )
    return pl.DataFrame(rows)


def _daily_effect_rows(
    daily: pl.DataFrame,
    *,
    direction: str,
) -> pl.DataFrame:
    active = (pl.col("condition_n_z") > 0) & (pl.col("control_n_z") > 0)
    return (
        daily.sort("Date", "candidate_id")
        .with_columns(
            pl.lit(direction).alias("direction"),
            active.alias("active"),
            pl.when(active)
            .then(pl.col("condition_sum_z") / pl.col("condition_n_z"))
            .alias("condition_mean_z"),
            pl.when(active)
            .then(pl.col("control_sum_z") / pl.col("control_n_z"))
            .alias("control_mean_z"),
            pl.when(active)
            .then(
                pl.col("condition_sum_z") / pl.col("condition_n_z")
                - pl.col("control_sum_z") / pl.col("control_n_z")
            )
            .alias("daily_delta_z"),
            pl.when(active)
            .then(
                pl.col("condition_sum_raw_bp") / pl.col("condition_n_z")
                - pl.col("control_sum_raw_bp") / pl.col("control_n_z")
            )
            .alias("daily_delta_raw_bp"),
            pl.when(active)
            .then(
                pl.col("condition_sum_residual_bp")
                / pl.col("condition_n_z")
                - pl.col("control_sum_residual_bp")
                / pl.col("control_n_z")
            )
            .alias("daily_delta_residual_bp"),
        )
        .select(
            "Date",
            "direction",
            "candidate_id",
            "active",
            "eligible_pool_rows",
            "condition_n",
            "condition_n_z",
            "control_n_z",
            "condition_unique_quotes",
            "selection_rate",
            "condition_mean_z",
            "control_mean_z",
            "daily_delta_z",
            "daily_delta_raw_bp",
            "daily_delta_residual_bp",
            "stock_day_equal_delta_z",
            "stock_time_equal_delta_z",
        )
    )


def _partition_rows(daily: pl.DataFrame, *, direction: str) -> pl.DataFrame:
    dates = daily["Date"].unique().to_list()
    coverage = daily.select("Date", "candidate_id").unique()
    expected_rows = len(dates) * len(STATE_CANDIDATE_IDS)
    if len(coverage) != expected_rows:
        raise StateBiasReportError(
            "state partition requires every candidate on every session: "
            f"rows={len(coverage)}, expected={expected_rows}"
        )
    inconsistent_pool = (
        daily.group_by("Date")
        .agg(pl.col("eligible_pool_rows").n_unique().alias("n_pool_values"))
        .filter(pl.col("n_pool_values") != 1)
    )
    if not inconsistent_pool.is_empty():
        raise StateBiasReportError(
            "one session has inconsistent eligible-pool counts: "
            f"{inconsistent_pool.head(3).to_dicts()}"
        )
    counts = daily.select(
        "Date",
        "candidate_id",
        "eligible_pool_rows",
        "condition_n",
    ).pivot(
        on="candidate_id",
        index=["Date", "eligible_pool_rows"],
        values="condition_n",
    )
    missing = sorted(set(STATE_CANDIDATE_IDS) - set(counts.columns))
    if missing:
        raise StateBiasReportError(f"partition is missing candidates {missing}")
    partition = counts.with_columns(
        pl.lit(direction).alias("direction"),
        (
            pl.col("eligible_pool_rows")
            - pl.col("negative_state")
            - pl.col("positive_state")
        ).alias("non_signed_state_rows"),
        pl.when(pl.col("eligible_pool_rows") > 0)
        .then(pl.col("negative_state") / pl.col("eligible_pool_rows"))
        .alias("negative_state_rate"),
        pl.when(pl.col("eligible_pool_rows") > 0)
        .then(pl.col("positive_state") / pl.col("eligible_pool_rows"))
        .alias("positive_state_rate"),
    ).sort("Date")
    invalid = partition.filter(pl.col("non_signed_state_rows") < 0)
    if not invalid.is_empty():
        raise StateBiasReportError(
            "state partitions overlap or exceed the eligible pool: "
            f"{invalid.head(3).to_dicts()}"
        )
    return partition


def _add_joint_primary_adjustments(summary: pl.DataFrame) -> pl.DataFrame:
    """Adjust the four full-period direction-by-state shared HAC tests.

    The base reporter's ``bh_q_full`` is a two-candidate family within one
    direction.  This report also exposes one conservative joint family across
    both directions so an omnibus "any bullish/bearish cell" claim is not made
    from four unadjusted searches.  Holm is valid under arbitrary dependence,
    which matters because the two sign states are near complements.
    """

    rows = summary.to_dicts()
    full_indexes = [
        index for index, row in enumerate(rows) if row["period"] == "full"
    ]
    if len(full_indexes) != len(EXPECTED_LABELS) * len(STATE_CANDIDATE_IDS):
        raise StateBiasReportError(
            "joint primary family must contain four full-period rows"
        )
    family_size = len(full_indexes)
    finite: list[tuple[int, float]] = []
    for index, row in enumerate(rows):
        row["joint_primary_family"] = (
            "full_shared_hac_direction_by_signed_state_4"
        )
        row["joint_bonferroni_p_full"] = None
        row["joint_holm_p_full"] = None
        if index not in full_indexes:
            continue
        value = row["hac_p_two_sided"]
        if value is not None and math.isfinite(float(value)):
            finite.append((index, float(value)))
    running = 0.0
    for rank, (index, p_value) in enumerate(
        sorted(finite, key=lambda item: item[1])
    ):
        rows[index]["joint_bonferroni_p_full"] = min(
            1.0,
            p_value * family_size,
        )
        running = max(running, (family_size - rank) * p_value)
        rows[index]["joint_holm_p_full"] = min(1.0, running)
    return pl.DataFrame(rows)


def build_report(
    inputs: tuple[Path, ...],
    *,
    hac_lag: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if len(inputs) != 2:
        raise StateBiasReportError("provide exactly one short and one long input")
    direction_summaries: list[pl.DataFrame] = []
    direction_monthly: list[pl.DataFrame] = []
    direction_partitions: list[pl.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    seen_directions: set[str] = set()
    date_sets: list[set[str]] = []
    for path in inputs:
        daily = pl.read_parquet(_resolve_daily_path(path)).sort(
            "Date",
            "candidate_id",
        )
        manifests.append(_load_and_validate_manifest(path, daily))
        direction = _validate_direction_input(daily)
        if direction in seen_directions:
            raise StateBiasReportError(f"duplicate {direction} input")
        seen_directions.add(direction)
        date_sets.append(set(daily["Date"].unique().to_list()))
        core, monthly = REPORTER.build_report(daily, hac_lag=hac_lag)
        extra = _distribution_rows(
            daily,
            direction=direction,
            hac_lag=hac_lag,
        )
        core = core.with_columns(pl.lit(direction).alias("direction"))
        join_keys = [
            "contract_id",
            "family_id",
            "family_version",
            "label_profile_id",
            "label_profile_hash",
            "pool_id",
            "pool_version",
            "control_mode",
            "control_sample_size",
            "primary_estimand",
            "stock_time_bucket",
            "direction",
            "candidate_id",
            "period",
        ]
        duplicate_extra = sorted(
            (set(core.columns) & set(extra.columns)) - set(join_keys)
        )
        direction_summaries.append(
            core.join(
                extra.drop(duplicate_extra),
                on=join_keys,
                how="inner",
                validate="1:1",
            )
        )
        direction_monthly.append(
            monthly.with_columns(pl.lit(direction).alias("direction"))
        )
        direction_partitions.append(
            _partition_rows(daily, direction=direction)
        )
    if seen_directions != {"short", "long"}:
        raise StateBiasReportError(f"directions={seen_directions}")
    if date_sets[0] != date_sets[1]:
        raise StateBiasReportError("short and long inputs use different sessions")
    if manifests[0]["sessions"] != manifests[1]["sessions"]:
        raise StateBiasReportError(
            "short and long manifests request different ordered sessions"
        )
    if manifests[0]["source_identities"] != manifests[1]["source_identities"]:
        raise StateBiasReportError(
            "short and long manifests do not share identical source identities"
        )
    summary = pl.concat(direction_summaries, how="diagonal_relaxed").sort(
        "direction",
        "period",
        "candidate_id",
    )
    summary = _add_joint_primary_adjustments(summary)
    return (
        summary,
        pl.concat(direction_monthly, how="diagonal_relaxed").sort(
            "direction",
            "month",
            "candidate_id",
        ),
        pl.concat(direction_partitions, how="vertical").sort(
            "direction",
            "Date",
        ),
    )


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        frame.write_parquet(temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
        temporary = Path(handle.name)
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
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_report(
    inputs: tuple[Path, ...],
    *,
    output_dir: Path,
    hac_lag: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    summary, monthly, partition = build_report(inputs, hac_lag=hac_lag)
    daily_effects: list[pl.DataFrame] = []
    for path in inputs:
        daily = pl.read_parquet(_resolve_daily_path(path))
        label_profile_id = str(_one_value(daily, "label_profile_id"))
        direction = EXPECTED_LABELS[label_profile_id][0]
        daily_effects.append(
            _daily_effect_rows(daily, direction=direction)
        )
    daily_effect = pl.concat(daily_effects, how="vertical").sort(
        "direction",
        "Date",
        "candidate_id",
    )
    sensitivity_frames: list[pl.DataFrame] = []
    sensitivity_lags = sorted(set((*HAC_SENSITIVITY_LAGS, hac_lag)))
    for lag in sensitivity_lags:
        lag_summary = (
            summary if lag == hac_lag else build_report(inputs, hac_lag=lag)[0]
        )
        sensitivity_frames.append(
            lag_summary.filter(pl.col("period") == "full")
            .select(
                "direction",
                "candidate_id",
                "equal_day_delta_z_mean",
                "hac_t",
                "hac_p_two_sided",
                "joint_holm_p_full",
                "joint_bonferroni_p_full",
                "daily_delta_raw_bp_mean",
                "daily_delta_raw_bp_hac_t",
                "daily_delta_raw_bp_hac_p",
                "daily_delta_residual_bp_mean",
                "daily_delta_residual_bp_hac_t",
                "daily_delta_residual_bp_hac_p",
                "stock_day_equal_delta_z_mean",
                "stock_day_equal_hac_t",
                "stock_day_equal_hac_p_two_sided",
                "stock_time_equal_delta_z_mean",
                "stock_time_equal_hac_t",
                "stock_time_equal_hac_p_two_sided",
            )
            .with_columns(pl.lit(lag).alias("hac_lag"))
        )
    sensitivity = pl.concat(sensitivity_frames, how="vertical").sort(
        "hac_lag",
        "direction",
        "candidate_id",
    )
    _atomic_write_parquet(summary, output_dir / "period_summary.parquet")
    _atomic_write_parquet(monthly, output_dir / "monthly_stats.parquet")
    _atomic_write_parquet(partition, output_dir / "daily_state_partition.parquet")
    _atomic_write_parquet(
        daily_effect,
        output_dir / "daily_effects.parquet",
    )
    _atomic_write_parquet(
        sensitivity,
        output_dir / "hac_sensitivity.parquet",
    )
    _atomic_write_json(
        {
            "contract": REPORT_CONTRACT_ID,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": [str(_resolve_daily_path(path).resolve()) for path in inputs],
            "source_manifests": [
                str(_manifest_path(path).resolve()) for path in inputs
            ],
            "lineage_validation": (
                "failures empty; requested sessions complete; raw source "
                "identities identical across directions"
            ),
            "hac": {
                "kernel": "Bartlett",
                "lag_requested": hac_lag,
                "lag_unit": "ordered active session observations",
                "p_value": "two-sided asymptotic normal",
                "sensitivity_lags": sensitivity_lags,
            },
            "primary_estimand": "shared_pool",
            "joint_primary_family": (
                "four full-period direction-by-state shared HAC tests; "
                "Holm and Bonferroni adjusted under arbitrary dependence"
            ),
            "condition": "unmodified FillLots_atLow sign state",
            "artifacts": {
                "period_summary": "period_summary.parquet",
                "monthly_stats": "monthly_stats.parquet",
                "daily_effects": "daily_effects.parquet",
                "daily_state_partition": "daily_state_partition.parquet",
                "hac_sensitivity": "hac_sensitivity.parquet",
            },
            "period_summary": summary.to_dicts(),
        },
        output_dir / "report.json",
    )
    return summary, monthly, partition


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report raw FillLots_atLow state bias for short and long labels"
    )
    parser.add_argument(
        "inputs",
        nargs=2,
        type=Path,
        help="short and long artifact directories (order does not matter)",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hac-lag", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.hac_lag < 0:
        parser.error("--hac-lag cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary, _, partition = write_report(
        tuple(args.inputs),
        output_dir=args.output_dir,
        hac_lag=args.hac_lag,
    )
    if not args.quiet:
        columns = [
            "direction",
            "candidate_id",
            "period",
            "event_rows",
            "selection_rate_mean",
            "delta_z_pooled",
            "daily_delta_z_mean",
            "daily_delta_z_hac_t",
            "positive_days",
            "active_days",
            "positive_months",
            "months",
            "daily_delta_raw_bp_mean",
            "daily_delta_residual_bp_mean",
            "stock_day_equal_delta_z_mean",
            "stock_time_equal_delta_z_mean",
        ]
        with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=6):
            print(summary.select(columns))
        active_partition = partition.filter(pl.col("eligible_pool_rows") > 0)
        if active_partition.is_empty():
            print("Mean non-signed-state rate: unavailable (no active pool)")
        else:
            non_signed_rate = float(
                (
                    active_partition["non_signed_state_rows"]
                    / active_partition["eligible_pool_rows"]
                ).mean()
            )
            print(f"Mean non-signed-state rate: {non_signed_rate:.6%}")
        print(f"Report directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
