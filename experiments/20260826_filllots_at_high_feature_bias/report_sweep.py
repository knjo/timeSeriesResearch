"""Build one reproducible long/short FillLots-at-high inference report.

The directional sweep artifacts already contain additive daily statistics.
This adapter validates their complete lineage, delegates the established
period/month/HAC/BH calculations to the canonical timeSeries reporter, and
adds the joint direction-by-candidate family and distribution diagnostics.
It never reloads market data or changes event membership.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl

from src.research.intradaySignal.framework.profiles.benchmark import (
    LONG_TAKER_TP109_V1,
)
from src.research.intradaySignal.framework.profiles.filllots_at_high import (
    CANDIDATE_DESCRIPTIONS,
    CANDIDATE_IDS,
    FILLLOTS_AT_HIGH_FEATURE_FAMILY_ID,
    FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION,
)
from src.research.intradaySignal.framework.profiles.negfill import (
    SHORT_TAKER_STOP108_V1,
)


COMMON_REPORTER_PATH = (
    Path(__file__).resolve().parents[1]
    / "20260825_filllots_at_low_framework"
    / "report_sweep.py"
)
SWEEP_CONTRACT_ID = "filllots_at_high_feature_bias_sweep/v1"
REPORT_CONTRACT_ID = "filllots_at_high_feature_bias_report/v1"
REGISTRY_ID = "filllots-at-high-feature-v1"
PRIMARY_HAC_LAG = 5
HAC_SENSITIVITY_LAGS = (0, 1, 3, 5, 10)
PERIODS: tuple[tuple[str, str | None], ...] = (
    ("full", None),
    ("2025", "2025"),
    ("2026", "2026"),
)
REQUIRED_SOURCE_IDENTITIES = (
    "tick_data",
    "tick_feature",
    "pre_market",
    "txf_tick",
)
SOURCE_IDENTITY_FIELDS = ("path", "size", "mtime_ns")
EXPECTED_DIRECTION_LABELS = {
    "short": (
        SHORT_TAKER_STOP108_V1.profile_id,
        SHORT_TAKER_STOP108_V1.fingerprint,
    ),
    "long": (
        LONG_TAKER_TP109_V1.profile_id,
        LONG_TAKER_TP109_V1.fingerprint,
    ),
}


class HighFeatureReportError(ValueError):
    """Raised when directional artifacts violate the joint report contract."""


@dataclass(frozen=True)
class DirectionInput:
    """Validated artifact and its immutable lineage metadata."""

    direction: str
    artifact_dir: Path
    daily_path: Path
    manifest_path: Path
    daily: pl.DataFrame
    manifest: dict[str, Any]


def _load_common_reporter() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "filllots_at_low_report_for_filllots_at_high",
        COMMON_REPORTER_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load common reporter: {COMMON_REPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMMON = _load_common_reporter()


def _one_value(frame: pl.DataFrame, column: str) -> Any:
    if column not in frame.columns:
        raise HighFeatureReportError(f"daily statistics missing {column}")
    values = frame[column].unique().to_list()
    if len(values) != 1:
        raise HighFeatureReportError(f"daily statistics mix {column}: {values}")
    return values[0]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HighFeatureReportError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HighFeatureReportError(f"{path} must contain a JSON object")
    return payload


def _validate_source_identities(
    manifest: dict[str, Any],
    *,
    sessions: tuple[str, ...],
    context: str,
) -> None:
    identity_spec = manifest.get("source_identity")
    if not isinstance(identity_spec, dict):
        raise HighFeatureReportError(f"{context}: source_identity is missing")
    sources = identity_spec.get("sources")
    if not isinstance(sources, list) or set(sources) != set(
        REQUIRED_SOURCE_IDENTITIES
    ) or len(sources) != len(REQUIRED_SOURCE_IDENTITIES):
        raise HighFeatureReportError(
            f"{context}: source identities must include exactly "
            f"{REQUIRED_SOURCE_IDENTITIES}"
        )
    fields = identity_spec.get("fields")
    if not isinstance(fields, list) or not set(SOURCE_IDENTITY_FIELDS).issubset(
        fields
    ):
        raise HighFeatureReportError(
            f"{context}: source identity fields must include "
            f"{SOURCE_IDENTITY_FIELDS}"
        )

    identities = manifest.get("source_identities")
    if not isinstance(identities, dict) or set(identities) != set(sessions):
        raise HighFeatureReportError(
            f"{context}: source identities do not cover every session"
        )
    for date in sessions:
        date_identity = identities[date]
        if not isinstance(date_identity, dict) or set(date_identity) != set(
            REQUIRED_SOURCE_IDENTITIES
        ):
            raise HighFeatureReportError(
                f"{context}: {date} source identities are incomplete"
            )
        for source in REQUIRED_SOURCE_IDENTITIES:
            identity = date_identity[source]
            if not isinstance(identity, dict) or not set(
                SOURCE_IDENTITY_FIELDS
            ).issubset(identity):
                raise HighFeatureReportError(
                    f"{context}: {date}/{source} identity is incomplete"
                )
            if not isinstance(identity["path"], str) or not identity["path"]:
                raise HighFeatureReportError(
                    f"{context}: {date}/{source} path is invalid"
                )
            for field in ("size", "mtime_ns"):
                value = identity[field]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise HighFeatureReportError(
                        f"{context}: {date}/{source} {field} is invalid"
                    )


def _validate_daily_contract(
    daily: pl.DataFrame,
    *,
    direction: str,
    sessions: tuple[str, ...],
) -> None:
    # This is the single source of truth for count, shared-control, optional
    # matched-estimand, period, and inference-input validation.
    try:
        COMMON.build_report(daily, hac_lag=PRIMARY_HAC_LAG)
    except (ValueError, ArithmeticError) as exc:
        raise HighFeatureReportError(
            f"{direction}: invalid daily inference statistics: {exc}"
        ) from exc

    expected_values = {
        "contract_id": SWEEP_CONTRACT_ID,
        "family_id": FILLLOTS_AT_HIGH_FEATURE_FAMILY_ID,
        "family_version": FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION,
        "label_profile_id": EXPECTED_DIRECTION_LABELS[direction][0],
        "label_profile_hash": EXPECTED_DIRECTION_LABELS[direction][1],
        "control_mode": "full",
        "control_sample_size": 0,
        "primary_estimand": "shared_pool",
    }
    for column, expected in expected_values.items():
        actual = _one_value(daily, column)
        if actual != expected:
            raise HighFeatureReportError(
                f"{direction}: {column}={actual!r}, expected {expected!r}"
            )

    required_matched = {
        "condition_unique_quotes",
        "selection_rate",
        "condition_sum_raw_bp",
        "control_sum_raw_bp",
        "condition_sum_residual_bp",
        "control_sum_residual_bp",
        "stock_day_equal_delta_z",
        "stock_time_equal_delta_z",
        "stock_time_bucket",
    }
    missing_matched = sorted(required_matched - set(daily.columns))
    if missing_matched:
        raise HighFeatureReportError(
            f"{direction}: missing matched daily metrics {missing_matched}"
        )
    invalid_quotes = daily.filter(
        pl.col("condition_unique_quotes").is_null()
        | (pl.col("condition_unique_quotes") < 0)
        | (pl.col("condition_unique_quotes") > pl.col("condition_n"))
    )
    if not invalid_quotes.is_empty():
        raise HighFeatureReportError(
            f"{direction}: condition_unique_quotes is inconsistent"
        )
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
        .otherwise(
            pl.col("selection_rate").is_not_null()
            & ~pl.col("selection_rate").is_nan()
        )
    )
    if not invalid_rate.is_empty():
        raise HighFeatureReportError(f"{direction}: selection_rate is invalid")
    for column in (
        "condition_sum_raw_bp",
        "control_sum_raw_bp",
        "condition_sum_residual_bp",
        "control_sum_residual_bp",
    ):
        invalid = daily.filter(pl.col(column).is_null() | ~pl.col(column).is_finite())
        if not invalid.is_empty():
            raise HighFeatureReportError(f"{direction}: {column} must be finite")

    candidates = tuple(daily["candidate_id"].unique().sort().to_list())
    if candidates != tuple(sorted(CANDIDATE_IDS)):
        raise HighFeatureReportError(
            f"{direction}: daily candidate set is not the full 12-candidate registry"
        )
    dates = tuple(daily["Date"].cast(pl.String).unique().sort().to_list())
    if dates != sessions:
        raise HighFeatureReportError(
            f"{direction}: daily sessions do not align with the manifest"
        )
    coverage = daily.select("Date", "candidate_id").unique()
    expected_rows = len(sessions) * len(CANDIDATE_IDS)
    if len(coverage) != expected_rows or len(daily) != expected_rows:
        raise HighFeatureReportError(
            f"{direction}: every candidate must appear once on every session; "
            f"rows={len(daily)}, expected={expected_rows}"
        )


def _load_direction_input(artifact_dir: Path, *, direction: str) -> DirectionInput:
    if direction not in EXPECTED_DIRECTION_LABELS:  # pragma: no cover
        raise ValueError(direction)
    artifact_dir = artifact_dir.resolve()
    if not artifact_dir.is_dir():
        raise HighFeatureReportError(
            f"{direction}: artifact directory does not exist: {artifact_dir}"
        )
    daily_path = artifact_dir / "daily_stats.parquet"
    manifest_path = artifact_dir / "run_manifest.json"
    for path in (daily_path, manifest_path):
        if not path.is_file():
            raise HighFeatureReportError(f"{direction}: required input missing: {path}")

    daily = pl.read_parquet(daily_path).sort("Date", "candidate_id")
    manifest = _read_json(manifest_path)
    context = f"{direction} manifest"
    expected_manifest = {
        "contract": SWEEP_CONTRACT_ID,
        "candidate_registry": REGISTRY_ID,
        "family_id": FILLLOTS_AT_HIGH_FEATURE_FAMILY_ID,
        "family_version": FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION,
        "label_profile_id": EXPECTED_DIRECTION_LABELS[direction][0],
        "label_profile_hash": EXPECTED_DIRECTION_LABELS[direction][1],
        "primary_estimand": "shared_pool",
        "control_sample_size": None,
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise HighFeatureReportError(
                f"{context}: {field}={manifest.get(field)!r}, expected {expected!r}"
            )
    if manifest.get("candidate_ids") != list(CANDIDATE_IDS):
        raise HighFeatureReportError(
            f"{context}: candidate_ids must be the ordered full registry"
        )
    if manifest.get("candidate_definitions") != CANDIDATE_DESCRIPTIONS:
        raise HighFeatureReportError(
            f"{context}: candidate definitions drifted from the framework profile"
        )
    if manifest.get("failures") != []:
        raise HighFeatureReportError(
            f"{context}: failures must be present and empty; "
            f"found {manifest.get('failures')!r}"
        )
    sessions_value = manifest.get("sessions")
    if (
        not isinstance(sessions_value, list)
        or not sessions_value
        or any(not isinstance(date, str) for date in sessions_value)
        or len(sessions_value) != len(set(sessions_value))
        or sessions_value != sorted(sessions_value)
    ):
        raise HighFeatureReportError(
            f"{context}: sessions must be a non-empty ordered unique string list"
        )
    sessions = tuple(sessions_value)
    _validate_source_identities(manifest, sessions=sessions, context=context)
    _validate_daily_contract(daily, direction=direction, sessions=sessions)

    manifest_daily_pairs = (
        ("contract", "contract_id"),
        ("family_id", "family_id"),
        ("family_version", "family_version"),
        ("label_profile_id", "label_profile_id"),
        ("label_profile_hash", "label_profile_hash"),
        ("pool_id", "pool_id"),
        ("pool_version", "pool_version"),
        ("stock_time_bucket", "stock_time_bucket"),
        ("primary_estimand", "primary_estimand"),
    )
    for manifest_field, daily_column in manifest_daily_pairs:
        if manifest.get(manifest_field) != _one_value(daily, daily_column):
            raise HighFeatureReportError(
                f"{context}: {manifest_field} disagrees with daily statistics"
            )
    return DirectionInput(
        direction=direction,
        artifact_dir=artifact_dir,
        daily_path=daily_path,
        manifest_path=manifest_path,
        daily=daily,
        manifest=manifest,
    )


def _validate_pair(short: DirectionInput, long: DirectionInput) -> None:
    if short.manifest["sessions"] != long.manifest["sessions"]:
        raise HighFeatureReportError(
            "short and long manifests use different session calendars"
        )
    if short.manifest["source_identities"] != long.manifest["source_identities"]:
        raise HighFeatureReportError(
            "short and long manifests use different raw source identities"
        )
    comparable_fields = (
        "candidate_registry",
        "family_id",
        "family_version",
        "candidate_ids",
        "candidate_definitions",
        "pool_id",
        "pool_version",
        "stock_time_bucket",
        "source_identity",
        "start",
        "end",
        "step",
        "control_sample_size",
        "primary_estimand",
    )
    for field in comparable_fields:
        if short.manifest.get(field) != long.manifest.get(field):
            raise HighFeatureReportError(
                f"short and long manifests disagree on {field}"
            )


def holm_adjust(
    p_values: dict[str, float | None],
    *,
    family_size: int | None = None,
) -> dict[str, float | None]:
    """Return Holm step-down adjusted p-values, preserving null tests.

    ``family_size`` may exceed the number of finite p-values.  The joint study
    uses the fixed preregistered family size of 24, conservatively treating a
    non-computable test as an unranked member of that family.
    """

    total = len(p_values) if family_size is None else family_size
    if total < len(p_values) or total < 1:
        raise ValueError("family_size must cover every registered hypothesis")
    adjusted = {key: None for key in p_values}
    ranked = sorted(
        (float(value), key)
        for key, value in p_values.items()
        if value is not None and math.isfinite(float(value))
    )
    running = 0.0
    for rank, (p_value, key) in enumerate(ranked, start=1):
        running = max(running, (total - rank + 1) * p_value)
        adjusted[key] = min(1.0, max(0.0, running))
    return adjusted


def _add_joint_holm(summary: pl.DataFrame) -> pl.DataFrame:
    full = summary.filter(pl.col("period") == "full")
    expected = len(EXPECTED_DIRECTION_LABELS) * len(CANDIDATE_IDS)
    if len(full) != expected:
        raise HighFeatureReportError(
            f"joint Holm family has {len(full)} rows, expected {expected}"
        )
    keyed_p = {
        f"{row['direction']}::{row['candidate_id']}": row["hac_p_two_sided"]
        for row in full.iter_rows(named=True)
    }
    if len(keyed_p) != expected:
        raise HighFeatureReportError("joint Holm family contains duplicate tests")
    adjusted = holm_adjust(keyed_p, family_size=expected)
    adjustment = pl.DataFrame(
        [
            {
                "direction": key.split("::", maxsplit=1)[0],
                "candidate_id": key.split("::", maxsplit=1)[1],
                "_joint_holm_p": value,
            }
            for key, value in adjusted.items()
        ],
        schema_overrides={"_joint_holm_p": pl.Float64},
    )
    return (
        summary.join(
            adjustment,
            on=["direction", "candidate_id"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.when(pl.col("period") == "full")
            .then(pl.col("_joint_holm_p"))
            .otherwise(None)
            .alias("joint_holm_p_full")
        )
        .drop("_joint_holm_p")
    )


def _daily_effects(source: DirectionInput) -> pl.DataFrame:
    active = (pl.col("condition_n_z") > 0) & (pl.col("control_n_z") > 0)
    return (
        source.daily.with_columns(
            pl.lit(source.direction).alias("direction"),
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
            .alias("daily_shared_delta_z"),
            pl.when(active)
            .then(pl.col("stock_day_equal_delta_z"))
            .alias("daily_stock_day_equal_delta_z"),
            pl.when(active)
            .then(pl.col("stock_time_equal_delta_z"))
            .alias("daily_stock_time_equal_delta_z"),
        )
        .select(
            "direction",
            "candidate_id",
            "Date",
            "active",
            "eligible_pool_rows",
            "condition_n",
            "condition_n_z",
            "control_n",
            "control_n_z",
            "condition_unique_quotes",
            "selection_rate",
            "condition_mean_z",
            "control_mean_z",
            "daily_shared_delta_z",
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
            "daily_stock_day_equal_delta_z",
            "daily_stock_time_equal_delta_z",
        )
    )


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    value = pl.Series("daily_shared_delta_z", values).quantile(
        probability,
        interpolation="linear",
    )
    return float(value) if value is not None else None


def _distribution_stats(daily_effects: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for direction in ("short", "long"):
        for candidate_id in CANDIDATE_IDS:
            candidate = daily_effects.filter(
                (pl.col("direction") == direction)
                & (pl.col("candidate_id") == candidate_id)
            )
            for period, year in PERIODS:
                subset = (
                    candidate
                    if year is None
                    else candidate.filter(pl.col("Date").str.starts_with(year))
                )
                values = [
                    float(value)
                    for value in subset["daily_shared_delta_z"]
                    .drop_nulls()
                    .to_list()
                    if math.isfinite(float(value))
                ]
                rows.append(
                    {
                        "direction": direction,
                        "candidate_id": candidate_id,
                        "period": period,
                        "sessions": subset["Date"].n_unique(),
                        "active_days": len(values),
                        "q05": _quantile(values, 0.05),
                        "q25": _quantile(values, 0.25),
                        "median": _quantile(values, 0.50),
                        "q75": _quantile(values, 0.75),
                        "q95": _quantile(values, 0.95),
                        "positive_day_ratio": (
                            sum(value > 0.0 for value in values) / len(values)
                            if values
                            else None
                        ),
                    }
                )
    return pl.DataFrame(
        rows,
        schema_overrides={
            "q05": pl.Float64,
            "q25": pl.Float64,
            "median": pl.Float64,
            "q75": pl.Float64,
            "q95": pl.Float64,
            "positive_day_ratio": pl.Float64,
        },
    )


def _hac_sensitivity(daily_effects: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for direction in ("short", "long"):
        for candidate_id in CANDIDATE_IDS:
            candidate = daily_effects.filter(
                (pl.col("direction") == direction)
                & (pl.col("candidate_id") == candidate_id)
            ).sort("Date")
            values = [
                float(value)
                for value in candidate["daily_shared_delta_z"]
                .drop_nulls()
                .to_list()
                if math.isfinite(float(value))
            ]
            equal = COMMON.equal_weight_stats(values)
            for lag in HAC_SENSITIVITY_LAGS:
                hac = COMMON.newey_west_mean_stats(values, lag)
                rows.append(
                    {
                        "direction": direction,
                        "candidate_id": candidate_id,
                        "active_days": len(values),
                        "equal_day_delta_z_mean": equal["mean"],
                        "hac_lag_requested": lag,
                        "hac_lag_effective": hac["lag"],
                        "hac_long_run_variance": hac["long_run_variance"],
                        "hac_se": hac["se"],
                        "hac_t": hac["t"],
                        "hac_p_two_sided": hac["p"],
                    }
                )
    return pl.DataFrame(
        rows,
        schema_overrides={
            "equal_day_delta_z_mean": pl.Float64,
            "hac_long_run_variance": pl.Float64,
            "hac_se": pl.Float64,
            "hac_t": pl.Float64,
            "hac_p_two_sided": pl.Float64,
        },
    )


def _period_diagnostics(daily_effects: pl.DataFrame) -> pl.DataFrame:
    """Build activity and equal-active-day BP diagnostics for each period."""

    rows: list[dict[str, Any]] = []
    for direction in ("short", "long"):
        for candidate_id in CANDIDATE_IDS:
            candidate = daily_effects.filter(
                (pl.col("direction") == direction)
                & (pl.col("candidate_id") == candidate_id)
            ).sort("Date")
            for period, year in PERIODS:
                subset = (
                    candidate
                    if year is None
                    else candidate.filter(pl.col("Date").str.starts_with(year))
                )
                active = subset.filter(pl.col("active"))
                raw_values = [
                    float(value)
                    for value in active["daily_delta_raw_bp"].drop_nulls().to_list()
                    if math.isfinite(float(value))
                ]
                residual_values = [
                    float(value)
                    for value in active["daily_delta_residual_bp"]
                    .drop_nulls()
                    .to_list()
                    if math.isfinite(float(value))
                ]
                raw_equal = COMMON.equal_weight_stats(raw_values)
                raw_hac = COMMON.newey_west_mean_stats(raw_values, PRIMARY_HAC_LAG)
                residual_equal = COMMON.equal_weight_stats(residual_values)
                residual_hac = COMMON.newey_west_mean_stats(
                    residual_values,
                    PRIMARY_HAC_LAG,
                )
                rows.append(
                    {
                        "direction": direction,
                        "candidate_id": candidate_id,
                        "period": period,
                        "mean_events_per_active_day": (
                            float(active["condition_n"].mean())
                            if not active.is_empty()
                            else None
                        ),
                        "mean_quotes_per_active_day": (
                            float(active["condition_unique_quotes"].mean())
                            if not active.is_empty()
                            else None
                        ),
                        "selection_rate_mean": (
                            float(active["selection_rate"].mean())
                            if not active.is_empty()
                            else None
                        ),
                        "selection_rate_median": (
                            float(active["selection_rate"].median())
                            if not active.is_empty()
                            else None
                        ),
                        "equal_day_delta_raw_bp_mean": raw_equal["mean"],
                        "raw_bp_hac_lag_effective": raw_hac["lag"],
                        "raw_bp_hac_se": raw_hac["se"],
                        "raw_bp_hac_t": raw_hac["t"],
                        "raw_bp_hac_p_two_sided": raw_hac["p"],
                        "equal_day_delta_residual_bp_mean": residual_equal["mean"],
                        "residual_bp_hac_lag_effective": residual_hac["lag"],
                        "residual_bp_hac_se": residual_hac["se"],
                        "residual_bp_hac_t": residual_hac["t"],
                        "residual_bp_hac_p_two_sided": residual_hac["p"],
                    }
                )
    return pl.DataFrame(
        rows,
        schema_overrides={
            "mean_events_per_active_day": pl.Float64,
            "mean_quotes_per_active_day": pl.Float64,
            "selection_rate_mean": pl.Float64,
            "selection_rate_median": pl.Float64,
            "equal_day_delta_raw_bp_mean": pl.Float64,
            "raw_bp_hac_se": pl.Float64,
            "raw_bp_hac_t": pl.Float64,
            "raw_bp_hac_p_two_sided": pl.Float64,
            "equal_day_delta_residual_bp_mean": pl.Float64,
            "residual_bp_hac_se": pl.Float64,
            "residual_bp_hac_t": pl.Float64,
            "residual_bp_hac_p_two_sided": pl.Float64,
        },
    )


def _direction_sort(frame: pl.DataFrame, *columns: str) -> pl.DataFrame:
    direction_order = pl.col("direction").replace_strict(
        {"short": 0, "long": 1},
        return_dtype=pl.Int8,
    )
    return (
        frame.with_columns(direction_order.alias("_direction_order"))
        .sort("_direction_order", *columns)
        .drop("_direction_order")
    )


def build_report(
    *,
    short_dir: Path,
    long_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Validate both directions and return every generated report table."""

    short = _load_direction_input(short_dir, direction="short")
    long = _load_direction_input(long_dir, direction="long")
    _validate_pair(short, long)

    summaries: list[pl.DataFrame] = []
    monthly_frames: list[pl.DataFrame] = []
    daily_frames: list[pl.DataFrame] = []
    for source in (short, long):
        summary, monthly = COMMON.build_report(
            source.daily,
            hac_lag=PRIMARY_HAC_LAG,
        )
        summaries.append(
            summary.with_columns(pl.lit(source.direction).alias("direction"))
        )
        monthly_frames.append(
            monthly.with_columns(pl.lit(source.direction).alias("direction"))
        )
        daily_frames.append(_daily_effects(source))

    period_summary = _add_joint_holm(
        _direction_sort(
            pl.concat(summaries, how="diagonal_relaxed").join(
                _period_diagnostics(
                    pl.concat(daily_frames, how="vertical")
                ),
                on=["direction", "candidate_id", "period"],
                how="inner",
                validate="1:1",
            ),
            "period",
            "candidate_id",
        )
    )
    monthly_stats = _direction_sort(
        pl.concat(monthly_frames, how="diagonal_relaxed"),
        "month",
        "candidate_id",
    )
    daily_effects = _direction_sort(
        pl.concat(daily_frames, how="vertical"),
        "candidate_id",
        "Date",
    )
    distribution_stats = _direction_sort(
        _distribution_stats(daily_effects),
        "period",
        "candidate_id",
    )
    hac_sensitivity = _direction_sort(
        _hac_sensitivity(daily_effects),
        "candidate_id",
        "hac_lag_requested",
    )
    return (
        period_summary,
        monthly_stats,
        daily_effects,
        distribution_stats,
        hac_sensitivity,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lineage(source: DirectionInput) -> dict[str, Any]:
    manifest = source.manifest
    return {
        "artifact_dir": str(source.artifact_dir),
        "daily_stats": {
            "path": str(source.daily_path),
            "sha256": _sha256(source.daily_path),
            "rows": len(source.daily),
        },
        "run_manifest": {
            "path": str(source.manifest_path),
            "sha256": _sha256(source.manifest_path),
        },
        "sweep_contract": manifest["contract"],
        "candidate_registry": manifest["candidate_registry"],
        "family_id": manifest["family_id"],
        "family_version": manifest["family_version"],
        "label_profile_id": manifest["label_profile_id"],
        "label_profile_hash": manifest["label_profile_hash"],
        "pool_id": manifest["pool_id"],
        "pool_version": manifest["pool_version"],
        "primary_estimand": manifest["primary_estimand"],
        "control_sample_size": manifest["control_sample_size"],
        "stock_time_bucket": manifest["stock_time_bucket"],
        "start": manifest.get("start"),
        "end": manifest.get("end"),
        "step": manifest.get("step"),
        "sessions": manifest["sessions"],
        "source_identity": manifest["source_identity"],
        "source_identities": manifest["source_identities"],
    }


def write_report(
    *,
    short_dir: Path,
    long_dir: Path,
    output_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    tables = build_report(short_dir=short_dir, long_dir=long_dir)
    (
        period_summary,
        monthly_stats,
        daily_effects,
        distribution_stats,
        hac_sensitivity,
    ) = tables

    short = _load_direction_input(short_dir, direction="short")
    long = _load_direction_input(long_dir, direction="long")
    _validate_pair(short, long)
    artifacts = {
        "period_summary": period_summary,
        "monthly_stats": monthly_stats,
        "daily_effects": daily_effects,
        "distribution_stats": distribution_stats,
        "hac_sensitivity": hac_sensitivity,
    }
    for name, frame in artifacts.items():
        COMMON._atomic_write_parquet(frame, output_dir / f"{name}.parquet")

    COMMON._atomic_write_json(
        {
            "contract": REPORT_CONTRACT_ID,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "lineage": {
                "validation": {
                    "full_candidate_registry": list(CANDIDATE_IDS),
                    "candidate_definitions": CANDIDATE_DESCRIPTIONS,
                    "failures_required_empty": True,
                    "same_session_calendar": True,
                    "identical_source_identities_across_directions": True,
                    "required_sources": list(REQUIRED_SOURCE_IDENTITIES),
                },
                "short": _lineage(short),
                "long": _lineage(long),
            },
            "inference": {
                "primary_estimand": (
                    "daily condition mean z minus the same-day complete shared "
                    "eligible-pool mean z"
                ),
                "period_equal_day": "equal weight over active sessions",
                "pooled_effect": (
                    "condition and control rows pooled only over sessions where "
                    "both cohorts have z-valid observations"
                ),
                "hac": {
                    "kernel": "Bartlett",
                    "lag_requested": PRIMARY_HAC_LAG,
                    "lag_unit": "ordered active sessions after null removal",
                    "p_value": "two-sided asymptotic normal for the HAC t-statistic",
                },
                "within_direction_bh": (
                    "separate 12-candidate full-period BH families for short and "
                    "long shared HAC tests, computed by the common reporter"
                ),
                "joint_holm": (
                    "one fixed 24-test family over direction by candidate full-period "
                    "shared HAC p-values; Holm step-down under arbitrary dependence"
                ),
                "matched_metrics": {
                    "stock_day_equal": (
                        "runner-provided equal-stock condition-minus-pool daily z"
                    ),
                    "stock_time_equal": (
                        "runner-provided equal-stock, equal-time-bucket "
                        "condition-minus-pool daily z"
                    ),
                },
                "bp_diagnostics": (
                    "equal-active-day condition-minus-control raw and beta-residual "
                    "basis-point means with Bartlett HAC(5) inference"
                ),
                "distribution": (
                    "linear-interpolated active-day quantiles of daily shared delta-z; "
                    "positive means strictly greater than zero"
                ),
                "hac_sensitivity_lags": list(HAC_SENSITIVITY_LAGS),
            },
            "artifacts": {
                name: {
                    "path": f"{name}.parquet",
                    "rows": len(frame),
                    "columns": frame.columns,
                }
                for name, frame in artifacts.items()
            },
            "period_summary": period_summary.to_dicts(),
            "distribution_stats": distribution_stats.to_dicts(),
            "hac_sensitivity": hac_sensitivity.to_dicts(),
        },
        output_dir / "report.json",
    )
    return tables


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the joint FillLots_atHigh long/short bias report"
    )
    parser.add_argument("--short-dir", required=True, type=Path)
    parser.add_argument("--long-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    period_summary, _, _, _, _ = write_report(
        short_dir=args.short_dir,
        long_dir=args.long_dir,
        output_dir=args.output_dir,
    )
    print(
        f"Wrote {len(period_summary)} period rows to "
        f"{args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
