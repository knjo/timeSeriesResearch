"""Report the preregistered dense checks for promoted FillLots-at-high events.

The sparse promotion artifact is verified against its hashed study spec and
period summary, then recomputed before either dense input is accepted.  Dense
statistics reuse the canonical daily inference math; this module adds only
the fixed promotion lineage and the preregistered dense acceptance decision.
"""

from __future__ import annotations

import argparse
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

from src.research.timeSeries.framework.profiles.filllots_at_high import (
    CANDIDATE_DESCRIPTIONS,
    CANDIDATE_IDS,
    FILLLOTS_AT_HIGH_FEATURE_FAMILY_ID,
    FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION,
)


STUDY_DIR = Path(__file__).resolve().parent
SPARSE_REPORTER_PATH = STUDY_DIR / "report_sweep.py"
PROMOTION_SELECTOR_PATH = STUDY_DIR / "select_promotions.py"
STUDY_SPEC_PATH = STUDY_DIR / "study_spec.json"
PROMOTION_CONTRACT_ID = "filllots_at_high_sparse_promotion/v1"
REPORT_CONTRACT_ID = "filllots_at_high_dense_report/v1"
DECISION_CONTRACT_ID = "filllots_at_high_dense_acceptance/v1"
EXPECTED_STUDY_CONTRACT = "filllots_at_high_feature_bias_study/v1"
EXPECTED_ACCEPTANCE_SPEC = (
    "positive full-period equal-day delta-z with HAC p < 0.05, positive "
    "2025 and 2026 effects, and positive-month ratio >= 0.60"
)
DENSE_HAC_P_MAX = 0.05
DENSE_POSITIVE_MONTH_RATIO_MIN = 0.60


class DenseReportError(ValueError):
    """Raised when promotion or dense-run lineage violates the contract."""


@dataclass(frozen=True)
class PromotionInput:
    path: Path
    payload: dict[str, Any]
    study_spec_path: Path
    study_spec: dict[str, Any]
    sparse_period_summary_path: Path
    selected: dict[str, str]


@dataclass(frozen=True)
class DenseDirectionInput:
    direction: str
    candidate_id: str
    artifact_dir: Path
    daily_path: Path
    manifest_path: Path
    daily: pl.DataFrame
    manifest: dict[str, Any]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SPARSE = _load_module(
    "filllots_at_high_sparse_report_for_dense",
    SPARSE_REPORTER_PATH,
)
SELECTOR = _load_module(
    "filllots_at_high_promotion_selector_for_dense",
    PROMOTION_SELECTOR_PATH,
)


def _validate_hashed_reference(
    reference: Any,
    *,
    name: str,
) -> Path:
    if not isinstance(reference, dict):
        raise DenseReportError(f"promotion {name} reference is missing")
    path_value = reference.get("path")
    digest = reference.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise DenseReportError(f"promotion {name} path is invalid")
    if not isinstance(digest, str) or len(digest) != 64:
        raise DenseReportError(f"promotion {name} sha256 is invalid")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise DenseReportError(f"promotion {name} input is missing: {path}")
    actual = SPARSE._sha256(path)
    if actual != digest:
        raise DenseReportError(
            f"promotion {name} sha256={digest}, actual={actual}"
        )
    return path


def _load_promotion(path: Path) -> PromotionInput:
    path = path.resolve()
    payload = SPARSE._read_json(path)
    if payload.get("contract") != PROMOTION_CONTRACT_ID:
        raise DenseReportError(
            f"promotion contract={payload.get('contract')!r}, "
            f"expected {PROMOTION_CONTRACT_ID!r}"
        )
    study_spec_path = _validate_hashed_reference(
        payload.get("study_spec"),
        name="study_spec",
    )
    if study_spec_path != STUDY_SPEC_PATH.resolve():
        raise DenseReportError(
            "promotion study spec is not this versioned study_spec.json"
        )
    period_path = _validate_hashed_reference(
        payload.get("period_summary"),
        name="period_summary",
    )
    study_spec = SPARSE._read_json(study_spec_path)
    if study_spec.get("contract") != EXPECTED_STUDY_CONTRACT:
        raise DenseReportError("promotion study contract is invalid")
    family = study_spec.get("candidate_family")
    if not isinstance(family, dict) or family.get("ids") != list(CANDIDATE_IDS):
        raise DenseReportError("study spec candidate family drifted")
    dense_check = study_spec.get("dense_check")
    if (
        not isinstance(dense_check, dict)
        or dense_check.get("session_step") != 1
        or dense_check.get("acceptance") != EXPECTED_ACCEPTANCE_SPEC
    ):
        raise DenseReportError("study spec dense-check contract drifted")

    sparse_summary = pl.read_parquet(period_path)
    recomputed_gate, recomputed_selected = SELECTOR.select_promotions(
        sparse_summary,
        study_spec,
    )
    if payload.get("selected") != recomputed_selected:
        raise DenseReportError(
            "promotion selections do not match the hashed sparse inputs"
        )
    if payload.get("gate_rows") != recomputed_gate.to_dicts():
        raise DenseReportError(
            "promotion gate rows do not match the hashed sparse inputs"
        )
    if len(recomputed_selected) != 2:
        raise DenseReportError("promotion must select exactly one candidate per label")
    selected: dict[str, str] = {}
    for row in recomputed_selected:
        direction = row.get("direction")
        candidate_id = row.get("candidate_id")
        if direction not in ("short", "long"):
            raise DenseReportError(f"unexpected promotion direction {direction!r}")
        if direction in selected:
            raise DenseReportError(f"duplicate {direction} promotion")
        if candidate_id not in CANDIDATE_IDS:
            raise DenseReportError(f"unknown promoted candidate {candidate_id!r}")
        if row.get("decision") != "promoted":
            raise DenseReportError(
                f"{direction}/{candidate_id} decision is not promoted"
            )
        selected[str(direction)] = str(candidate_id)
    if set(selected) != {"short", "long"}:
        raise DenseReportError(f"promotion directions={sorted(selected)}")
    return PromotionInput(
        path=path,
        payload=payload,
        study_spec_path=study_spec_path,
        study_spec=study_spec,
        sparse_period_summary_path=period_path,
        selected=selected,
    )


def _validate_dense_daily(
    daily: pl.DataFrame,
    *,
    direction: str,
    candidate_id: str,
    sessions: tuple[str, ...],
) -> None:
    try:
        SPARSE.COMMON.build_report(daily, hac_lag=SPARSE.PRIMARY_HAC_LAG)
    except (ValueError, ArithmeticError) as exc:
        raise DenseReportError(
            f"{direction}: invalid dense daily inference statistics: {exc}"
        ) from exc
    label_id, label_hash = SPARSE.EXPECTED_DIRECTION_LABELS[direction]
    expected = {
        "contract_id": SPARSE.SWEEP_CONTRACT_ID,
        "family_id": FILLLOTS_AT_HIGH_FEATURE_FAMILY_ID,
        "family_version": FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION,
        "label_profile_id": label_id,
        "label_profile_hash": label_hash,
        "control_mode": "full",
        "control_sample_size": 0,
        "primary_estimand": "shared_pool",
    }
    for column, expected_value in expected.items():
        actual = SPARSE._one_value(daily, column)
        if actual != expected_value:
            raise DenseReportError(
                f"{direction}: {column}={actual!r}, expected {expected_value!r}"
            )
    actual_candidates = daily["candidate_id"].unique().to_list()
    if actual_candidates != [candidate_id]:
        raise DenseReportError(
            f"{direction}: dense daily candidate set={actual_candidates}, "
            f"expected {[candidate_id]}"
        )
    actual_dates = tuple(daily["Date"].cast(pl.String).unique().sort().to_list())
    if actual_dates != sessions or len(daily) != len(sessions):
        raise DenseReportError(
            f"{direction}: dense daily rows do not cover every manifest session"
        )

    required = {
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
    missing = sorted(required - set(daily.columns))
    if missing:
        raise DenseReportError(f"{direction}: missing dense diagnostics {missing}")
    invalid_quotes = daily.filter(
        pl.col("condition_unique_quotes").is_null()
        | (pl.col("condition_unique_quotes") < 0)
        | (pl.col("condition_unique_quotes") > pl.col("condition_n"))
    )
    if not invalid_quotes.is_empty():
        raise DenseReportError(f"{direction}: invalid condition_unique_quotes")
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
        raise DenseReportError(f"{direction}: invalid selection_rate")
    for column in (
        "condition_sum_raw_bp",
        "control_sum_raw_bp",
        "condition_sum_residual_bp",
        "control_sum_residual_bp",
    ):
        invalid = daily.filter(pl.col(column).is_null() | ~pl.col(column).is_finite())
        if not invalid.is_empty():
            raise DenseReportError(f"{direction}: {column} must be finite")


def _load_dense_input(
    artifact_dir: Path,
    *,
    direction: str,
    candidate_id: str,
    study_spec: dict[str, Any],
) -> DenseDirectionInput:
    artifact_dir = artifact_dir.resolve()
    if not artifact_dir.is_dir():
        raise DenseReportError(
            f"{direction}: dense artifact directory is missing: {artifact_dir}"
        )
    daily_path = artifact_dir / "daily_stats.parquet"
    manifest_path = artifact_dir / "run_manifest.json"
    for required_path in (daily_path, manifest_path):
        if not required_path.is_file():
            raise DenseReportError(
                f"{direction}: dense artifact is missing {required_path.name}"
            )
    daily = pl.read_parquet(daily_path).sort("Date", "candidate_id")
    manifest = SPARSE._read_json(manifest_path)
    label_id, label_hash = SPARSE.EXPECTED_DIRECTION_LABELS[direction]
    expected_manifest = {
        "contract": SPARSE.SWEEP_CONTRACT_ID,
        "candidate_registry": SPARSE.REGISTRY_ID,
        "family_id": FILLLOTS_AT_HIGH_FEATURE_FAMILY_ID,
        "family_version": FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION,
        "label_profile_id": label_id,
        "label_profile_hash": label_hash,
        "primary_estimand": "shared_pool",
        "control_sample_size": None,
        "step": 1,
        "end": study_spec["data_end_inclusive"],
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise DenseReportError(
                f"{direction} manifest {field}={manifest.get(field)!r}, "
                f"expected {expected!r}"
            )
    if manifest.get("candidate_ids") != [candidate_id]:
        raise DenseReportError(
            f"{direction} dense manifest candidate set is not the promotion"
        )
    if manifest.get("candidate_definitions") != {
        candidate_id: CANDIDATE_DESCRIPTIONS[candidate_id]
    }:
        raise DenseReportError(
            f"{direction} dense candidate definition drifted"
        )
    if manifest.get("failures") != []:
        raise DenseReportError(
            f"{direction} dense manifest failures={manifest.get('failures')!r}"
        )
    sessions_value = manifest.get("sessions")
    if (
        not isinstance(sessions_value, list)
        or not sessions_value
        or any(not isinstance(date, str) for date in sessions_value)
        or sessions_value != sorted(sessions_value)
        or len(sessions_value) != len(set(sessions_value))
    ):
        raise DenseReportError(
            f"{direction} dense sessions must be ordered and unique"
        )
    sessions = tuple(sessions_value)
    SPARSE._validate_source_identities(
        manifest,
        sessions=sessions,
        context=f"{direction} dense manifest",
    )
    _validate_dense_daily(
        daily,
        direction=direction,
        candidate_id=candidate_id,
        sessions=sessions,
    )
    for manifest_field, daily_column in (
        ("contract", "contract_id"),
        ("family_id", "family_id"),
        ("family_version", "family_version"),
        ("label_profile_id", "label_profile_id"),
        ("label_profile_hash", "label_profile_hash"),
        ("pool_id", "pool_id"),
        ("pool_version", "pool_version"),
        ("stock_time_bucket", "stock_time_bucket"),
        ("primary_estimand", "primary_estimand"),
    ):
        if manifest.get(manifest_field) != SPARSE._one_value(daily, daily_column):
            raise DenseReportError(
                f"{direction} manifest {manifest_field} disagrees with daily stats"
            )
    return DenseDirectionInput(
        direction=direction,
        candidate_id=candidate_id,
        artifact_dir=artifact_dir,
        daily_path=daily_path,
        manifest_path=manifest_path,
        daily=daily,
        manifest=manifest,
    )


def _validate_dense_pair(
    short: DenseDirectionInput,
    long: DenseDirectionInput,
) -> None:
    if short.manifest["sessions"] != long.manifest["sessions"]:
        raise DenseReportError(
            "dense short and long inputs use different complete session calendars"
        )
    if short.manifest["source_identities"] != long.manifest["source_identities"]:
        raise DenseReportError(
            "dense short and long inputs use different raw source identities"
        )
    for field in (
        "family_id",
        "family_version",
        "pool_id",
        "pool_version",
        "stock_time_bucket",
        "source_identity",
        "start",
        "end",
        "step",
        "control_sample_size",
        "primary_estimand",
    ):
        if short.manifest.get(field) != long.manifest.get(field):
            raise DenseReportError(f"dense manifests disagree on {field}")


def _selected_items(promotion: PromotionInput) -> tuple[tuple[str, str], ...]:
    return tuple(
        (direction, promotion.selected[direction])
        for direction in ("short", "long")
    )


def _period_diagnostics(
    daily_effects: pl.DataFrame,
    selected: tuple[tuple[str, str], ...],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for direction, candidate_id in selected:
        candidate = daily_effects.filter(
            (pl.col("direction") == direction)
            & (pl.col("candidate_id") == candidate_id)
        ).sort("Date")
        for period, year in SPARSE.PERIODS:
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
            raw_equal = SPARSE.COMMON.equal_weight_stats(raw_values)
            raw_hac = SPARSE.COMMON.newey_west_mean_stats(
                raw_values,
                SPARSE.PRIMARY_HAC_LAG,
            )
            residual_equal = SPARSE.COMMON.equal_weight_stats(residual_values)
            residual_hac = SPARSE.COMMON.newey_west_mean_stats(
                residual_values,
                SPARSE.PRIMARY_HAC_LAG,
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
    return pl.DataFrame(rows)


def _distribution_stats(
    daily_effects: pl.DataFrame,
    selected: tuple[tuple[str, str], ...],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for direction, candidate_id in selected:
        candidate = daily_effects.filter(
            (pl.col("direction") == direction)
            & (pl.col("candidate_id") == candidate_id)
        )
        for period, year in SPARSE.PERIODS:
            subset = (
                candidate
                if year is None
                else candidate.filter(pl.col("Date").str.starts_with(year))
            )
            values = [
                float(value)
                for value in subset["daily_shared_delta_z"].drop_nulls().to_list()
                if math.isfinite(float(value))
            ]
            rows.append(
                {
                    "direction": direction,
                    "candidate_id": candidate_id,
                    "period": period,
                    "sessions": subset["Date"].n_unique(),
                    "active_days": len(values),
                    "q05": SPARSE._quantile(values, 0.05),
                    "q25": SPARSE._quantile(values, 0.25),
                    "median": SPARSE._quantile(values, 0.50),
                    "q75": SPARSE._quantile(values, 0.75),
                    "q95": SPARSE._quantile(values, 0.95),
                    "positive_day_ratio": (
                        sum(value > 0.0 for value in values) / len(values)
                        if values
                        else None
                    ),
                }
            )
    return pl.DataFrame(rows)


def _hac_sensitivity(
    daily_effects: pl.DataFrame,
    selected: tuple[tuple[str, str], ...],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for direction, candidate_id in selected:
        candidate = daily_effects.filter(
            (pl.col("direction") == direction)
            & (pl.col("candidate_id") == candidate_id)
        ).sort("Date")
        values = [
            float(value)
            for value in candidate["daily_shared_delta_z"].drop_nulls().to_list()
            if math.isfinite(float(value))
        ]
        equal = SPARSE.COMMON.equal_weight_stats(values)
        for lag in SPARSE.HAC_SENSITIVITY_LAGS:
            hac = SPARSE.COMMON.newey_west_mean_stats(values, lag)
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
    return pl.DataFrame(rows)


def _positive(value: Any) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) > 0.0


def _add_acceptance(
    summary: pl.DataFrame,
    promotion: PromotionInput,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    acceptance_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for direction, candidate_id in _selected_items(promotion):
        candidate = summary.filter(
            (pl.col("direction") == direction)
            & (pl.col("candidate_id") == candidate_id)
        )
        rows = {row["period"]: row for row in candidate.iter_rows(named=True)}
        if set(rows) != {"full", "2025", "2026"}:
            raise DenseReportError(
                f"{direction}/{candidate_id}: incomplete dense periods"
            )
        full = rows["full"]
        gates = {
            "full_effect_positive": _positive(full["equal_day_delta_z_mean"]),
            "full_hac_p_lt_0_05": (
                full["hac_p_two_sided"] is not None
                and math.isfinite(float(full["hac_p_two_sided"]))
                and float(full["hac_p_two_sided"]) < DENSE_HAC_P_MAX
            ),
            "effect_2025_positive": _positive(
                rows["2025"]["equal_day_delta_z_mean"]
            ),
            "effect_2026_positive": _positive(
                rows["2026"]["equal_day_delta_z_mean"]
            ),
            "full_positive_month_ratio_ge_0_60": (
                full["positive_month_ratio"] is not None
                and math.isfinite(float(full["positive_month_ratio"]))
                and float(full["positive_month_ratio"])
                >= DENSE_POSITIVE_MONTH_RATIO_MIN
            ),
        }
        accepted = all(gates.values())
        acceptance_rows.append(
            {
                "direction": direction,
                "candidate_id": candidate_id,
                "promotion_decision": "promoted",
                **{f"dense_{key}": value for key, value in gates.items()},
                "dense_accepted": accepted,
                "dense_acceptance_decision": (
                    "accepted" if accepted else "rejected"
                ),
            }
        )
        decisions.append(
            {
                "direction": direction,
                "candidate_id": candidate_id,
                "promotion_decision": "promoted",
                "dense_acceptance_decision": (
                    "accepted" if accepted else "rejected"
                ),
                "accepted": accepted,
                "thresholds": {
                    "hac_p_strictly_less_than": DENSE_HAC_P_MAX,
                    "positive_month_ratio_min": DENSE_POSITIVE_MONTH_RATIO_MIN,
                },
                "gates": gates,
                "metrics": {
                    "full_equal_day_delta_z_mean": full[
                        "equal_day_delta_z_mean"
                    ],
                    "full_hac_p_two_sided": full["hac_p_two_sided"],
                    "effect_2025": rows["2025"]["equal_day_delta_z_mean"],
                    "effect_2026": rows["2026"]["equal_day_delta_z_mean"],
                    "full_positive_month_ratio": full["positive_month_ratio"],
                    "full_equal_day_delta_raw_bp_mean": full[
                        "equal_day_delta_raw_bp_mean"
                    ],
                    "full_equal_day_delta_residual_bp_mean": full[
                        "equal_day_delta_residual_bp_mean"
                    ],
                    "full_stock_day_equal_delta_z_mean": full[
                        "stock_day_equal_delta_z_mean"
                    ],
                    "full_stock_day_equal_hac_p_two_sided": full[
                        "stock_day_equal_hac_p_two_sided"
                    ],
                    "full_stock_time_equal_delta_z_mean": full[
                        "stock_time_equal_delta_z_mean"
                    ],
                    "full_stock_time_equal_hac_p_two_sided": full[
                        "stock_time_equal_hac_p_two_sided"
                    ],
                },
            }
        )
    acceptance = pl.DataFrame(acceptance_rows)
    return (
        summary.join(
            acceptance,
            on=["direction", "candidate_id"],
            how="inner",
            validate="m:1",
        ),
        decisions,
    )


def _load_inputs(
    *,
    promotion_json: Path,
    short_dir: Path,
    long_dir: Path,
) -> tuple[PromotionInput, DenseDirectionInput, DenseDirectionInput]:
    promotion = _load_promotion(promotion_json)
    short = _load_dense_input(
        short_dir,
        direction="short",
        candidate_id=promotion.selected["short"],
        study_spec=promotion.study_spec,
    )
    long = _load_dense_input(
        long_dir,
        direction="long",
        candidate_id=promotion.selected["long"],
        study_spec=promotion.study_spec,
    )
    _validate_dense_pair(short, long)
    return promotion, short, long


def build_report(
    *,
    promotion_json: Path,
    short_dir: Path,
    long_dir: Path,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    list[dict[str, Any]],
]:
    promotion, short, long = _load_inputs(
        promotion_json=promotion_json,
        short_dir=short_dir,
        long_dir=long_dir,
    )
    summaries: list[pl.DataFrame] = []
    monthlies: list[pl.DataFrame] = []
    daily_frames: list[pl.DataFrame] = []
    for source in (short, long):
        summary, monthly = SPARSE.COMMON.build_report(
            source.daily,
            hac_lag=SPARSE.PRIMARY_HAC_LAG,
        )
        summaries.append(
            summary.with_columns(pl.lit(source.direction).alias("direction"))
        )
        monthlies.append(
            monthly.with_columns(pl.lit(source.direction).alias("direction"))
        )
        daily_frames.append(SPARSE._daily_effects(source))
    selected = _selected_items(promotion)
    daily_effects = SPARSE._direction_sort(
        pl.concat(daily_frames, how="vertical"),
        "candidate_id",
        "Date",
    )
    summary = SPARSE._direction_sort(
        pl.concat(summaries, how="diagonal_relaxed").join(
            _period_diagnostics(daily_effects, selected),
            on=["direction", "candidate_id", "period"],
            how="inner",
            validate="1:1",
        ),
        "period",
        "candidate_id",
    )
    summary, decisions = _add_acceptance(summary, promotion)
    monthly = SPARSE._direction_sort(
        pl.concat(monthlies, how="diagonal_relaxed"),
        "month",
        "candidate_id",
    )
    distribution = SPARSE._direction_sort(
        _distribution_stats(daily_effects, selected),
        "period",
        "candidate_id",
    )
    sensitivity = SPARSE._direction_sort(
        _hac_sensitivity(daily_effects, selected),
        "candidate_id",
        "hac_lag_requested",
    )
    return summary, monthly, daily_effects, distribution, sensitivity, decisions


def write_report(
    *,
    promotion_json: Path,
    short_dir: Path,
    long_dir: Path,
    output_dir: Path,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    list[dict[str, Any]],
]:
    result = build_report(
        promotion_json=promotion_json,
        short_dir=short_dir,
        long_dir=long_dir,
    )
    summary, monthly, daily, distribution, sensitivity, decisions = result
    promotion, short, long = _load_inputs(
        promotion_json=promotion_json,
        short_dir=short_dir,
        long_dir=long_dir,
    )
    artifacts = {
        "period_summary": summary,
        "monthly_stats": monthly,
        "daily_effects": daily,
        "distribution_stats": distribution,
        "hac_sensitivity": sensitivity,
    }
    for name, frame in artifacts.items():
        SPARSE.COMMON._atomic_write_parquet(
            frame,
            output_dir / f"{name}.parquet",
        )
    decision_payload = {
        "contract": DECISION_CONTRACT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "promotion": {
            "path": str(promotion.path),
            "sha256": SPARSE._sha256(promotion.path),
            "contract": PROMOTION_CONTRACT_ID,
        },
        "study_spec": {
            "path": str(promotion.study_spec_path),
            "sha256": SPARSE._sha256(promotion.study_spec_path),
            "dense_check": promotion.study_spec["dense_check"],
        },
        "decisions": decisions,
    }
    SPARSE.COMMON._atomic_write_json(
        decision_payload,
        output_dir / "decision.json",
    )
    SPARSE.COMMON._atomic_write_json(
        {
            "contract": REPORT_CONTRACT_ID,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "lineage": {
                "promotion": {
                    "path": str(promotion.path),
                    "sha256": SPARSE._sha256(promotion.path),
                    "contract": PROMOTION_CONTRACT_ID,
                    "hashed_study_spec": promotion.payload["study_spec"],
                    "hashed_sparse_period_summary": promotion.payload[
                        "period_summary"
                    ],
                    "selected": promotion.payload["selected"],
                },
                "short": SPARSE._lineage(short),
                "long": SPARSE._lineage(long),
                "validation": (
                    "promotion recomputed from hashed sparse inputs; promoted-only "
                    "one-candidate manifests; step=1; failures empty; complete daily "
                    "calendar and raw source identities identical across directions"
                ),
            },
            "inference": {
                "primary_estimand": (
                    "daily condition mean z minus the same-day complete shared "
                    "eligible-pool mean z"
                ),
                "hac": {
                    "kernel": "Bartlett",
                    "lag_requested": SPARSE.PRIMARY_HAC_LAG,
                    "lag_unit": "ordered active sessions after null removal",
                    "p_value": "two-sided asymptotic normal",
                    "sensitivity_lags": list(SPARSE.HAC_SENSITIVITY_LAGS),
                },
                "raw_and_residual_bp": (
                    "equal-active-day condition-minus-control means with HAC(5)"
                ),
                "matched_attribution": (
                    "runner-provided stock-day-equal and stock-time-equal daily z"
                ),
                "dense_acceptance": promotion.study_spec["dense_check"],
            },
            "artifacts": {
                **{
                    name: {
                        "path": f"{name}.parquet",
                        "rows": len(frame),
                        "columns": frame.columns,
                    }
                    for name, frame in artifacts.items()
                },
                "decision": "decision.json",
            },
            "period_summary": summary.to_dicts(),
            "decisions": decisions,
        },
        output_dir / "report.json",
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report dense checks for promoted FillLots_atHigh candidates"
    )
    parser.add_argument("--promotion-json", required=True, type=Path)
    parser.add_argument("--short-dir", required=True, type=Path)
    parser.add_argument("--long-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _, _, _, _, _, decisions = write_report(
        promotion_json=args.promotion_json,
        short_dir=args.short_dir,
        long_dir=args.long_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(decisions, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
