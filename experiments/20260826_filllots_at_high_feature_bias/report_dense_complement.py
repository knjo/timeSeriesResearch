"""Describe the dense calendar complement not used by the sparse screen.

This is an internal calendar-complement robustness view.  It is deliberately
separate from the preregistered dense acceptance and is not future
out-of-sample evidence: all dates were already present in the available
historical dense calendar when the sparse screen was run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl


STUDY_DIR = Path(__file__).resolve().parent
DENSE_REPORTER_PATH = STUDY_DIR / "report_dense.py"
REPORT_CONTRACT_ID = "filllots_at_high_dense_complement_report/v1"
DECISION_CONTRACT_ID = (
    "filllots_at_high_dense_complement_robustness/v1"
)
SPARSE_STEP = 5
EXPECTED_DENSE_SESSIONS = 377
EXPECTED_SPARSE_SESSIONS = 76
EXPECTED_COMPLEMENT_SESSIONS = 301
NOTABLE_POSSIBLY_INACTIVE_DATE = "20250407"
ROBUSTNESS_SCOPE = "internal_calendar_complement"


class ComplementReportError(ValueError):
    """Raised when sparse/dense calendars cannot define the frozen complement."""


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DENSE = _load_module(
    "filllots_at_high_dense_report_for_complement",
    DENSE_REPORTER_PATH,
)


def _load_sparse_pair(
    short_dir: Path,
    long_dir: Path,
) -> tuple[Any, Any]:
    try:
        short = DENSE.SPARSE._load_direction_input(
            short_dir,
            direction="short",
        )
        long = DENSE.SPARSE._load_direction_input(
            long_dir,
            direction="long",
        )
        DENSE.SPARSE._validate_pair(short, long)
    except (ValueError, ArithmeticError) as exc:
        raise ComplementReportError(
            f"invalid sparse calendar artifacts: {exc}"
        ) from exc
    for source in (short, long):
        if source.manifest.get("candidate_ids") != list(DENSE.CANDIDATE_IDS):
            raise ComplementReportError(
                f"{source.direction} sparse manifest is not the full 12 registry"
            )
        if source.manifest.get("step") != SPARSE_STEP:
            raise ComplementReportError(
                f"{source.direction} sparse manifest step must be {SPARSE_STEP}"
            )
        if source.manifest.get("failures") != []:
            raise ComplementReportError(
                f"{source.direction} sparse manifest has failures"
            )
    return short, long


def _complement_sessions(
    *,
    dense_short: Any,
    dense_long: Any,
    sparse_short: Any,
    sparse_long: Any,
) -> tuple[str, ...]:
    dense_sessions = tuple(dense_short.manifest["sessions"])
    if dense_sessions != tuple(dense_long.manifest["sessions"]):
        raise ComplementReportError("dense calendars differ across directions")
    sparse_sessions = tuple(sparse_short.manifest["sessions"])
    if sparse_sessions != tuple(sparse_long.manifest["sessions"]):
        raise ComplementReportError("sparse calendars differ across directions")
    if len(dense_sessions) != EXPECTED_DENSE_SESSIONS:
        raise ComplementReportError(
            f"dense calendar has {len(dense_sessions)} sessions, "
            f"expected {EXPECTED_DENSE_SESSIONS}"
        )
    if len(sparse_sessions) != EXPECTED_SPARSE_SESSIONS:
        raise ComplementReportError(
            f"sparse calendar has {len(sparse_sessions)} sessions, "
            f"expected {EXPECTED_SPARSE_SESSIONS}"
        )
    if sparse_sessions != dense_sessions[::SPARSE_STEP]:
        raise ComplementReportError(
            "sparse sessions are not the exact step-5 selection of dense sessions"
        )

    for dense_source, sparse_source in (
        (dense_short, sparse_short),
        (dense_long, sparse_long),
    ):
        dense_identities = dense_source.manifest["source_identities"]
        sparse_identities = sparse_source.manifest["source_identities"]
        for date in sparse_sessions:
            if sparse_identities[date] != dense_identities.get(date):
                raise ComplementReportError(
                    f"{dense_source.direction} {date} sparse/dense source "
                    "identities differ"
                )
        for field in (
            "start",
            "end",
            "pool_id",
            "pool_version",
            "stock_time_bucket",
            "primary_estimand",
            "control_sample_size",
            "source_identity",
        ):
            if dense_source.manifest.get(field) != sparse_source.manifest.get(field):
                raise ComplementReportError(
                    f"{dense_source.direction} sparse/dense manifests disagree "
                    f"on {field}"
                )

    selected = set(sparse_sessions)
    complement = tuple(date for date in dense_sessions if date not in selected)
    if len(complement) != EXPECTED_COMPLEMENT_SESSIONS:
        raise ComplementReportError(
            f"calendar complement has {len(complement)} sessions, "
            f"expected {EXPECTED_COMPLEMENT_SESSIONS}"
        )
    return complement


def _descriptive_checks(
    summary: pl.DataFrame,
    selected: tuple[tuple[str, str], ...],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    check_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for direction, candidate_id in selected:
        candidate = summary.filter(
            (pl.col("direction") == direction)
            & (pl.col("candidate_id") == candidate_id)
        )
        periods = {
            row["period"]: row for row in candidate.iter_rows(named=True)
        }
        if set(periods) != {"full", "2025", "2026"}:
            raise ComplementReportError(
                f"{direction}/{candidate_id} has incomplete complement periods"
            )
        full = periods["full"]

        def positive(value: Any) -> bool:
            return (
                value is not None
                and math.isfinite(float(value))
                and float(value) > 0.0
            )

        checks = {
            "full_effect_positive": positive(full["equal_day_delta_z_mean"]),
            "full_hac_p_lt_0_05": (
                full["hac_p_two_sided"] is not None
                and math.isfinite(float(full["hac_p_two_sided"]))
                and float(full["hac_p_two_sided"])
                < DENSE.DENSE_HAC_P_MAX
            ),
            "effect_2025_positive": positive(
                periods["2025"]["equal_day_delta_z_mean"]
            ),
            "effect_2026_positive": positive(
                periods["2026"]["equal_day_delta_z_mean"]
            ),
            "full_positive_month_ratio_ge_0_60": (
                full["positive_month_ratio"] is not None
                and math.isfinite(float(full["positive_month_ratio"]))
                and float(full["positive_month_ratio"])
                >= DENSE.DENSE_POSITIVE_MONTH_RATIO_MIN
            ),
        }
        all_checks = all(checks.values())
        check_rows.append(
            {
                "direction": direction,
                "candidate_id": candidate_id,
                "robustness_scope": ROBUSTNESS_SCOPE,
                **{
                    f"complement_descriptive_{name}": value
                    for name, value in checks.items()
                },
                "complement_all_descriptive_checks": all_checks,
                "complement_descriptive_result": (
                    "directionally_stable"
                    if all_checks
                    else "not_directionally_stable"
                ),
            }
        )
        decisions.append(
            {
                "direction": direction,
                "candidate_id": candidate_id,
                "scope": ROBUSTNESS_SCOPE,
                "result": (
                    "directionally_stable"
                    if all_checks
                    else "not_directionally_stable"
                ),
                "all_descriptive_checks": all_checks,
                "checks": checks,
                "thresholds": {
                    "hac_p_strictly_less_than": DENSE.DENSE_HAC_P_MAX,
                    "positive_month_ratio_min": (
                        DENSE.DENSE_POSITIVE_MONTH_RATIO_MIN
                    ),
                },
                "metrics": {
                    "full_equal_day_delta_z_mean": full[
                        "equal_day_delta_z_mean"
                    ],
                    "full_hac_p_two_sided": full["hac_p_two_sided"],
                    "effect_2025": periods["2025"][
                        "equal_day_delta_z_mean"
                    ],
                    "effect_2026": periods["2026"][
                        "equal_day_delta_z_mean"
                    ],
                    "full_positive_month_ratio": full[
                        "positive_month_ratio"
                    ],
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
    checks = pl.DataFrame(check_rows)
    return (
        summary.join(
            checks,
            on=["direction", "candidate_id"],
            how="inner",
            validate="m:1",
        ),
        decisions,
    )


def _load_inputs(
    *,
    promotion_json: Path,
    dense_short_dir: Path,
    dense_long_dir: Path,
    sparse_short_dir: Path,
    sparse_long_dir: Path,
) -> tuple[Any, Any, Any, Any, Any, tuple[str, ...]]:
    try:
        promotion, dense_short, dense_long = DENSE._load_inputs(
            promotion_json=promotion_json,
            short_dir=dense_short_dir,
            long_dir=dense_long_dir,
        )
    except (ValueError, ArithmeticError) as exc:
        raise ComplementReportError(f"invalid dense inputs: {exc}") from exc
    sparse_short, sparse_long = _load_sparse_pair(
        sparse_short_dir,
        sparse_long_dir,
    )
    complement = _complement_sessions(
        dense_short=dense_short,
        dense_long=dense_long,
        sparse_short=sparse_short,
        sparse_long=sparse_long,
    )
    return (
        promotion,
        dense_short,
        dense_long,
        sparse_short,
        sparse_long,
        complement,
    )


def build_report(
    *,
    promotion_json: Path,
    dense_short_dir: Path,
    dense_long_dir: Path,
    sparse_short_dir: Path,
    sparse_long_dir: Path,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    list[dict[str, Any]],
]:
    (
        promotion,
        dense_short,
        dense_long,
        _,
        _,
        complement,
    ) = _load_inputs(
        promotion_json=promotion_json,
        dense_short_dir=dense_short_dir,
        dense_long_dir=dense_long_dir,
        sparse_short_dir=sparse_short_dir,
        sparse_long_dir=sparse_long_dir,
    )
    summaries: list[pl.DataFrame] = []
    monthlies: list[pl.DataFrame] = []
    daily_frames: list[pl.DataFrame] = []
    for source in (dense_short, dense_long):
        complement_daily = source.daily.filter(pl.col("Date").is_in(complement))
        if len(complement_daily) != len(complement):
            raise ComplementReportError(
                f"{source.direction}: complement daily coverage is incomplete"
            )
        complement_source = replace(source, daily=complement_daily)
        summary, monthly = DENSE.SPARSE.COMMON.build_report(
            complement_daily,
            hac_lag=DENSE.SPARSE.PRIMARY_HAC_LAG,
        )
        summaries.append(
            summary.with_columns(pl.lit(source.direction).alias("direction"))
        )
        monthlies.append(
            monthly.with_columns(pl.lit(source.direction).alias("direction"))
        )
        daily_frames.append(DENSE.SPARSE._daily_effects(complement_source))

    selected = DENSE._selected_items(promotion)
    daily_effects = DENSE.SPARSE._direction_sort(
        pl.concat(daily_frames, how="vertical"),
        "candidate_id",
        "Date",
    )
    summary = DENSE.SPARSE._direction_sort(
        pl.concat(summaries, how="diagonal_relaxed").join(
            DENSE._period_diagnostics(daily_effects, selected),
            on=["direction", "candidate_id", "period"],
            how="inner",
            validate="1:1",
        ),
        "period",
        "candidate_id",
    )
    summary, decisions = _descriptive_checks(summary, selected)
    monthly = DENSE.SPARSE._direction_sort(
        pl.concat(monthlies, how="diagonal_relaxed"),
        "month",
        "candidate_id",
    )
    distribution = DENSE.SPARSE._direction_sort(
        DENSE._distribution_stats(daily_effects, selected),
        "period",
        "candidate_id",
    )
    sensitivity = DENSE.SPARSE._direction_sort(
        DENSE._hac_sensitivity(daily_effects, selected),
        "candidate_id",
        "hac_lag_requested",
    )
    return summary, monthly, daily_effects, distribution, sensitivity, decisions


def write_report(
    *,
    promotion_json: Path,
    dense_short_dir: Path,
    dense_long_dir: Path,
    sparse_short_dir: Path,
    sparse_long_dir: Path,
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
        dense_short_dir=dense_short_dir,
        dense_long_dir=dense_long_dir,
        sparse_short_dir=sparse_short_dir,
        sparse_long_dir=sparse_long_dir,
    )
    summary, monthly, daily, distribution, sensitivity, decisions = result
    (
        promotion,
        dense_short,
        dense_long,
        sparse_short,
        sparse_long,
        complement,
    ) = _load_inputs(
        promotion_json=promotion_json,
        dense_short_dir=dense_short_dir,
        dense_long_dir=dense_long_dir,
        sparse_short_dir=sparse_short_dir,
        sparse_long_dir=sparse_long_dir,
    )
    artifacts = {
        "period_summary": summary,
        "monthly_stats": monthly,
        "daily_effects": daily,
        "distribution_stats": distribution,
        "hac_sensitivity": sensitivity,
    }
    for name, frame in artifacts.items():
        DENSE.SPARSE.COMMON._atomic_write_parquet(
            frame,
            output_dir / f"{name}.parquet",
        )

    notable = daily.filter(pl.col("Date") == NOTABLE_POSSIBLY_INACTIVE_DATE)
    notable_audit = {
        "date": NOTABLE_POSSIBLY_INACTIVE_DATE,
        "in_complement": NOTABLE_POSSIBLY_INACTIVE_DATE in set(complement),
        "directions": [
            {
                "direction": row["direction"],
                "candidate_id": row["candidate_id"],
                "active": row["active"],
                "condition_n": row["condition_n"],
                "condition_n_z": row["condition_n_z"],
            }
            for row in notable.iter_rows(named=True)
        ],
    }
    payload = {
        "contract": DECISION_CONTRACT_ID,
        "report_contract": REPORT_CONTRACT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "descriptive_internal_calendar_complement_robustness",
        "preregistered_dense_acceptance": False,
        "future_out_of_sample": False,
        "interpretation": (
            "The complement excludes dates selected by the step-5 sparse screen, "
            "but all complement dates belong to the same already-available dense "
            "historical calendar. Results are descriptive robustness only."
        ),
        "calendar": {
            "dense_requested_sessions": len(dense_short.manifest["sessions"]),
            "sparse_selection_sessions": len(sparse_short.manifest["sessions"]),
            "complement_requested_sessions": len(complement),
            "selection_rule": "dense ordered sessions excluding dense[::5]",
            "complement_sessions": list(complement),
            "notable_possibly_inactive_session": notable_audit,
        },
        "lineage": {
            "promotion": {
                "path": str(promotion.path),
                "sha256": DENSE.SPARSE._sha256(promotion.path),
                "selected": promotion.payload["selected"],
            },
            "dense_short": DENSE.SPARSE._lineage(dense_short),
            "dense_long": DENSE.SPARSE._lineage(dense_long),
            "sparse_short": DENSE.SPARSE._lineage(sparse_short),
            "sparse_long": DENSE.SPARSE._lineage(sparse_long),
            "validation": (
                "sparse exact full registry, step=5, failures empty; dense step=1; "
                "ordered sparse sessions equal dense[::5]; matching-date raw source "
                "identities equal; complement contains 301 requested sessions"
            ),
        },
        "inference": {
            "primary_estimand": (
                "daily condition mean z minus the same-day complete shared "
                "eligible-pool mean z"
            ),
            "hac": {
                "kernel": "Bartlett",
                "lag_requested": DENSE.SPARSE.PRIMARY_HAC_LAG,
                "sensitivity_lags": list(DENSE.SPARSE.HAC_SENSITIVITY_LAGS),
                "p_value": "two-sided asymptotic normal",
            },
            "descriptive_checks": DENSE.EXPECTED_ACCEPTANCE_SPEC,
            "raw_and_residual_bp": (
                "equal-active-day condition-minus-control means with HAC(5)"
            ),
            "matched_attribution": (
                "runner-provided stock-day-equal and stock-time-equal daily z"
            ),
        },
        "artifacts": {
            name: {
                "path": f"{name}.parquet",
                "rows": len(frame),
                "columns": frame.columns,
            }
            for name, frame in artifacts.items()
        },
        "decisions": decisions,
        "period_summary": summary.to_dicts(),
    }
    DENSE.SPARSE.COMMON._atomic_write_json(
        payload,
        output_dir / "complement_decision.json",
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report internal dense-minus-sparse calendar robustness"
    )
    parser.add_argument("--promotion-json", required=True, type=Path)
    parser.add_argument("--dense-short-dir", required=True, type=Path)
    parser.add_argument("--dense-long-dir", required=True, type=Path)
    parser.add_argument("--sparse-short-dir", required=True, type=Path)
    parser.add_argument("--sparse-long-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _, _, _, _, _, decisions = write_report(
        promotion_json=args.promotion_json,
        dense_short_dir=args.dense_short_dir,
        dense_long_dir=args.dense_long_dir,
        sparse_short_dir=args.sparse_short_dir,
        sparse_long_dir=args.sparse_long_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(decisions, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
