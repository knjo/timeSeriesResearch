from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import polars as pl


MODULE_PATH = Path(__file__).with_name("build_decision_artifact.py")
SPEC = importlib.util.spec_from_file_location(
    "fh_neg_final_decision_builder", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise ImportError(MODULE_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_parquet(path: Path, frame: pl.DataFrame | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame if frame is not None else pl.DataFrame({"row": [1]})
    output.write_parquet(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity_record(path: Path, project_root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(project_root)),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def fixture_dates() -> list[str]:
    start = dt.date(2025, 2, 11)
    dates = [
        (start + dt.timedelta(days=index)).strftime("%Y%m%d")
        for index in range(376)
    ]
    dates.append("20260825")
    return dates


def create_fixture(root: Path, *, plateau_pass: bool = True) -> BUILDER.DecisionPaths:
    project = root / "project"
    study = (
        project
        / "src/research/timeSeries/test/20260827_fh_neg_rejection_robustness"
    )
    artifacts = study / "artifacts"
    calendar_dir = artifacts / "calendar_retrospective_v1"
    lineage_dir = artifacts / "lineage_retrospective_v1"
    inference_dir = artifacts / "inference_retrospective_v1"
    secondary_dir = artifacts / "secondary_diagnostics_v1"
    oos_dir = secondary_dir / "artifact_oos_20260826_descriptive_v1"
    plateau_run = artifacts / "threshold_plateau_v1"
    plateau_eval = artifacts / "threshold_plateau_evaluation_v1"
    output_dir = artifacts / "final_decision_v1"
    dates = fixture_dates()

    write_json(
        study / "study_contract.json",
        {
            "contract_id": BUILDER.PARENT_CONTRACT,
            "contract_status": "frozen_before_v1_outcome_run",
            "freeze_date": "20260827",
            "prospective_date_rule": "Date > 20260827",
            "scope_boundary": {"production_promotion_authorized": False},
        },
    )
    write_json(
        study / "study_contract_amendment_1.json",
        {
            "amendment_id": f"{BUILDER.PARENT_CONTRACT}/amendment_1",
            "parent_contract_id": BUILDER.PARENT_CONTRACT,
            "deployment_override": {
                "stage_1_factorial_can_authorize_deployment": False
            },
        },
    )
    write_json(
        study / "study_contract_amendment_2.json",
        {
            "amendment_id": f"{BUILDER.PARENT_CONTRACT}/amendment_2",
            "parent_contract_id": BUILDER.PARENT_CONTRACT,
            "depends_on": f"{BUILDER.PARENT_CONTRACT}/amendment_1",
            "materiality_boundary": {
                "new_post_outcome_delta_min_added": False
            },
        },
    )
    write_json(
        study / "threshold_plateau_contract.json",
        {
            "contract_id": BUILDER.PLATEAU_CONTRACT,
            "evidence_class": "retrospective_diagnostic",
            "production_promotion_authorized": False,
            "date_range": ["20250211", "20260825"],
            "fixed_grid": {
                "depth_ratio_lte": [-6, -8, -10],
                "gap_ticks_gte": [3, 4, 5],
                "bear_score_gte": [1, 2, 3],
                "cartesian_cells": 27,
                "selection_rule": (
                    "Do not rank cells or replace the center. Every cell and "
                    "failed/zero-support period remains in the ledger."
                ),
            },
            "inference": {
                "alpha": 0.10,
                "circular_block_lengths": [10, 5, 20],
                "bootstrap_seed": 20_260_827,
                "bootstrap_repetitions": 5_000,
                "family_correction": (
                    "single-step joint-valid max-stat across all 27 fixed cells"
                ),
            },
        },
    )

    calendar = pl.DataFrame(
        {
            "Date": dates,
            "evidence_class": ["retrospective_diagnostic"] * len(dates),
            "source_complete": [True] * len(dates),
            "run_status": ["success"] * len(dates),
            "session_status": ["zero_control_ineligible"]
            + ["evaluable"] * (len(dates) - 1),
            "eligible_pool_rows": [1_000] * len(dates),
            "control_n_z": [0] + [100] * (len(dates) - 1),
            "aggregate_complete": [True] * len(dates),
            "membership_complete": [True] * len(dates),
        }
    )
    write_parquet(calendar_dir / "requested_calendar.parquet", calendar)
    write_json(
        calendar_dir / "requested_calendar.json",
        {
            "contract": "fh_neg_rejection_requested_calendar/v1",
            "factorial_contract": BUILDER.PARENT_CONTRACT,
            "factorial_family_id": BUILDER.FAMILY_ID,
            "factorial_family_version": BUILDER.FAMILY_VERSION,
            "evidence_class": "retrospective_diagnostic",
            "start": "20250211",
            "end": "20260825",
            "requested_sessions": len(dates),
            "successful_sessions": len(dates),
            "evaluable_sessions": len(dates) - 1,
            "zero_control_sessions": 1,
            "failed_run_sessions": 0,
            "missing_source_sessions": 0,
            "not_run_or_incomplete_sessions": 0,
        },
    )

    reference = artifacts / "reference_lineage_v1/reference_lineage.parquet"
    reference_manifest = artifacts / "reference_lineage_v1/reference_lineage.json"
    parent_manifest = artifacts / "dgb_factorial_v1/run_manifest.json"
    parent_daily = artifacts / "dgb_factorial_v1/daily_stats.parquet"
    write_parquet(reference)
    write_json(reference_manifest, {"contract": "reference/v1"})
    write_json(
        parent_manifest,
        {
            "contract": BUILDER.PARENT_CONTRACT,
            "family_id": BUILDER.FAMILY_ID,
            "family_version": BUILDER.FAMILY_VERSION,
            "failures": [],
        },
    )
    for name in BUILDER.LINEAGE_FILES[:-1]:
        write_parquet(lineage_dir / name)
    lineage_inputs = {
        "reference_lineage": identity_record(reference, project),
        "reference_lineage_manifest": identity_record(
            reference_manifest, project
        ),
        "requested_calendar": identity_record(
            calendar_dir / "requested_calendar.parquet", project
        ),
        "requested_calendar_manifest": identity_record(
            calendar_dir / "requested_calendar.json", project
        ),
        "run_manifest": identity_record(parent_manifest, project),
    }
    write_json(
        lineage_dir / "lineage_support_summary.json",
        {
            "contract": "fh_neg_rejection_lineage_support_audit/v1",
            "factorial_contract": BUILDER.PARENT_CONTRACT,
            "factorial_family_id": BUILDER.FAMILY_ID,
            "factorial_family_version": BUILDER.FAMILY_VERSION,
            "status": "pass",
            "scope": {
                "evidence_class": "retrospective_diagnostic",
                "start": "20250211",
                "end": "20260825",
                "requested_sessions": len(dates),
            },
            "outcome_artifacts_read": [],
            "outcome_columns_read": [],
            "selector_static_audit": {
                "status": "pass",
                "forbidden_input_count": 0,
                "future_syntax_finding_count": 0,
            },
            "legacy_parity": {"all_applicable_sessions_passed": True},
            "input_identities": lineage_inputs,
            "support": {
                "factorial_cells": {
                    BUILDER.ANCHOR_ID: {
                        "events": 6_000_000,
                        "active_sessions": len(dates),
                    }
                }
            },
            "validation_gaps": [
                {"id": "synthetic_gap", "detail": "fixture limitation"}
            ],
        },
    )

    positive_source = project / "references/positive_source.txt"
    positive_source.parent.mkdir(parents=True, exist_ok=True)
    positive_source.write_text("positive control", encoding="utf-8")
    positive_path = study / "positive_control_reference.json"
    write_json(
        positive_path,
        {
            "artifact_id": (
                "fh_neg_rejection_robustness/positive_control_reference/v1"
            ),
            "verdict": {
                "positive_control_component": "pass_reference",
                "full_g0_status": "not_decided_by_this_artifact",
            },
            "contract_alignment": {
                "condition": {
                    "pass": True,
                    "expected": "FillLots_atLow < 0",
                },
                "directional_label": {
                    "pass": True,
                    "expected_profile_id": "negfill.short_taker_stop108.v1",
                    "canonical_profile_hash": "a" * 64,
                    "current_factorial_profile_hash": "a" * 64,
                },
                "eligible_pool": {
                    "pass": True,
                    "expected_pool_id": (
                        "tw_equity_daytrade_rank150_090030_123000_bidref_v1"
                    ),
                },
                "control_and_estimand": {
                    "pass": True,
                    "canonical_control_mode": "full",
                    "current_factorial_control_mode": "full",
                    "canonical_primary_estimand": "shared_pool",
                    "current_factorial_primary_estimand": "shared_pool",
                },
                "pipeline_linkage": {"pass": True},
            },
            "canonical_reference": {
                "preregistered_cell": {
                    "equal_day_delta_z": 0.02,
                    "equal_day_raw_bp_uplift": 1.2,
                    "equal_day_residual_bp_uplift": 5.7,
                },
                "cross_year_direction": [
                    {"period": "2025", "equal_day_delta_z": 0.01},
                    {"period": "2026", "equal_day_delta_z": 0.03},
                ],
                "interpretation": "synthetic positive calibration",
            },
            "fingerprints": {
                "positive_source": identity_record(positive_source, project)
            },
            "lineage_limits": ["synthetic reference"],
        },
    )

    inference_calendar = calendar.with_columns(
        pl.lit("evaluation_expected").alias("calendar_status")
    )
    write_parquet(
        inference_dir / "requested_calendar.parquet", inference_calendar
    )
    input_daily = pl.DataFrame(
        {
            "Date": [date for date in dates for _ in BUILDER.CELL_IDS],
            "candidate_id": list(BUILDER.CELL_IDS) * len(dates),
        }
    )
    write_parquet(inference_dir / "input_daily_stats.parquet", input_daily)
    write_parquet(parent_daily, input_daily)
    daily_rows = []
    for date_index, date in enumerate(dates):
        active = date_index > 0
        for cell_id in BUILDER.CELL_IDS:
            daily_rows.append(
                {
                    "Date": date,
                    "candidate_id": cell_id,
                    "cell_id": cell_id,
                    "condition_n_z": 10 if active else 0,
                    "selection_rate": 0.001 if active else None,
                    "daily_delta_z": 0.1 if active else None,
                    "condition_active": active,
                    "effect_active": active,
                    "period_status": (
                        "active" if active else "zero_control_ineligible"
                    ),
                }
            )
    inference_daily = pl.DataFrame(daily_rows)
    write_parquet(inference_dir / "daily_effects.parquet", inference_daily)
    loo_daily_rows = []
    for date_index, date in enumerate(dates):
        active = date_index > 0
        for contrast_id, spec in BUILDER.LOO_SPECS.items():
            loo_daily_rows.append(
                {
                    "Date": date,
                    "contrast_id": contrast_id,
                    "removed_gate": spec["removed_gate"],
                    "full_cell_id": BUILDER.ANCHOR_ID,
                    "reduced_cell_id": spec["reduced_cell_id"],
                    "full_period_status": (
                        "active" if active else "zero_control_ineligible"
                    ),
                    "full_support": 10 if active else 0,
                    "reduced_support": 20 if active else 0,
                    "common_active": active,
                    "full_daily_delta_z": 0.1 if active else None,
                    "reduced_daily_delta_z": 0.07 if active else None,
                    "paired_increment_delta_z": 0.03 if active else None,
                }
            )
    write_parquet(
        inference_dir / "loo_daily.parquet", pl.DataFrame(loo_daily_rows)
    )
    write_parquet(inference_dir / "factorial_contrast_daily.parquet")
    cell_rows = []
    for cell_id in BUILDER.CELL_IDS:
        cell_rows.append(
            {
                "cell_id": cell_id,
                "calendar_periods": len(dates),
                "effect_active_periods": len(dates) - 1,
                "condition_support": 10 * (len(dates) - 1),
                "equal_day_delta_z": 0.1,
                "positive_day_ratio": 1.0,
                "selection_rate_mean": 0.001,
            }
        )
    write_parquet(inference_dir / "cell_summary.parquet", pl.DataFrame(cell_rows))
    loo_rows = []
    reduced = {
        "D_increment": ("depth", "d0_g1_b1"),
        "G_increment": ("far", "d1_g0_b1"),
        "B_increment": ("bear", "d1_g1_b0"),
    }
    for contrast_id, (gate, reduced_id) in reduced.items():
        loo_rows.append(
            {
                "contrast_id": contrast_id,
                "removed_gate": gate,
                "full_cell_id": BUILDER.ANCHOR_ID,
                "reduced_cell_id": reduced_id,
                "calendar_periods": len(dates),
                "common_active_periods": len(dates) - 1,
                "full_support_on_common_periods": 10 * (len(dates) - 1),
                "reduced_support_on_common_periods": 20 * (len(dates) - 1),
                "full_equal_day_delta_z_common": 0.1,
                "reduced_equal_day_delta_z_common": 0.07,
                "paired_increment_delta_z": 0.03,
                "median_paired_increment_delta_z": 0.03,
                "paired_increment_std": 0.0,
                "positive_increment_ratio": 1.0,
                "bootstrap_probability_gt_zero": 0.95,
                "bootstrap_p_one_sided_positive": 0.05,
                "bootstrap_ci_low": 0.01,
                "bootstrap_ci_high": 0.05,
            }
        )
    write_parquet(inference_dir / "loo_summary.parquet", pl.DataFrame(loo_rows))
    g4_names = {
        "D_increment": "D_at_G1_B1",
        "G_increment": "G_at_D1_B1",
        "B_increment": "B_at_D1_G1",
    }
    factorial_rows = [
        {
            "contrast_id": g4_names[contrast_id],
            "decision_role": "frozen_G4_parsimony",
            "g4_contrast_id": contrast_id,
            "calendar_periods": len(dates),
            "common_active_periods": len(dates) - 1,
            "paired_contrast_delta_z": 0.03,
            "median_paired_contrast_delta_z": 0.03,
            "bootstrap_probability_gt_zero": 0.95,
            "bootstrap_p_one_sided_positive": 0.05,
            "bootstrap_ci_low": 0.01,
            "bootstrap_ci_high": 0.05,
        }
        for contrast_id in BUILDER.LOO_IDS
    ]
    write_parquet(
        inference_dir / "factorial_contrasts.parquet",
        pl.DataFrame(factorial_rows),
    )
    bootstrap_rows = [
        {
            "target_kind": "loo_increment",
            "target_id": contrast_id,
            "block_length_requested": block_length,
            "primary_block_length": block_length == 10,
            "calendar_periods": len(dates),
            "active_periods": len(dates) - 1,
            "point_estimate": 0.03,
            "bootstrap_repetitions": 5_000,
            "bootstrap_probability_gt_zero": 0.95,
            "p_one_sided_positive_centered": 0.05,
            "ci_low": 0.01,
            "ci_high": 0.05,
            "seed": 20_260_827,
            "method": (
                "circular_moving_block_percentile_and_centered_test"
            ),
        }
        for contrast_id in BUILDER.LOO_IDS
        for block_length in (5, 10, 20)
    ]
    write_parquet(
        inference_dir / "bootstrap.parquet", pl.DataFrame(bootstrap_rows)
    )
    fwer_rows = []
    for cell_id in BUILDER.CELL_IDS:
        for block_length in (5, 10, 20):
            fwer_rows.append(
                {
                    "target_id": cell_id,
                    "block_length_requested": block_length,
                    "primary_block_length": block_length == 10,
                    "family_id": BUILDER.FAMILY_ID,
                    "calendar_periods": len(dates),
                    "active_periods": len(dates) - 1,
                    "point_estimate": 0.1,
                    "family_alpha": 0.10,
                    "simultaneous_confidence_level": 0.90,
                    "raw_trial_count": 8,
                    "bootstrap_repetitions": 5000,
                    "joint_valid_bootstrap_repetitions": 5000,
                    "joint_valid_bootstrap_rate": 1.0,
                    "seed": 20_260_827,
                    "method": (
                        "circular_moving_block_single_step_max_t_one_sided"
                    ),
                    "simultaneous_lower_bound": 0.08,
                    "single_step_fwer_p": 0.001,
                    "dependence_effective_trial_count": 1.6,
                }
            )
    write_parquet(inference_dir / "fwer_audit.parquet", pl.DataFrame(fwer_rows))
    temporal_rows = []
    for index, block_dates in enumerate(
        BUILDER._chronological_blocks(dates, 3), start=1
    ):
        active_periods = len(block_dates) - (1 if index == 1 else 0)
        temporal_rows.append(
            {
                "target_kind": "cell_effect",
                "target_id": BUILDER.ANCHOR_ID,
                "block_id": f"T{index:02d}",
                "block_start": block_dates[0],
                "block_end": block_dates[-1],
                "calendar_periods": len(block_dates),
                "active_periods": active_periods,
                "support": active_periods * 10,
                "equal_period_value": 0.1,
            }
        )
    write_parquet(
        inference_dir / "temporal_blocks.parquet", pl.DataFrame(temporal_rows)
    )
    ledger_rows = []
    for cell_id in BUILDER.CELL_IDS:
        ledger_rows.append(
            {
                "trial_id": cell_id,
                "selected": cell_id == BUILDER.ANCHOR_ID,
                "total_selection_exposure_is_lower_bound": True,
                "historical_selection_exposure_included": False,
            }
        )
    write_parquet(
        inference_dir / "trial_ledger.parquet", pl.DataFrame(ledger_rows)
    )
    write_json(
        inference_dir / "run_manifest.json",
        {
            "contract": "fh_neg_rejection_robustness_evaluator/v1",
            "full_cell_id": BUILDER.ANCHOR_ID,
            "effect_zero_support_policy": "undefined_not_zero",
            "raw_trial_count": 8,
            "fwer_method": (
                "circular_moving_block_single_step_max_t_one_sided"
            ),
            "cell_ids": list(BUILDER.CELL_IDS),
            "config": {
                "family_id": BUILDER.FAMILY_ID,
                "family_version": BUILDER.FAMILY_VERSION,
                "selected_cell_id": BUILDER.ANCHOR_ID,
                "bootstrap_seed": 20_260_827,
                "bootstrap_reps": 5_000,
                "primary_block_length": 10,
                "family_alpha": 0.10,
            },
            "parsimony_contract": {
                "contrasts": BUILDER.LOO_SPECS,
            },
            "loo_pairs": [
                {"contrast_id": contrast_id, **spec}
                for contrast_id, spec in BUILDER.LOO_SPECS.items()
            ],
        },
    )

    secondary_daily = inference_daily.with_columns(
        pl.when(pl.col("effect_active"))
        .then(pl.lit(12.0))
        .alias("daily_delta_raw_bp"),
        pl.when(pl.col("effect_active"))
        .then(pl.lit(18.0))
        .alias("daily_delta_residual_bp"),
    )
    write_parquet(
        secondary_dir / "secondary_daily_effects.parquet", secondary_daily
    )
    bp_rows = [
        {
            "cell_id": cell_id,
            "active_sessions": len(dates) - 1,
            "equal_active_raw_bp": 12.0,
            "equal_active_residual_bp": 18.0,
        }
        for cell_id in BUILDER.CELL_IDS
    ]
    write_parquet(secondary_dir / "equal_active_bp.parquet", pl.DataFrame(bp_rows))
    fixture_months = sorted({date[:6] for date in dates})
    monthly_effect_rows = []
    for cell_id in BUILDER.CELL_IDS:
        for month in fixture_months:
            month_dates = [date for date in dates if date.startswith(month)]
            active_sessions = sum(date != dates[0] for date in month_dates)
            monthly_effect_rows.append(
                {
                    "cell_id": cell_id,
                    "month": month,
                    "calendar_sessions": len(month_dates),
                    "active_sessions": active_sessions,
                    "month_mean_delta_z": 0.1 if active_sessions else None,
                    "month_mean_raw_bp": 12.0 if active_sessions else None,
                    "month_mean_residual_bp": 18.0 if active_sessions else None,
                    "month_active": active_sessions > 0,
                }
            )
    write_parquet(
        secondary_dir / "monthly_effects.parquet",
        pl.DataFrame(monthly_effect_rows),
    )
    monthly_rows = [
        {
            "cell_id": cell_id,
            "requested_months": len(fixture_months),
            "active_months": len(fixture_months),
            "positive_months": len(fixture_months),
            "positive_month_ratio": 1.0,
        }
        for cell_id in BUILDER.CELL_IDS
    ]
    write_parquet(
        secondary_dir / "monthly_summary.parquet", pl.DataFrame(monthly_rows)
    )
    hac_rows = [
        {
            "cell_id": cell_id,
            "requested_lag": lag,
            "active_sessions": len(dates) - 1,
            "mean_delta_z": 0.1,
            "hac_standard_error": 0.01,
            "hac_t_stat": 10.0,
            "p_one_sided_positive_normal": 0.001,
            "one_sided_90_lcb": 0.08,
            "kernel": "Bartlett Newey-West",
            "inference_role": "secondary_diagnostic_only",
        }
        for cell_id in BUILDER.CELL_IDS
        for lag in (1, 5, 10, 20)
    ]
    write_parquet(
        secondary_dir / "hac_diagnostics.parquet", pl.DataFrame(hac_rows)
    )
    interval_rows = []
    for block_index, block_dates in enumerate(
        BUILDER._chronological_blocks(dates, 3), start=1
    ):
        active_periods = len(block_dates) - (1 if block_index == 1 else 0)
        for block_length in (5, 10, 20):
            interval_rows.append(
                {
                    "target_id": BUILDER.ANCHOR_ID,
                    "block_id": f"T{block_index:02d}",
                    "block_start": block_dates[0],
                    "block_end": block_dates[-1],
                    "active_periods": active_periods,
                    "point_delta_z": 0.1,
                    "block_length_requested": block_length,
                    "primary_block_length": block_length == 10,
                    "one_sided_90_lcb": 0.08,
                    "one_sided_90_ucb": 0.12,
                    "interval_status": "ok",
                }
            )
    write_parquet(
        secondary_dir / "temporal_basic_intervals.parquet",
        pl.DataFrame(interval_rows),
    )
    secondary_source = project / "references/secondary_source.txt"
    secondary_source.write_text("secondary", encoding="utf-8")
    write_json(
        secondary_dir / "run_manifest.json",
        {
            "contract": "fh_neg_rejection_secondary_diagnostics/v1",
            "parent_contract": BUILDER.PARENT_CONTRACT,
            "amendment": (
                "fh_neg_rejection_robustness/fixed_factorial_v1/amendment_2"
            ),
            "family_id": BUILDER.FAMILY_ID,
            "family_version": BUILDER.FAMILY_VERSION,
            "anchor_cell_id": BUILDER.ANCHOR_ID,
            "evidence_class": "retrospective_diagnostic",
            "decision_scope": {
                "secondary_diagnostics_only": True,
                "condition_selection_performed": False,
                "deployment_decision_created": False,
            },
            "config": {
                "anchor_cell_id": BUILDER.ANCHOR_ID,
                "block_lengths": [5, 10, 20],
                "bootstrap_repetitions": 5_000,
                "bootstrap_seed": 20_260_827,
                "confidence_level": 0.90,
                "hac_lags": [1, 5, 10, 20],
                "primary_block_length": 10,
                "temporal_block_count": 3,
            },
            "effect_policy": {
                "inactive_effect": "undefined_not_zero",
                "period_weighting": "equal finite active session",
            },
            "hac": {
                "kernel": "Bartlett Newey-West",
                "lags": [1, 5, 10, 20],
                "role": "secondary_diagnostic_only",
            },
            "input_lineage": {
                "amendment_2": identity_record(
                    study / "study_contract_amendment_2.json", project
                ),
                "evaluator_daily_effects": identity_record(
                    inference_dir / "daily_effects.parquet", project
                ),
                "evaluator_manifest": identity_record(
                    inference_dir / "run_manifest.json", project
                ),
                "evaluator_requested_calendar": identity_record(
                    inference_dir / "requested_calendar.parquet", project
                ),
                "evaluator_temporal_blocks": identity_record(
                    inference_dir / "temporal_blocks.parquet", project
                ),
                "runner_daily_stats": identity_record(parent_daily, project),
            },
        },
    )

    oos_cells = [
        {
            "Date": "20260826",
            "evidence_class": "artifact_oos_non_pristine",
            "cell_id": cell_id,
            "condition_n_z": 1500,
            "control_n_z": 3_000_000,
            "daily_delta_z": -0.06,
            "daily_delta_raw_bp": -55.0,
            "daily_delta_residual_bp": -11.0,
            "effect_defined": True,
            "analysis_role": "one_session_descriptive_only",
            "inference_performed": False,
        }
        for cell_id in BUILDER.CELL_IDS
    ]
    write_parquet(oos_dir / "cell_descriptive.parquet", pl.DataFrame(oos_cells))
    oos_loo = [
        {
            "Date": "20260826",
            "evidence_class": "artifact_oos_non_pristine",
            "contrast_id": contrast_id,
            "removed_gate": reduced[contrast_id][0],
            "full_cell_id": BUILDER.ANCHOR_ID,
            "reduced_cell_id": reduced[contrast_id][1],
            "full_condition_n_z": 1500,
            "reduced_condition_n_z": 3000,
            "full_minus_loo_z": -0.02,
            "full_minus_loo_raw_bp": -4.0,
            "full_minus_loo_residual_bp": -3.0,
            "common_effect_defined": True,
            "analysis_role": "one_session_descriptive_only",
            "inference_performed": False,
        }
        for contrast_id in BUILDER.LOO_IDS
    ]
    write_parquet(oos_dir / "loo_descriptive.parquet", pl.DataFrame(oos_loo))
    oos_calendar = artifacts / "calendar_artifact_oos_v1/requested_calendar.parquet"
    write_parquet(
        oos_calendar,
        pl.DataFrame(
            {
                "Date": ["20260826"],
                "evidence_class": ["artifact_oos_non_pristine"],
            }
        ),
    )
    write_json(
        oos_dir / "run_manifest.json",
        {
            "contract": "fh_neg_rejection_artifact_oos_descriptive/v1",
            "parent_contract": BUILDER.PARENT_CONTRACT,
            "family_id": BUILDER.FAMILY_ID,
            "family_version": BUILDER.FAMILY_VERSION,
            "Date": "20260826",
            "evidence_class": "artifact_oos_non_pristine",
            "calendar_artifact": "calendar_artifact_oos_v1",
            "analysis_scope": {
                "descriptive_only": True,
                "hac_performed": False,
                "bootstrap_performed": False,
                "p_value_computed": False,
                "confidence_interval_computed": False,
                "inference_performed": False,
                "decision_created": False,
            },
            "separation_policy": {
                "retrospective_pooled": False,
                "one_session_only": True,
                "retrospective_outputs_referenced": False,
            },
            "input_lineage": {
                "amendment_2": identity_record(
                    study / "study_contract_amendment_2.json", project
                ),
                "artifact_oos_calendar": identity_record(
                    oos_calendar, project
                ),
                "runner_daily_stats": identity_record(parent_daily, project),
            },
        },
    )

    historical_source = project / "references/historical_source.txt"
    historical_source.write_text("historical", encoding="utf-8")
    generator = project / "src/build_historical.py"
    generator.parent.mkdir(parents=True, exist_ok=True)
    generator.write_text("generator", encoding="utf-8")
    historical_path = study / "historical_trial_exposure.json"
    write_json(
        historical_path,
        {
            "contract": "fh_neg_rejection_historical_trial_exposure/v1",
            "arithmetic_inventory_lower_bounds": {
                "condition_selector_configurations_across_distinct_historical_families": 28,
                "condition_direction_tests_across_distinct_historical_families": 46,
                "condition_label_outcome_tests_across_distinct_historical_families": 51,
                "valid_single_delta_z_max_stat_family": False,
            },
            "implication_for_current_study": {
                "historical_exposure_status": (
                    "material and incompletely bounded above"
                ),
                "new_dgb_factorial_current_family_size": 8,
            },
            "generator": identity_record(generator, project),
            "families": [
                {
                    "family_id": "synthetic_history",
                    "sources": [identity_record(historical_source, project)],
                }
            ],
        },
    )

    plateau_daily = pl.DataFrame(
        {
            "Date": [date for date in dates for _ in BUILDER.PLATEAU_CELL_IDS],
            "candidate_id": list(BUILDER.PLATEAU_CELL_IDS) * len(dates),
        }
    )
    write_parquet(plateau_run / "daily_stats.parquet", plateau_daily)
    write_parquet(plateau_run / "summary.parquet")
    write_json(
        plateau_run / "run_manifest.json",
        {
            "contract": BUILDER.PLATEAU_CONTRACT,
            "family_id": BUILDER.PLATEAU_FAMILY_ID,
            "family_version": BUILDER.PLATEAU_FAMILY_VERSION,
            "center_candidate_id": BUILDER.PLATEAU_CENTER_ID,
            "evidence_class": "retrospective_diagnostic",
            "outcome_informed_freeze": True,
            "production_promotion_authorized": False,
            "step": 1,
            "start": "20250211",
            "end": "20260825",
            "frozen_contract_sha256": sha256(
                study / "threshold_plateau_contract.json"
            ),
            "threshold_design_fingerprint": (
                BUILDER.EXPECTED_PLATEAU_DESIGN_FINGERPRINT
            ),
            "candidate_ids": list(BUILDER.PLATEAU_CELL_IDS),
            "sessions": dates,
            "failures": [],
        },
    )
    write_parquet(plateau_eval / "requested_calendar.parquet", calendar)
    plateau_effect_rows = []
    failing_candidate = BUILDER.PLATEAU_CELL_IDS[0]
    for date_index, date in enumerate(dates):
        for candidate_id in BUILDER.PLATEAU_CELL_IDS:
            normal_active = date_index > 0
            active = normal_active and (
                plateau_pass
                or candidate_id != failing_candidate
                or date_index <= 59
            )
            plateau_effect_rows.append(
                {
                    "Date": date,
                    "candidate_id": candidate_id,
                    "condition_n_z": 100 if active else 0,
                    "realized_selection_rate": 0.001 if active else 0.0,
                    "effect_active": active,
                    "daily_delta_z": 0.1 if active else None,
                    "plateau_design_fingerprint": (
                        BUILDER.EXPECTED_PLATEAU_DESIGN_FINGERPRINT
                    ),
                }
            )
    plateau_effects = pl.DataFrame(plateau_effect_rows)
    write_parquet(plateau_eval / "daily_effects.parquet", plateau_effects)
    plateau_cells = []
    for candidate_id in BUILDER.PLATEAU_CELL_IDS:
        active_sessions = (
            len(dates) - 1
            if plateau_pass or candidate_id != failing_candidate
            else 59
        )
        plateau_cells.append(
            {
                "period": "full",
                "candidate_id": candidate_id,
                "calendar_sessions": len(dates),
                "active_sessions": active_sessions,
                "effect_valid_events": active_sessions * 100,
                "equal_active_session_delta_z": 0.10,
                "selection_rate_equal_session_mean": (
                    active_sessions * 0.001 / len(dates)
                ),
                "support_floor_pass": active_sessions >= 60,
            }
        )
    write_parquet(
        plateau_eval / "cell_summary.parquet", pl.DataFrame(plateau_cells)
    )
    plateau_year_rows = []
    for period in ("2025", "2026"):
        period_dates = [date for date in dates if date.startswith(period)]
        for candidate_id in BUILDER.PLATEAU_CELL_IDS:
            detail = plateau_effects.filter(
                (pl.col("candidate_id") == candidate_id)
                & pl.col("Date").is_in(period_dates)
            )
            active_sessions = int(detail["effect_active"].sum() or 0)
            plateau_year_rows.append(
                {
                    "period": period,
                    "candidate_id": candidate_id,
                    "calendar_sessions": len(period_dates),
                    "active_sessions": active_sessions,
                    "effect_valid_events": active_sessions * 100,
                    "equal_active_session_delta_z": (
                        0.10 if active_sessions else None
                    ),
                    "selection_rate_equal_session_mean": float(
                        detail["realized_selection_rate"].mean()
                    ),
                }
            )
    write_parquet(
        plateau_eval / "year_summary.parquet",
        pl.DataFrame(plateau_year_rows),
    )
    write_parquet(
        plateau_eval / "fwer_audit.parquet",
        pl.DataFrame(
            [
                {
                    "candidate_id": candidate_id,
                    "block_length_requested": block_length,
                    "primary_block_length": block_length == 10,
                    "calendar_sessions": len(dates),
                    "active_sessions": (
                        len(dates) - 1
                        if plateau_pass or candidate_id != failing_candidate
                        else 59
                    ),
                    "point_estimate": 0.10,
                    "family_alpha": 0.10,
                    "simultaneous_confidence_level": 0.90,
                    "raw_trial_count": 27,
                    "bootstrap_repetitions": 5_000,
                    "joint_valid_bootstrap_repetitions": 5_000,
                    "joint_valid_bootstrap_rate": 1.0,
                    "seed": 20_260_827,
                    "method": (
                        "circular_moving_block_single_step_max_t_one_sided"
                    ),
                    "simultaneous_lower_bound": 0.08,
                }
                for candidate_id in BUILDER.PLATEAU_CELL_IDS
                for block_length in (5, 10, 20)
            ]
        ),
    )
    neighbor_rows = [
        {
            "period": period,
            "neighbor_id": neighbor_id,
            "center_effect": 0.10,
            "neighbor_effect": 0.10,
            "neighbor_qualifies": True,
            "period_local_sign_and_scale_pass": True,
        }
        for period in ("full", "2025", "2026")
        for neighbor_id in BUILDER.PLATEAU_NEIGHBOR_IDS
    ]
    write_parquet(
        plateau_eval / "neighbor_diagnostics.parquet",
        pl.DataFrame(neighbor_rows),
    )
    write_parquet(
        plateau_eval / "monotonicity.parquet",
        pl.DataFrame(
            [
                {
                    "period": period,
                    "factor": factor,
                    "factor_level_non_decreasing": True,
                }
                for period in ("full", "2025", "2026")
                for factor in (
                    "depth_ratio_lte",
                    "gap_ticks_gte",
                    "bear_score_gte",
                )
            ]
        ),
    )
    write_parquet(
        plateau_eval / "slice_reversals.parquet",
        pl.DataFrame(
            [
                {
                    "period": period,
                    "factor": factor,
                    "slice_id": f"slice_{slice_index}",
                    "transition_index": transition,
                    "slice_reversal": False,
                }
                for period in ("full", "2025", "2026")
                for factor in (
                    "depth_ratio_lte",
                    "gap_ticks_gte",
                    "bear_score_gte",
                )
                for slice_index in range(9)
                for transition in (0, 1)
            ]
        ),
    )
    write_parquet(
        plateau_eval / "trial_ledger.parquet",
        pl.DataFrame({"trial_id": list(BUILDER.PLATEAU_CELL_IDS)}),
    )
    input_hashes = {
        "daily_aggregates": sha256(plateau_run / "daily_stats.parquet"),
        "plateau_run_manifest": sha256(plateau_run / "run_manifest.json"),
        "requested_calendar": sha256(calendar_dir / "requested_calendar.parquet"),
        "requested_calendar_manifest": sha256(
            calendar_dir / "requested_calendar.json"
        ),
        "threshold_contract": sha256(study / "threshold_plateau_contract.json"),
    }
    write_json(
        plateau_eval / "plateau_decision.json",
        {
            "contract": BUILDER.PLATEAU_DECISION_CONTRACT,
            "plateau_contract_id": BUILDER.PLATEAU_CONTRACT,
            "family_id": BUILDER.PLATEAU_FAMILY_ID,
            "family_version": BUILDER.PLATEAU_FAMILY_VERSION,
            "center_candidate_id": BUILDER.PLATEAU_CENTER_ID,
            "input_artifact_sha256": input_hashes,
            "diagnostic_plateau_pass": plateau_pass,
            "decision_status": (
                "diagnostic_plateau_pass"
                if plateau_pass
                else "diagnostic_plateau_fail"
            ),
            "production_promotion_authorized": False,
            "failed_gate_ids": (
                []
                if plateau_pass
                else ["P1_support_floor_each_cell"]
            ),
            "gates": {
                "P1_support_floor_each_cell": {
                    "decision_role": "explicit_gate",
                    "pass": plateau_pass,
                },
                "P2_center_family_uncertainty": {
                    "decision_role": "explicit_gate",
                    "pass": True,
                },
                "P3_full_period_local_sign_and_scale": {
                    "decision_role": "explicit_gate",
                    "pass": True,
                },
            },
            "mandatory_disclosures": {
                "year_local_sign_and_scale": {
                    "decision_role": "mandatory_disclosure_not_overall_gate",
                    "cross_year_both_pass": True,
                    "period_pass": {
                        "full": True,
                        "2025": True,
                        "2026": True,
                    },
                },
                "factor_level_monotonicity": {
                    "all_factor_periods_non_decreasing": True,
                    "failed_factor_periods": [],
                },
                "slice_reversals": {
                    "comparison_rows": 162,
                    "reversal_rows": 0,
                },
                "prevalence_and_support": {
                    "periods": ["full", "2025", "2026"]
                },
            },
            "warnings": ["outcome-informed freeze"],
        },
    )
    output_names = [
        name
        for name in BUILDER.PLATEAU_EVALUATION_FILES
        if name != "run_manifest.json"
    ]
    output_artifacts = {
        name: {
            "size": (plateau_eval / name).stat().st_size,
            "sha256": sha256(plateau_eval / name),
        }
        for name in output_names
    }
    write_json(
        plateau_eval / "run_manifest.json",
        {
            "contract": BUILDER.PLATEAU_EVALUATOR_CONTRACT,
            "plateau_contract_id": BUILDER.PLATEAU_CONTRACT,
            "family_id": BUILDER.PLATEAU_FAMILY_ID,
            "family_version": BUILDER.PLATEAU_FAMILY_VERSION,
            "center_candidate_id": BUILDER.PLATEAU_CENTER_ID,
            "frozen_contract_sha256": sha256(
                study / "threshold_plateau_contract.json"
            ),
            "threshold_design_fingerprint": (
                BUILDER.EXPECTED_PLATEAU_DESIGN_FINGERPRINT
            ),
            "decision_contract": BUILDER.PLATEAU_DECISION_CONTRACT,
            "estimand": "shared_pool_equal_active_session_delta_z",
            "zero_event_policy": "undefined_not_zero",
            "production_promotion_authorized": False,
            "diagnostic_plateau_pass": plateau_pass,
            "decision_status": (
                "diagnostic_plateau_pass"
                if plateau_pass
                else "diagnostic_plateau_fail"
            ),
            "candidate_ids": list(BUILDER.PLATEAU_CELL_IDS),
            "input_artifact_sha256": input_hashes,
            "output_artifacts": output_artifacts,
        },
    )

    return BUILDER.DecisionPaths(
        project_root=project,
        study_dir=study,
        calendar_dir=calendar_dir,
        lineage_dir=lineage_dir,
        positive_control=positive_path,
        inference_dir=inference_dir,
        secondary_dir=secondary_dir,
        artifact_oos_dir=oos_dir,
        historical_exposure=historical_path,
        plateau_run_dir=plateau_run,
        plateau_evaluation_dir=plateau_eval,
        output_dir=output_dir,
    )


def refresh_plateau_output_identity(
    paths: BUILDER.DecisionPaths, name: str
) -> None:
    manifest_path = paths.plateau_evaluation_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = paths.plateau_evaluation_dir / name
    manifest["output_artifacts"][name] = {
        "size": artifact.stat().st_size,
        "sha256": sha256(artifact),
    }
    write_json(manifest_path, manifest)


class FinalDecisionBuilderTest(unittest.TestCase):
    def test_all_diagnostics_pass_still_cannot_authorize_deployment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary), plateau_pass=True)
            result = BUILDER.build_decision(paths)
            decision = result.decision
            self.assertEqual(decision["overall_status"], "RESEARCH_SHADOW_ONLY")
            self.assertFalse(decision["deployment_authorized"])
            self.assertFalse(decision["production_promotion_authorized"])
            self.assertIsNone(decision["selected_replacement_condition"])
            for gate_id in (
                "G1_support",
                "G2_primary_bias",
                "G3_temporal_stability",
            ):
                self.assertEqual(
                    decision["gates"][gate_id][
                        "prospective_deployment_status"
                    ],
                    "UNAVAILABLE_NOT_EVALUATED",
                )
            self.assertEqual(
                decision["gates"]["G6_evidence_class"]["status"],
                "FAIL_CLOSED_NO_PROSPECTIVE_FIRST_LIVE_ARTIFACT",
            )

    def test_g4_statistical_pass_does_not_claim_economic_materiality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary))
            gate = BUILDER.build_decision(paths).decision["gates"]["G4_parsimony"]
            self.assertEqual(gate["statistical_parsimony_status"], "PASS_FROZEN_RULE")
            self.assertEqual(
                gate["economic_materiality_status"],
                "NOT_ESTABLISHED_NO_PREREGISTERED_MINIMUM",
            )
            self.assertTrue(
                all(
                    row["reduced_support_on_common_periods"]
                    > row["full_support_on_common_periods"]
                    for row in gate["contrasts"]
                )
            )

    def test_g3_reports_block_length_conclusion_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary))
            interval_path = (
                paths.secondary_dir / "temporal_basic_intervals.parquet"
            )
            intervals = pl.read_parquet(interval_path).with_columns(
                pl.when(
                    (pl.col("block_length_requested") == 5)
                    & (pl.col("block_id") == "T03")
                )
                .then(pl.lit(-0.01))
                .otherwise(pl.col("one_sided_90_ucb"))
                .alias("one_sided_90_ucb")
            )
            write_parquet(interval_path, intervals)
            gate = BUILDER.build_decision(paths).decision["gates"][
                "G3_temporal_stability"
            ]
            self.assertEqual(
                gate["retrospective_diagnostic_status"],
                "PASS_DIAGNOSTIC_ONLY",
            )
            self.assertEqual(gate["sensitivity_conclusion_change_lengths"], [5])
            self.assertFalse(
                gate["block_length_sensitivity"]["5"][
                    "diagnostic_rule_pass"
                ]
            )

    def test_one_day_artifact_oos_is_separate_and_not_gate_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary))
            result = BUILDER.build_decision(paths)
            oos = result.report_data["artifact_oos_20260826"]
            self.assertEqual(oos["status"], "DESCRIPTIVE_ONLY_NOT_GATE_EVIDENCE")
            self.assertLess(oos["anchor"]["daily_delta_z"], 0.0)
            self.assertEqual(
                result.decision["evidence_classes"]["artifact_oos_non_pristine"][
                    "sessions"
                ],
                1,
            )

    def test_plateau_fail_is_preserved_but_overall_remains_shadow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary), plateau_pass=False)
            result = BUILDER.build_decision(paths)
            gate = result.decision["gates"]["G7_threshold_plateau"]
            self.assertEqual(gate["retrospective_diagnostic_status"], "FAIL_DIAGNOSTIC")
            self.assertEqual(result.decision["overall_status"], "RESEARCH_SHADOW_ONLY")
            self.assertFalse(result.decision["deployment_authorized"])

    def test_missing_plateau_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary))
            (paths.plateau_evaluation_dir / "slice_reversals.parquet").unlink()
            with self.assertRaisesRegex(
                BUILDER.DecisionArtifactError, "required artifacts are missing"
            ):
                BUILDER.build_decision(paths)

    def test_recorded_reference_fingerprint_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary))
            payload = json.loads(paths.positive_control.read_text(encoding="utf-8"))
            record = payload["fingerprints"]["positive_source"]
            source = paths.project_root / record["path"]
            source.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(
                BUILDER.DecisionArtifactError, "recorded SHA-256 drifted"
            ):
                BUILDER.build_decision(paths)

    def test_missing_required_value_is_not_inferred(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary))
            payload = json.loads(paths.positive_control.read_text(encoding="utf-8"))
            del payload["canonical_reference"]["preregistered_cell"][
                "equal_day_delta_z"
            ]
            write_json(paths.positive_control, payload)
            with self.assertRaisesRegex(
                BUILDER.DecisionArtifactError, "value is missing"
            ):
                BUILDER.build_decision(paths)

    def test_duplicate_or_missing_factorial_cell_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary))
            summary_path = paths.inference_dir / "cell_summary.parquet"
            summary = pl.read_parquet(summary_path)
            tampered = pl.concat([summary.head(7), summary.head(1)])
            write_parquet(summary_path, tampered)
            with self.assertRaisesRegex(
                BUILDER.DecisionArtifactError, "cell summary: duplicate keys"
            ):
                BUILDER.build_decision(paths)

    def test_wrong_frozen_loo_mapping_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary))
            loo_path = paths.inference_dir / "loo_summary.parquet"
            loo = pl.read_parquet(loo_path).with_columns(
                pl.when(pl.col("contrast_id") == "G_increment")
                .then(pl.lit("d0_g0_b0"))
                .otherwise(pl.col("reduced_cell_id"))
                .alias("reduced_cell_id")
            )
            write_parquet(loo_path, loo)
            with self.assertRaisesRegex(
                BUILDER.DecisionArtifactError,
                "inference G_increment reduced_cell_id drifted",
            ):
                BUILDER.build_decision(paths)

    def test_stage1_summary_must_match_daily_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary))
            summary_path = paths.inference_dir / "cell_summary.parquet"
            summary = pl.read_parquet(summary_path).with_columns(
                pl.when(pl.col("cell_id") == BUILDER.ANCHOR_ID)
                .then(pl.col("equal_day_delta_z") + 0.01)
                .otherwise(pl.col("equal_day_delta_z"))
                .alias("equal_day_delta_z")
            )
            write_parquet(summary_path, summary)
            with self.assertRaisesRegex(
                BUILDER.DecisionArtifactError,
                "summary disagrees with detailed evidence",
            ):
                BUILDER.build_decision(paths)

    def test_loo_summary_must_match_daily_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary))
            summary_path = paths.inference_dir / "loo_summary.parquet"
            summary = pl.read_parquet(summary_path).with_columns(
                pl.when(pl.col("contrast_id") == "G_increment")
                .then(pl.col("paired_increment_delta_z") + 0.01)
                .otherwise(pl.col("paired_increment_delta_z"))
                .alias("paired_increment_delta_z")
            )
            write_parquet(summary_path, summary)
            with self.assertRaisesRegex(
                BUILDER.DecisionArtifactError,
                "summary disagrees with detailed evidence",
            ):
                BUILDER.build_decision(paths)

    def test_secondary_bp_summary_must_match_daily_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary))
            summary_path = paths.secondary_dir / "equal_active_bp.parquet"
            summary = pl.read_parquet(summary_path).with_columns(
                pl.when(pl.col("cell_id") == BUILDER.ANCHOR_ID)
                .then(pl.col("equal_active_raw_bp") + 1.0)
                .otherwise(pl.col("equal_active_raw_bp"))
                .alias("equal_active_raw_bp")
            )
            write_parquet(summary_path, summary)
            with self.assertRaisesRegex(
                BUILDER.DecisionArtifactError,
                "summary disagrees with detailed evidence",
            ):
                BUILDER.build_decision(paths)

    def test_temporal_boundary_must_match_deterministic_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary))
            temporal_path = paths.inference_dir / "temporal_blocks.parquet"
            temporal = pl.read_parquet(temporal_path).with_columns(
                pl.when(pl.col("block_id") == "T02")
                .then(pl.lit("20250101"))
                .otherwise(pl.col("block_start"))
                .alias("block_start")
            )
            write_parquet(temporal_path, temporal)
            with self.assertRaisesRegex(
                BUILDER.DecisionArtifactError,
                "boundary or support drifted",
            ):
                BUILDER.build_decision(paths)

    def test_plateau_neighbor_must_match_period_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary))
            neighbor_path = (
                paths.plateau_evaluation_dir
                / "neighbor_diagnostics.parquet"
            )
            neighbors = pl.read_parquet(neighbor_path).with_columns(
                pl.when(
                    (pl.col("period") == "full")
                    & (
                        pl.col("neighbor_id")
                        == BUILDER.PLATEAU_NEIGHBOR_IDS[0]
                    )
                )
                .then(pl.col("neighbor_effect") + 0.01)
                .otherwise(pl.col("neighbor_effect"))
                .alias("neighbor_effect")
            )
            write_parquet(neighbor_path, neighbors)
            refresh_plateau_output_identity(
                paths, "neighbor_diagnostics.parquet"
            )
            with self.assertRaisesRegex(
                BUILDER.DecisionArtifactError,
                "neighbor-table effect: summary disagrees",
            ):
                BUILDER.build_decision(paths)

    def test_plateau_design_fingerprint_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary))
            manifest_path = paths.plateau_run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["threshold_design_fingerprint"] = "b" * 64
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                BUILDER.DecisionArtifactError,
                "plateau run design fingerprint drifted",
            ):
                BUILDER.build_decision(paths)

    def test_plateau_overall_pass_must_match_bound_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary))
            decision_path = paths.plateau_evaluation_dir / "plateau_decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["gates"]["P1_support_floor_each_cell"]["pass"] = False
            write_json(decision_path, decision)
            refresh_plateau_output_identity(paths, "plateau_decision.json")
            with self.assertRaisesRegex(
                BUILDER.DecisionArtifactError,
                "P1_support_floor_each_cell contradicts bound evidence",
            ):
                BUILDER.build_decision(paths)

    def test_non_finite_decision_metric_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_fixture(Path(temporary))
            bp_path = paths.secondary_dir / "equal_active_bp.parquet"
            bp = pl.read_parquet(bp_path).with_columns(
                pl.when(pl.col("cell_id") == BUILDER.ANCHOR_ID)
                .then(pl.lit(float("nan")))
                .otherwise(pl.col("equal_active_residual_bp"))
                .alias("equal_active_residual_bp")
            )
            write_parquet(bp_path, bp)
            with self.assertRaisesRegex(
                BUILDER.DecisionArtifactError, "value is not finite"
            ):
                BUILDER.build_decision(paths)

    def test_writer_rejects_non_finite_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            result = BUILDER.DecisionResult(
                decision={"bad": float("nan")}, report_data={"ok": True}
            )
            with self.assertRaisesRegex(
                BUILDER.DecisionArtifactError, "strict finite JSON"
            ):
                BUILDER.write_result(result, output)

    def test_writer_emits_only_decision_and_report_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = create_fixture(root)
            result = BUILDER.build_decision(paths)
            BUILDER.write_result(result, paths.output_dir)
            self.assertEqual(
                {path.name for path in paths.output_dir.iterdir()},
                {"decision.json", "report_data.json"},
            )
            decision = json.loads(
                (paths.output_dir / "decision.json").read_text(encoding="utf-8")
            )
            self.assertFalse(decision["deployment_authorized"])
            self.assertEqual(decision["overall_status"], "RESEARCH_SHADOW_ONLY")


if __name__ == "__main__":
    unittest.main()
