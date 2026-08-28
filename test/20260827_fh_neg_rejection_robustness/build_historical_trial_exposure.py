"""Build the reproducible lower-bound historical trial-exposure inventory.

This audit intentionally counts only configurations recoverable from checked-in
code or JSON. It does not infer informal notebook, chat, or discarded trials.
Condition/direction delta-z tests, model components, model-gate thresholds, and
execution diagnostics remain separate families because they do not share an
estimand or selection objective.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT = Path(__file__).with_name("historical_trial_exposure.json")


PATHS = {
    "legacy_fillhigh_scan": (
        "src/research/timeSeries/test/20260821_condition_explore/"
        "fillhigh_explore.py"
    ),
    "legacy_fillhigh_report": (
        "src/research/timeSeries/test/20260821_condition_explore/"
        "fillhigh_report.py"
    ),
    "legacy_fh2_short_scan": (
        "src/research/timeSeries/test/20260821_fh15_line/fh2_explore.py"
    ),
    "legacy_fh2_short_report": (
        "src/research/timeSeries/test/20260821_fh15_line/fh2_report.py"
    ),
    "legacy_fh2_long_scan": (
        "src/research/timeSeries/test/20260821_fh15_line/fh2_long.py"
    ),
    "legacy_fh15_builder": (
        "src/research/timeSeries/test/20260821_fh15_line/fh15_build.py"
    ),
    "legacy_fh15_models": (
        "src/research/timeSeries/test/20260821_fh15_line/stage_fh15.py"
    ),
    "legacy_fh15_thresholds": (
        "src/research/timeSeries/test/20260821_fh15_line/fh15_portfolio.py"
    ),
    "legacy_fh15_twin_split": (
        "src/research/timeSeries/test/20260821_fh15_line/twin_split.py"
    ),
    "legacy_fh15_overlap": (
        "src/research/timeSeries/test/20260821_fh15_line/"
        "same_trade_overlap.py"
    ),
    "legacy_digest_report": (
        "src/research/timeSeries/report/studies/"
        "20260819_20260821_legacy_digest/REPORT.md"
    ),
    "legacy_experiment_table": (
        "src/research/timeSeries/report/studies/"
        "20260819_20260821_legacy_digest/EXPERIMENT_TABLE.md"
    ),
    "feature_profile": (
        "src/research/timeSeries/framework/profiles/filllots_at_high.py"
    ),
    "feature_bias_spec": (
        "src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/"
        "study_spec.json"
    ),
    "feature_bias_runner": (
        "src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/"
        "run_sweep.py"
    ),
    "feature_bias_promotion": (
        "src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/"
        "select_promotions.py"
    ),
    "feature_bias_reporter": (
        "src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/"
        "report_sweep.py"
    ),
    "feature_bias_dense_reporter": (
        "src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/"
        "report_dense.py"
    ),
    "feature_bias_complement_reporter": (
        "src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/"
        "report_dense_complement.py"
    ),
    "feature_bias_sparse_short_manifest": (
        "src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/"
        "artifacts/sparse_short_v1/run_manifest.json"
    ),
    "feature_bias_sparse_long_manifest": (
        "src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/"
        "artifacts/sparse_long_v1/run_manifest.json"
    ),
    "feature_bias_promotion_artifact": (
        "src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/"
        "artifacts/sparse_report_v1/promotion.json"
    ),
    "feature_bias_dense_short_manifest": (
        "src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/"
        "artifacts/dense_short_neg_rejection_v1/run_manifest.json"
    ),
    "feature_bias_dense_long_manifest": (
        "src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/"
        "artifacts/dense_long_pos_breakout_v1/run_manifest.json"
    ),
    "feature_bias_dense_decision_artifact": (
        "src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/"
        "artifacts/dense_report_v1/decision.json"
    ),
    "signal_spec": (
        "src/research/timeSeries/test/20260826_filllots_at_high_signal/"
        "study_spec.json"
    ),
    "signal_study": (
        "src/research/timeSeries/test/20260826_filllots_at_high_signal/study.py"
    ),
    "signal_runner": (
        "src/research/timeSeries/test/20260826_filllots_at_high_signal/"
        "run_model_backtest.py"
    ),
    "signal_verifier": (
        "src/research/timeSeries/test/20260826_filllots_at_high_signal/"
        "verify_run.py"
    ),
    "signal_overlap": (
        "src/research/timeSeries/test/20260826_filllots_at_high_signal/"
        "audit_negfill_overlap.py"
    ),
    "signal_run_manifest": (
        "src/research/timeSeries/test/20260826_filllots_at_high_signal/"
        "artifacts/backtest_v1/run_manifest.json"
    ),
    "signal_selection_artifact": (
        "src/research/timeSeries/test/20260826_filllots_at_high_signal/"
        "artifacts/backtest_v1/selection_decision.json"
    ),
    "signal_variant_summary_artifact": (
        "src/research/timeSeries/test/20260826_filllots_at_high_signal/"
        "artifacts/backtest_v1/variant_period_summary.parquet"
    ),
    "signal_fee_sensitivity_artifact": (
        "src/research/timeSeries/test/20260826_filllots_at_high_signal/"
        "artifacts/backtest_v1/fee_sensitivity.parquet"
    ),
}


def _path(key: str) -> Path:
    path = PROJECT_ROOT / PATHS[key]
    if not path.is_file():
        raise FileNotFoundError(f"missing audit source {key}: {path}")
    return path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _source_records(keys: list[str]) -> list[dict[str, str]]:
    records = []
    for key in keys:
        path = _path(key)
        records.append(
            {
                "role": key,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256_bytes(path.read_bytes()),
            }
        )
    return records


def _source_bundle_fingerprint(records: list[dict[str, str]]) -> str:
    return _sha256_json(
        [{"path": row["path"], "sha256": row["sha256"]} for row in records]
    )


def _tree(key: str) -> ast.AST:
    path = _path(key)
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literal_assignment(key: str, name: str) -> Any:
    matches: list[Any] = []
    for node in ast.walk(_tree(key)):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        value = ast.literal_eval(node.value)
        matches.append(value)
    if len(matches) != 1:
        raise ValueError(f"{key}: expected one literal assignment {name}, got {len(matches)}")
    return matches[0]


def _dict_assignment_keys(key: str, name: str) -> list[str]:
    matches: list[list[str]] = []
    for node in ast.walk(_tree(key)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        keys = []
        for item in node.value.keys:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                raise ValueError(f"{key}: {name} contains a non-literal key")
            keys.append(item.value)
        matches.append(keys)
    if len(matches) != 1:
        raise ValueError(f"{key}: expected one dict assignment {name}, got {len(matches)}")
    return matches[0]


def _loop_literal_values(key: str, target_name: str) -> list[Any]:
    matches: list[list[Any]] = []
    for node in ast.walk(_tree(key)):
        if not isinstance(node, ast.For):
            continue
        if not isinstance(node.target, ast.Name) or node.target.id != target_name:
            continue
        try:
            value = ast.literal_eval(node.iter)
        except (ValueError, TypeError):
            continue
        if isinstance(value, (list, tuple)):
            matches.append(list(value))
    if len(matches) != 1:
        raise ValueError(
            f"{key}: expected one literal loop for {target_name}, got {len(matches)}"
        )
    return matches[0]


def _read_json(key: str) -> dict[str, Any]:
    return json.loads(_path(key).read_text(encoding="utf-8"))


def _family(
    *,
    family_id: str,
    category: str,
    estimand_or_objective: str,
    definitions: dict[str, Any],
    lower_bound: dict[str, Any],
    source_keys: list[str],
    selection_role: str,
    delta_z_max_stat_membership: str,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    sources = _source_records(source_keys)
    return {
        "family_id": family_id,
        "category": category,
        "estimand_or_objective": estimand_or_objective,
        "selection_role": selection_role,
        "delta_z_max_stat_membership": delta_z_max_stat_membership,
        "known_reconstructable_lower_bound": lower_bound,
        "definitions": definitions,
        "definition_fingerprint": _sha256_json(definitions),
        "sources": sources,
        "source_bundle_fingerprint": _source_bundle_fingerprint(sources),
        "limitations": limitations or [],
    }


def build_payload() -> dict[str, Any]:
    fillhigh_ids = _dict_assignment_keys("legacy_fillhigh_scan", "conds")
    fh2_short_ids = _dict_assignment_keys("legacy_fh2_short_scan", "conds")
    fh2_long_ids = _dict_assignment_keys("legacy_fh2_long_scan", "conds")
    fh15_model_runs = [list(row) for row in _literal_assignment("legacy_fh15_models", "runs")]
    fh15_qn = _loop_literal_values("legacy_fh15_thresholds", "qn")
    fh15_qm = _loop_literal_values("legacy_fh15_thresholds", "qm")

    feature_spec = _read_json("feature_bias_spec")
    feature_ids = list(feature_spec["candidate_family"]["ids"])
    profile_ids = list(_literal_assignment("feature_profile", "CANDIDATE_IDS"))
    if feature_ids != profile_ids:
        raise ValueError("20260826 feature-bias spec and profile registry differ")

    signal_spec = _read_json("signal_spec")
    signal_selectable = list(_literal_assignment("signal_runner", "SELECTABLE_VARIANTS"))
    signal_gate_variants = list(signal_spec["gate_variants"])
    if signal_selectable != signal_gate_variants:
        raise ValueError("20260826 signal spec and selectable gate variants differ")
    signal_all_variants = _dict_assignment_keys("signal_runner", "gates")
    signal_fee_grid = list(_literal_assignment("signal_runner", "FEE_GRID_BP"))
    signal_models = [
        {
            "component": "timing",
            "target": _literal_assignment("signal_study", "TIMING_TARGET"),
            "lookback_event_dates": _literal_assignment(
                "signal_study", "TIMING_LOOKBACK_DAYS"
            ),
        },
        {
            "component": "selection",
            "target": _literal_assignment("signal_study", "SELECTION_TARGET"),
            "lookback_event_dates": _literal_assignment(
                "signal_study", "SELECTION_LOOKBACK_DAYS"
            ),
        },
    ]

    families = [
        _family(
            family_id="legacy_20260821_fillhigh_condition_direction",
            category="condition_selector_and_direction_tests",
            estimand_or_objective="shared sampled-pool delta_z under short and long labels",
            definitions={
                "candidate_ids": fillhigh_ids,
                "directions": ["short", "long"],
                "date_end": "20260817",
                "session_step": 5,
            },
            lower_bound={
                "selector_configurations": len(fillhigh_ids),
                "direction_tests": len(fillhigh_ids) * 2,
            },
            source_keys=[
                "legacy_fillhigh_scan",
                "legacy_fillhigh_report",
                "legacy_digest_report",
                "legacy_experiment_table",
            ],
            selection_role="Outcome-inspected condition family from which FH_neg15_chg was continued as FH15.",
            delta_z_max_stat_membership="Historical delta_z exposure, but not exchangeable with the new D/G/B family because the selector, sampler, pool implementation, and report contract differ.",
        ),
        _family(
            family_id="legacy_20260821_fh2_short_condition",
            category="condition_selector_and_direction_tests",
            estimand_or_objective="sampled-pool short delta_z",
            definitions={
                "candidate_ids": fh2_short_ids,
                "directions": ["short"],
                "date_end": "20260817",
                "session_step": 5,
            },
            lower_bound={
                "selector_configurations": len(fh2_short_ids),
                "direction_tests": len(fh2_short_ids),
            },
            source_keys=[
                "legacy_fh2_short_scan",
                "legacy_fh2_short_report",
                "legacy_digest_report",
                "legacy_experiment_table",
            ],
            selection_role="Outcome-inspected redesign/falsification family; no member is treated as a pristine confirmation.",
            delta_z_max_stat_membership="Separate historical short-delta_z family; disclose as prior exposure rather than merge into the new eight-cell max-stat distribution.",
        ),
        _family(
            family_id="legacy_20260821_fh2_long_condition_labels",
            category="condition_selector_direction_and_label_tests",
            estimand_or_objective="long delta_z under taker A1 and optimistic maker B1 entry labels",
            definitions={
                "candidate_ids": fh2_long_ids,
                "directions": ["long"],
                "label_entry_assumptions": ["taker_A1", "optimistic_maker_B1"],
                "date_end": "20260817",
                "session_step": 5,
            },
            lower_bound={
                "selector_configurations": len(fh2_long_ids),
                "direction_tests": len(fh2_long_ids),
                "label_outcome_tests": len(fh2_long_ids) * 2,
            },
            source_keys=[
                "legacy_fh2_long_scan",
                "legacy_digest_report",
                "legacy_experiment_table",
            ],
            selection_role="Outcome-inspected long-side and entry-assumption family.",
            delta_z_max_stat_membership="Not a short delta_z family and not eligible for the new short D/G/B max-stat family.",
        ),
        _family(
            family_id="legacy_20260821_fh15_model_components",
            category="model_trials",
            estimand_or_objective="rolling prediction IC/model usefulness within the fixed FH_neg15 condition",
            definitions={
                "fixed_selector": "change event AND FillLots_atHigh/depth < -15",
                "model_runs": fh15_model_runs,
                "feature_screening": True,
                "ridge_alpha": 0.1,
                "max_position_per_stock": 500,
            },
            lower_bound={
                "executable_model_configurations": len(fh15_model_runs)
            },
            source_keys=[
                "legacy_fh15_builder",
                "legacy_fh15_models",
                "legacy_digest_report",
                "legacy_experiment_table",
            ],
            selection_role="Both executable target/lookback model configurations produced predictions that were available for outcome inspection.",
            delta_z_max_stat_membership="Excluded: model prediction trials are downstream of the condition estimand.",
            limitations=[
                "The stage_fh15.py docstring says five model runs, but the executable runs assignment contains two. This inventory counts only the two AST-reconstructable runs and does not guess whether three removed variants were executed elsewhere."
            ],
        ),
        _family(
            family_id="legacy_20260821_fh15_threshold_grid",
            category="threshold_trials",
            estimand_or_objective="tune-window net PnL sum after the FH15 rolling models",
            definitions={
                "timing_quantiles": fh15_qn,
                "selection_quantiles": fh15_qm,
                "cartesian_product": True,
                "tune_start": "2025-12-05",
                "tune_end": "2026-03-31",
                "minimum_tune_days_strictly_greater_than": 30,
            },
            lower_bound={
                "threshold_pairs": len(fh15_qn) * len(fh15_qm),
                "selected_pairs": 1,
            },
            source_keys=[
                "legacy_fh15_thresholds",
                "legacy_digest_report",
                "legacy_experiment_table",
            ],
            selection_role="One of the nine q-pairs was selected by observed tune-window PnL.",
            delta_z_max_stat_membership="Excluded: this is a downstream PnL-selected threshold family, not a condition delta_z family.",
        ),
        _family(
            family_id="legacy_20260821_fh15_execution_diagnostics",
            category="execution_and_portfolio_trials",
            estimand_or_objective="net daily PnL, Sharpe, capital-weighted BP, drawdown, and overlap diagnostics",
            definitions={
                "portfolio_constructs": [
                    "negFill_baseline_book",
                    "FH15_book",
                    "stacked_negFill_plus_FH15"
                ],
                "derived_comparator": "same-capital scaled negFill baseline",
                "post_hoc_fh15_partitions": ["twin", "unique"],
                "same_trade_definition": "same stock-day and either absolute ToRef difference <= 1% or absolute time difference <= 60 seconds",
                "fee_bp": 19.3,
            },
            lower_bound={
                "portfolio_constructs": 3,
                "derived_leverage_comparators": 1,
                "post_hoc_partitions": 2,
            },
            source_keys=[
                "legacy_fh15_thresholds",
                "legacy_fh15_twin_split",
                "legacy_fh15_overlap",
                "legacy_digest_report",
                "legacy_experiment_table",
            ],
            selection_role="Outcome-inspected execution and post-hoc decomposition views; counts are not independent strategy trials.",
            delta_z_max_stat_membership="Excluded: execution/portfolio outcomes use PnL and overlap objectives.",
        ),
        _family(
            family_id="feature_bias_20260826_selector_direction",
            category="condition_selector_and_direction_tests",
            estimand_or_objective="complete shared-pool equal-active-day delta_z",
            definitions={
                "candidate_registry": "filllots-at-high-feature-v1",
                "candidate_ids": feature_ids,
                "directions": ["short", "long"],
                "sparse_session_step": feature_spec["sparse_screen"]["session_step"],
                "within_direction_correction": feature_spec["candidate_family"]["correction"],
                "dense_rule": feature_spec["dense_check"],
            },
            lower_bound={
                "selector_configurations": len(feature_ids),
                "direction_tests": len(feature_ids) * 2,
                "within_direction_family_size": len(feature_ids),
                "joint_direction_candidate_family_size": len(feature_ids) * 2,
                "promotions_per_direction_max": feature_spec["sparse_screen"]["promotion_cap"],
            },
            source_keys=[
                "feature_profile",
                "feature_bias_spec",
                "feature_bias_runner",
                "feature_bias_promotion",
                "feature_bias_reporter",
                "feature_bias_dense_reporter",
                "feature_bias_complement_reporter",
                "feature_bias_sparse_short_manifest",
                "feature_bias_sparse_long_manifest",
                "feature_bias_promotion_artifact",
                "feature_bias_dense_short_manifest",
                "feature_bias_dense_long_manifest",
                "feature_bias_dense_decision_artifact",
            ],
            selection_role="All 12 selectors were outcome-inspected under both directions; dense and complement checks repeat promoted candidate-direction hypotheses rather than create new selector definitions.",
            delta_z_max_stat_membership="Known historical shared-pool delta_z exposure. Keep as its own 12-per-direction/24-joint historical family; do not silently treat fh_neg_rejection as an ex-ante single test.",
        ),
        _family(
            family_id="signal_20260826_model_components",
            category="model_trials",
            estimand_or_objective="causal rolling timing and cross-sectional selection predictions inside fixed fh_neg_rejection",
            definitions={
                "selector": signal_spec["condition"],
                "model_components": signal_models,
                "ridge_alpha": signal_spec["model"]["alpha"],
                "feature_manifest": signal_spec["model"]["feature_manifest"],
                "feature_screening": signal_spec["model"]["feature_screening"],
            },
            lower_bound={
                "model_components": len(signal_models),
                "alternative_model_winner_searches": 0,
            },
            source_keys=[
                "signal_spec",
                "signal_study",
                "signal_runner",
                "signal_verifier",
                "signal_run_manifest",
            ],
            selection_role="Two sequential model components were run; code does not define an alpha/lookback model grid in this study.",
            delta_z_max_stat_membership="Excluded: downstream model components cannot be pooled with condition selector delta_z trials.",
        ),
        _family(
            family_id="signal_20260826_threshold_and_gate_variants",
            category="threshold_and_model_gate_trials",
            estimand_or_objective="development-period calendar-filled net daily Sharpe at 19.3bp",
            definitions={
                "threshold_configuration": signal_spec["threshold"],
                "all_evaluated_variants": signal_all_variants,
                "selectable_variants": signal_selectable,
                "selection_rule": signal_spec["selection_rule"],
            },
            lower_bound={
                "fixed_threshold_configurations": 1,
                "threshold_components": 2,
                "evaluated_gate_variants": len(signal_all_variants),
                "selectable_gate_variants": len(signal_selectable),
                "selected_gate_variants": 1,
            },
            source_keys=[
                "signal_spec",
                "signal_study",
                "signal_runner",
                "signal_verifier",
                "signal_run_manifest",
                "signal_selection_artifact",
                "signal_variant_summary_artifact",
            ],
            selection_role="One of three model-gated variants was selected by development Sharpe; static_only was evaluated as a non-selectable baseline.",
            delta_z_max_stat_membership="Excluded: gate selection uses strategy Sharpe after model thresholds and execution filters.",
        ),
        _family(
            family_id="signal_20260826_execution_sensitivity",
            category="execution_trials",
            estimand_or_objective="selected-variant PnL and capital-weighted BP under fee and overlap views",
            definitions={
                "fee_grid_bp": signal_fee_grid,
                "fee_grid_applied_after_gate_selection": True,
                "fixed_execution": signal_spec["execution"],
                "overlap_audit": "selected FillHigh book versus negFill",
            },
            lower_bound={
                "fee_sensitivity_levels": len(signal_fee_grid),
                "selected_signal_execution_configurations": 1,
                "overlap_diagnostic_families": 1,
            },
            source_keys=[
                "signal_spec",
                "signal_study",
                "signal_runner",
                "signal_verifier",
                "signal_overlap",
                "signal_run_manifest",
                "signal_selection_artifact",
                "signal_variant_summary_artifact",
                "signal_fee_sensitivity_artifact",
            ],
            selection_role="Fee levels and overlap decompositions were inspected after the gate variant was selected; they are sensitivity and attribution views, not selector trials.",
            delta_z_max_stat_membership="Excluded: execution PnL/cost/overlap objectives differ from condition delta_z.",
        ),
    ]

    condition_families = [
        family
        for family in families
        if family["category"].startswith("condition_selector")
    ]
    arithmetic_selector_total = sum(
        int(family["known_reconstructable_lower_bound"]["selector_configurations"])
        for family in condition_families
    )
    arithmetic_direction_total = sum(
        int(family["known_reconstructable_lower_bound"]["direction_tests"])
        for family in condition_families
    )
    arithmetic_label_outcome_total = sum(
        int(
            family["known_reconstructable_lower_bound"].get(
                "label_outcome_tests",
                family["known_reconstructable_lower_bound"]["direction_tests"],
            )
        )
        for family in condition_families
    )

    generator = _source_records_for_generator()
    return {
        "contract": "fh_neg_rejection_historical_trial_exposure/v1",
        "study_id": "20260827_fh_neg_rejection_robustness",
        "generated_date": "2026-08-27",
        "generator": generator,
        "scope": {
            "included": [
                "checked-in 20260821 FillHigh/FH15 code and archived outcome summaries",
                "checked-in 20260826 FillLots-at-high feature-bias code/config",
                "checked-in 20260826 fh_neg_rejection signal code/config"
            ],
            "excluded": [
                "informal chat or notebook trials not reconstructable from checked-in files",
                "discarded or overwritten code variants",
                "a guessed total for unknown research attempts"
            ],
            "interpretation": "Every count is a code/config-reconstructable lower bound, not a claim that the historical search universe is complete."
        },
        "counting_rules": [
            "A selector configuration is counted once per literal condition definition within its source registry.",
            "A direction or label outcome test is counted separately when the same selector was evaluated under a distinct directional label or entry assumption.",
            "Sparse, dense, complement, period, and rerun views of the same candidate-direction definition are evaluation views, not new selector configurations.",
            "Model target/lookback components, threshold grids, gate variants, fee levels, and portfolio decompositions remain in their own families.",
            "Counts may be arithmetically summarized for inventory only; differing estimands and pipelines are not one exchangeable max-stat family."
        ],
        "families": families,
        "arithmetic_inventory_lower_bounds": {
            "condition_selector_configurations_across_distinct_historical_families": arithmetic_selector_total,
            "condition_direction_tests_across_distinct_historical_families": arithmetic_direction_total,
            "condition_label_outcome_tests_across_distinct_historical_families": arithmetic_label_outcome_total,
            "valid_single_delta_z_max_stat_family": False,
            "reason": "The arithmetic total crosses dates, selector grammars, samplers, pools, labels, and inference contracts. It is disclosure context only."
        },
        "implication_for_current_study": {
            "new_dgb_factorial_current_family_size": 8,
            "historical_exposure_status": "material and incompletely bounded above",
            "max_stat_rule": "Apply the current eight-cell blocked max-stat to the current family only, while disclosing these historical families as prior selection exposure.",
            "deployment_rule": "Neither the historical inventory nor an adjusted retrospective statistic creates pristine evidence; only Date > 20260827 can enter prospective deployment gates."
        },
    }


def _source_records_for_generator() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": _sha256_bytes(path.read_bytes()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_payload()
    output = args.output
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    allowed_root = Path(__file__).resolve().parent
    try:
        output.resolve().relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"output must stay inside {allowed_root}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
