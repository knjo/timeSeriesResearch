"""Fail-closed verification for the frozen FillHigh signal study artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from study import (
    ARTIFACT_DIR,
    BACKTEST_DIR,
    EVENT_DATASET_DIR,
    EVENT_INDEX_PATH,
    FEATURE_MANIFEST,
    HOLDOUT_START,
    MODEL_DIR,
    PER_STOCK_POSITION_CAP_WAN,
    PRICE_CAP,
    PROJECT_ROOT,
    STUDY_DIR,
    TAKER_FEE_BP,
    THRESHOLD_MIN_OBSERVATIONS,
    THRESHOLD_MIN_SESSIONS,
    TIMING_LOOKBACK_DAYS,
    data_roots,
)


KEY_COLUMNS = ("Date", "QuoteCode", "ChannelSeq")
EXPECTED_EVENT_ROWS = 6_443_810
EXPECTED_DATASET_ROWS = 855_594
EXPECTED_MODEL_ROWS = 852_863
EXPECTED_SELECTED_ROWS = 41_545
EXPECTED_EVENT_DATES = 377
EXPECTED_EMPTY_DATASET_DATES = ["20250407"]
EXPECTED_FEATURE_HASH = (
    "09a29ca300424620a1cbf6331e585072d91a9a6fa2c8dd7ea4e4f1d6e6559468"
)

TIMING_THRESHOLD_PATH = MODEL_DIR / "timing_thresholds.parquet"
SELECTION_THRESHOLD_PATH = MODEL_DIR / "selection_thresholds.parquet"
THRESHOLDED_PATH = MODEL_DIR / "thresholded_events.parquet"
MODEL_INPUT_PATH = MODEL_DIR / "model_input.parquet"
SELECTED_SIGNALS_PATH = BACKTEST_DIR / "selected_signals.parquet"
VARIANT_SUMMARY_PATH = BACKTEST_DIR / "variant_period_summary.parquet"
FEE_SENSITIVITY_PATH = BACKTEST_DIR / "fee_sensitivity.parquet"
SELECTION_DECISION_PATH = BACKTEST_DIR / "selection_decision.json"
RUN_MANIFEST_PATH = BACKTEST_DIR / "run_manifest.json"
SAMPLING_AUDIT_PATH = ARTIFACT_DIR / "sampling_audit_v1" / "audit.json"
OVERLAP_PATH = ARTIFACT_DIR / "negfill_overlap_v1" / "overlap.json"
OVERLAP_SUMMARY_PATH = (
    ARTIFACT_DIR / "negfill_overlap_v1" / "period_summary.parquet"
)
VERIFICATION_DIR = ARTIFACT_DIR / "verification_v1"
VERIFICATION_PATH = VERIFICATION_DIR / "verification.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Verifier:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, condition: bool, **detail: Any) -> None:
        self.checks.append(
            {"name": name, "status": "pass" if condition else "fail", **detail}
        )

    @property
    def passed(self) -> bool:
        return all(row["status"] == "pass" for row in self.checks)


def _verify_index_and_dataset(verifier: Verifier) -> None:
    event_manifest = _read_json(EVENT_INDEX_PATH.with_suffix(".manifest.json"))
    event_counts = (
        pl.scan_parquet(EVENT_INDEX_PATH)
        .select(
            pl.len().alias("rows"),
            pl.struct(KEY_COLUMNS).n_unique().alias("unique_keys"),
            pl.col("Date").n_unique().alias("dates"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    verifier.check(
        "event_index_exact_unique_counts",
        event_counts
        == {
            "rows": EXPECTED_EVENT_ROWS,
            "unique_keys": EXPECTED_EVENT_ROWS,
            "dates": EXPECTED_EVENT_DATES,
        },
        observed=event_counts,
    )
    verifier.check(
        "event_index_manifest_reproduced",
        event_manifest.get("row_count") == EXPECTED_EVENT_ROWS
        and event_manifest.get("unique_key_count") == EXPECTED_EVENT_ROWS
        and event_manifest.get("event_date_count") == EXPECTED_EVENT_DATES
        and event_manifest.get("selector_reproduction_verified") is True
        and event_manifest.get("unverified_dates") == [],
        selector_version=event_manifest.get("selector_version"),
    )

    dataset_manifest = _read_json(EVENT_DATASET_DIR / "_dataset_manifest.json")
    partitions = dataset_manifest["partitions"]
    manifest_rows = sum(part["row_count"] for part in partitions.values())
    manifest_empty = sorted(
        date for date, part in partitions.items() if part["row_count"] == 0
    )
    parquet_paths = sorted(EVENT_DATASET_DIR.glob("*.parquet"))
    actual_rows = 0
    duplicate_rows = 0
    actual_empty: list[str] = []
    for path in parquet_paths:
        keys = pl.read_parquet(path, columns=list(KEY_COLUMNS))
        actual_rows += len(keys)
        duplicate_rows += len(keys) - keys.n_unique(subset=list(KEY_COLUMNS))
        if keys.is_empty():
            actual_empty.append(path.stem)
    verifier.check(
        "event_dataset_partition_contract",
        len(partitions) == EXPECTED_EVENT_DATES
        and len(parquet_paths) == EXPECTED_EVENT_DATES
        and manifest_rows == EXPECTED_DATASET_ROWS
        and actual_rows == EXPECTED_DATASET_ROWS
        and duplicate_rows == 0
        and manifest_empty == EXPECTED_EMPTY_DATASET_DATES
        and actual_empty == EXPECTED_EMPTY_DATASET_DATES
        and dataset_manifest.get("schema_policy") == "union"
        and dataset_manifest.get("tick_bar_alignment") == "causal_completed_slot"
        and dataset_manifest.get("strict_cardinality") is True,
        partitions=len(parquet_paths),
        rows=actual_rows,
        duplicate_rows=duplicate_rows,
        empty_dates=actual_empty,
    )


def _verify_features_and_models(verifier: Verifier) -> None:
    model_input = pl.read_parquet(MODEL_INPUT_PATH)
    FEATURE_MANIFEST.validate_columns(model_input.columns)
    forbidden = sorted(
        name
        for name in FEATURE_MANIFEST.ordered_names
        if name.startswith("Future") or name in {"Close", "txf_to_1330_bp"}
    )
    verifier.check(
        "fixed_future_free_feature_manifest",
        len(FEATURE_MANIFEST.features) == 38
        and FEATURE_MANIFEST.fingerprint == EXPECTED_FEATURE_HASH
        and not forbidden,
        feature_count=len(FEATURE_MANIFEST.features),
        feature_hash=FEATURE_MANIFEST.fingerprint,
        forbidden=forbidden,
    )
    verifier.check(
        "model_input_exact_unique_counts",
        len(model_input) == EXPECTED_MODEL_ROWS
        and model_input.n_unique(subset=list(KEY_COLUMNS)) == EXPECTED_MODEL_ROWS,
        rows=len(model_input),
        unique_keys=model_input.n_unique(subset=list(KEY_COLUMNS)),
    )
    for name in ("timing", "selection"):
        audit = _read_json(MODEL_DIR / f"{name}_model_audit.json")
        feature_sets = list(audit["screening_log"].values())
        verifier.check(
            f"{name}_model_fixed_feature_set",
            audit["feature_manifest"]["hash"] == EXPECTED_FEATURE_HASH
            and audit["feature_names"] == list(FEATURE_MANIFEST.ordered_names)
            and all(
                features == list(FEATURE_MANIFEST.ordered_names)
                for features in feature_sets
            ),
            fitted_sessions=len(feature_sets),
        )


def _verify_thresholds(verifier: Verifier) -> str:
    sessions = sorted(
        path.name[:8]
        for path in data_roots().tick_data_dir.glob("*_StockTick.parquet")
        if "20250211" <= path.name[:8] <= "20260825"
    )
    event_dates = (
        pl.read_parquet(MODEL_INPUT_PATH, columns=["Date"])["Date"]
        .unique()
        .sort()
        .to_list()
    )
    warmup_start = event_dates[TIMING_LOOKBACK_DAYS]
    verifier.check(
        "event_active_warmup_boundary",
        len(sessions) == EXPECTED_EVENT_DATES
        and len(event_dates) == EXPECTED_EVENT_DATES - 1
        and warmup_start == "20250513",
        calendar_sessions=len(sessions),
        event_active_dates=len(event_dates),
        warmup_start=warmup_start,
    )
    for name, path, threshold_col, quantile in (
        ("timing", TIMING_THRESHOLD_PATH, "timing_threshold", 0.30),
        ("selection", SELECTION_THRESHOLD_PATH, "selection_threshold", 0.10),
    ):
        thresholds = pl.read_parquet(path)
        finite = thresholds.filter(pl.col(threshold_col).is_finite())
        valid_history = finite.filter(
            (pl.col("history_end") >= pl.col("Date"))
            | (pl.col("history_sessions") < THRESHOLD_MIN_SESSIONS)
            | (pl.col("history_observations") < THRESHOLD_MIN_OBSERVATIONS)
            | (pl.col("window_basis") != "calendar_sessions")
            | ((pl.col("quantile") - quantile).abs() > 1e-12)
        ).is_empty()
        verifier.check(
            f"{name}_prior_only_thresholds",
            not finite.is_empty() and valid_history,
            threshold_rows=len(thresholds),
            finite_rows=len(finite),
            earliest_date=thresholds["Date"].min(),
            latest_date=thresholds["Date"].max(),
        )

    thresholded = pl.read_parquet(THRESHOLDED_PATH)
    prewarm = thresholded.filter(pl.col("Date") < warmup_start)
    prewarm_passes = prewarm.filter(
        pl.col("model_era").fill_null(False)
        | pl.col("passes_timing").fill_null(False)
        | pl.col("passes_selection").fill_null(False)
        | pl.col("passes_dual").fill_null(False)
    )
    model_era = thresholded.filter(pl.col("model_era"))
    verifier.check(
        "threshold_and_execution_warmup_fail_closed",
        prewarm_passes.is_empty()
        and not model_era.is_empty()
        and model_era["Date"].min() == warmup_start,
        prewarm_pass_rows=len(prewarm_passes),
        model_era_start=model_era["Date"].min(),
    )
    return warmup_start


def _verify_selected_signals(verifier: Verifier, warmup_start: str) -> None:
    selected = pl.read_parquet(SELECTED_SIGNALS_PATH)
    key_count = selected.n_unique(subset=list(KEY_COLUMNS))
    valid_static = (
        pl.col("time_bucket").is_not_null()
        & pl.col("to_ref_bucket").is_not_null()
        & ((pl.col("OTC") == True) | (pl.col("MD_L1Rate_30_re") > 0.25))
        & (pl.col("SpreadPairElapsed") > 0.1)
        & (
            (pl.col("AmountRank_canDayTrade") <= 100)
            | (pl.col("day_amount_rank") <= 100)
        )
        & (pl.col("BidPrice1") <= PRICE_CAP)
        & (
            pl.col("hft_strick_makerSpreadBP").is_null()
            | (pl.col("hft_strick_makerSpreadBP") > -70.0)
        )
    )
    invalid_static = selected.filter(~valid_static.fill_null(False))
    valid_capacity = (
        (pl.col("accLots") < pl.col("avg_bidLots1") + pl.col("avg_askLots1"))
        & (pl.col("stock_position_wan") < PER_STOCK_POSITION_CAP_WAN)
    )
    invalid_capacity = selected.filter(~valid_capacity.fill_null(False))
    valid_gate = (
        (pl.col("Date") >= warmup_start)
        & pl.col("model_era")
        & pl.col("passes_timing")
        & (pl.col("signal_variant") == "timing_only")
        & ((pl.col("TakerSell_CloseBP") - pl.col("label_raw_bp")).abs() <= 1e-12)
    )
    invalid_gate = selected.filter(~valid_gate.fill_null(False))
    verifier.check(
        "selected_signal_gate_and_capacity_contract",
        len(selected) == EXPECTED_SELECTED_ROWS
        and key_count == EXPECTED_SELECTED_ROWS
        and invalid_static.is_empty()
        and invalid_capacity.is_empty()
        and invalid_gate.is_empty(),
        rows=len(selected),
        unique_keys=key_count,
        invalid_static=len(invalid_static),
        invalid_capacity=len(invalid_capacity),
        invalid_gate=len(invalid_gate),
        first_signal_date=selected["Date"].min(),
    )

    decision = _read_json(SELECTION_DECISION_PATH)
    summary = pl.read_parquet(VARIANT_SUMMARY_PATH)
    eligible = summary.filter(
        (pl.col("period") == "development")
        & pl.col("signal_variant").is_in(
            ["timing_only", "selection_only", "dual"]
        )
        & (pl.col("active_days") >= 60)
        & (pl.col("trades") >= 1000)
    ).sort("daily_sharpe", descending=True)
    verifier.check(
        "development_only_gate_selection",
        decision["selected_variant"] == "timing_only"
        and decision["holdout_not_used_for_selection"] is True
        and eligible["signal_variant"][0] == "timing_only",
        ranking=eligible.select("signal_variant", "daily_sharpe").to_dicts(),
    )

    pnl = float(
        selected.select(
            (
                pl.col("BidPrice1")
                * (pl.col("label_raw_bp") - TAKER_FEE_BP)
                / 10.0
            ).sum()
        ).item()
    )
    holdout = selected.filter(pl.col("Date") >= HOLDOUT_START)
    holdout_pnl = float(
        holdout.select(
            (
                pl.col("BidPrice1")
                * (pl.col("label_raw_bp") - TAKER_FEE_BP)
                / 10.0
            ).sum()
        ).item()
    )
    full_reported = float(decision["full_backtest_stats"]["總報酬（萬）"]) * 10_000.0
    holdout_reported = (
        float(decision["holdout_backtest_stats"]["總報酬（萬）"]) * 10_000.0
    )
    verifier.check(
        "exact_backtest_pnl_parity",
        math.isclose(pnl, full_reported, rel_tol=1e-12, abs_tol=1e-6)
        and math.isclose(
            holdout_pnl, holdout_reported, rel_tol=1e-12, abs_tol=1e-6
        ),
        full_formula_pnl=pnl,
        full_reported_pnl=full_reported,
        holdout_formula_pnl=holdout_pnl,
        holdout_reported_pnl=holdout_reported,
    )

    holdout_summary = summary.filter(
        (pl.col("signal_variant") == "timing_only")
        & (pl.col("period") == "holdout_2026_ytd")
    ).row(0, named=True)
    fee25 = pl.read_parquet(FEE_SENSITIVITY_PATH).filter(
        (pl.col("period") == "holdout_2026_ytd")
        & (pl.col("fee_bp") == 25.0)
    ).row(0, named=True)
    verifier.check(
        "holdout_survives_cost_and_residual_checks",
        holdout_summary["cap_weighted_net_bp"] > 0.0
        and holdout_summary["cap_weighted_residual_net_bp"] > 0.0
        and fee25["net_pnl"] > 0.0,
        holdout_net_bp=holdout_summary["cap_weighted_net_bp"],
        holdout_residual_net_bp=holdout_summary[
            "cap_weighted_residual_net_bp"
        ],
        holdout_fee25_net_bp=fee25["cap_weighted_net_bp"],
    )

    images = [
        BACKTEST_DIR / "backtest_full.png",
        BACKTEST_DIR / "backtest_2026_holdout.png",
    ]
    verifier.check(
        "backtest_plots_nonempty",
        all(path.exists() and path.stat().st_size > 100_000 for path in images),
        files={path.name: path.stat().st_size for path in images if path.exists()},
    )


def _verify_audits_and_manifest(verifier: Verifier) -> None:
    sampling = _read_json(SAMPLING_AUDIT_PATH)
    verifier.check(
        "sampling_avoids_repeated_unchanged_price_states",
        sampling["result"] == "pass"
        and sampling["events"] == sampling["fresh_events"]
        and sampling["state_mismatch_events"] == 0
        and sampling["md_price_change_events"] > 0
        and sampling["large_fill_events"] > 0,
        audited_dates=len(sampling["audited_dates"]),
        events=sampling["events"],
        md_price_change_events=sampling["md_price_change_events"],
        large_fill_events=sampling["large_fill_events"],
    )

    overlap = _read_json(OVERLAP_PATH)
    overlap_summary = pl.read_parquet(OVERLAP_SUMMARY_PATH)
    disjoint_holdout = overlap_summary.filter(
        (pl.col("segment") == "stock_day_disjoint")
        & (pl.col("period") == "holdout_2026_ytd")
    ).row(0, named=True)
    verifier.check(
        "negfill_overlap_diagnostic_recorded",
        overlap["selected_events"] == EXPECTED_SELECTED_ROWS
        and overlap["exact_negfill_tick_events"] == 558
        and np.isclose(overlap["exact_negfill_tick_rate"], 0.013431219159947045)
        and overlap["selected_stock_days"] == 6828
        and overlap["negfill_stock_days"] == 6700
        and np.isclose(overlap["negfill_stock_day_rate"], 0.981253661394259)
        and disjoint_holdout["net_pnl"] < 0.0,
        exact_tick_rate=overlap["exact_negfill_tick_rate"],
        stock_day_rate=overlap["negfill_stock_day_rate"],
        stock_day_disjoint_holdout_net_pnl=disjoint_holdout["net_pnl"],
    )

    run_manifest = _read_json(RUN_MANIFEST_PATH)
    mismatches: list[str] = []
    missing: list[str] = []
    for relative, identity in run_manifest["artifacts"].items():
        path = STUDY_DIR / relative
        if not path.exists():
            missing.append(relative)
            continue
        if path.stat().st_size != identity["size"] or _sha256(path) != identity["sha256"]:
            mismatches.append(relative)
    verifier.check(
        "run_manifest_hashes_match",
        not missing and not mismatches,
        artifact_count=len(run_manifest["artifacts"]),
        missing=missing,
        mismatches=mismatches,
    )
    source_mismatches: list[str] = []
    source_missing: list[str] = []
    for relative, identity in run_manifest["source_files"].items():
        path = PROJECT_ROOT / relative
        if not path.exists():
            source_missing.append(relative)
            continue
        if path.stat().st_size != identity["size"] or _sha256(path) != identity["sha256"]:
            source_mismatches.append(relative)
    verifier.check(
        "source_and_environment_lineage_recorded",
        not source_missing
        and not source_mismatches
        and run_manifest.get("raw_source_identity_contract")
        == "path_size_mtime_ns"
        and bool(run_manifest.get("environment", {}).get("python")),
        source_file_count=len(run_manifest["source_files"]),
        missing=source_missing,
        mismatches=source_mismatches,
        environment=run_manifest.get("environment"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=VERIFICATION_PATH, help="JSON audit output"
    )
    args = parser.parse_args()

    verifier = Verifier()
    try:
        _verify_index_and_dataset(verifier)
        _verify_features_and_models(verifier)
        warmup_start = _verify_thresholds(verifier)
        _verify_selected_signals(verifier, warmup_start)
        _verify_audits_and_manifest(verifier)
    except Exception as exc:
        verifier.check(
            "verification_completed_without_exception",
            False,
            exception=f"{type(exc).__name__}: {exc}",
        )

    payload = {
        "contract": "filllots_at_high_signal_verification/v1",
        "result": "pass" if verifier.passed else "fail",
        "checks": verifier.checks,
    }
    _atomic_json(payload, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if verifier.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
