"""Ablate three inherited fh_neg_rejection execution gates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl

from src.research.timeSeries.framework.labels import (
    add_centered_labels,
    add_directional_labels,
)
from src.research.timeSeries.framework.profiles.negfill import (
    SHORT_TAKER_STOP108_V1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[5]
STUDY_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = STUDY_DIR / "artifacts" / "gate_factorial_20261013_v1"
SOURCE_STUDY_DIR = (
    PROJECT_ROOT
    / "src/research/timeSeries/test/20260827_fh_neg_rejection_robustness"
)
LEGACY_STUDY_DIR = (
    PROJECT_ROOT
    / "src/research/timeSeries/test/20260826_filllots_at_high_signal"
)
EXTENSION_DIR = (
    SOURCE_STUDY_DIR
    / "artifacts/static_backtest_extension_20261013_v1"
)
FROZEN_DIR = SOURCE_STUDY_DIR / "artifacts/static_backtest_retrospective_v1"
FEE_BP = 19.3
KEY_COLUMNS = ("Date", "QuoteCode", "ChannelSeq")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.write_parquet(temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _key_sha256(frame: pl.DataFrame) -> str:
    keys = frame.select(KEY_COLUMNS).sort(KEY_COLUMNS)
    digest = hashlib.sha256()
    for row in keys.iter_rows():
        digest.update("\x1f".join(str(value) for value in row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _extension_cohort(
    legacy: ModuleType,
    static_runner: ModuleType,
    extension_runner: ModuleType,
) -> pl.DataFrame:
    partitions = sorted(
        path
        for path in (EXTENSION_DIR / "event_dataset").glob("*.parquet")
        if path.stem > static_runner.END
    )
    if not partitions or partitions[-1].stem != "20261013":
        raise FileNotFoundError("complete extension partitions through 20261013 required")
    raw = pl.concat(
        [
            pl.read_parquet(
                path, columns=list(legacy.REQUIRED_MODEL_INPUT_COLUMNS)
            )
            for path in partitions
        ],
        how="vertical_relaxed",
    )
    featured = legacy.add_study_features(raw)
    labeled = add_directional_labels(featured, spec=SHORT_TAKER_STOP108_V1)
    labeled = labeled.filter(
        pl.col("label_valid")
        & pl.col("label_raw_bp").is_finite()
        & (pl.col("label_raw_bp").abs() < 10_000.0)
    )
    labeled = add_centered_labels(labeled)
    keep = tuple(
        dict.fromkeys(
            (
                *extension_runner.BACKTEST_COLUMNS,
                *legacy.FEATURE_MANIFEST.ordered_names,
            )
        )
    )
    model_input = labeled.select(keep).sort(
        ["Date", "QuoteCode", "TransTime", "ChannelSeq"]
    )
    bounded = static_runner.apply_robustness_pool_boundary(model_input)
    return static_runner.recompute_robustness_labels(bounded)


def _apply_variant(
    bucketed: pl.DataFrame,
    *,
    legacy: ModuleType,
    use_toref: bool,
    use_l1: bool,
    use_spread: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    required = (
        pl.col("time_bucket").is_not_null()
        & (
            (pl.col("AmountRank_canDayTrade") <= 100)
            | (pl.col("day_amount_rank") <= 100)
        )
        & (pl.col("BidPrice1") <= 1000.0)
        & (
            pl.col("hft_strick_makerSpreadBP").is_null()
            | (pl.col("hft_strick_makerSpreadBP") > -70.0)
        )
    )
    if use_toref:
        required &= pl.col("to_ref_bucket").is_not_null()
    if use_l1:
        required &= (pl.col("OTC") == True) | (
            pl.col("MD_L1Rate_30_re") > 0.25
        )
    if use_spread:
        required &= pl.col("SpreadPairElapsed") > 0.1
    pre_cap = bucketed.filter(required).sort(
        ["Date", "QuoteCode", "TransTime", "ChannelSeq"]
    )
    accepted = legacy.apply_capacity(pre_cap).sort(
        ["Date", "QuoteCode", "TransTime", "ChannelSeq"]
    )
    if accepted.select(pl.struct(KEY_COLUMNS).is_duplicated().any()).item():
        raise ValueError("accepted variant contains duplicate event keys")
    return pre_cap, accepted


def _calendar_between(calendar: list[str], start: str, end: str) -> list[str]:
    return [date for date in calendar if start <= date <= end]


def run() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"refusing to overwrite {OUTPUT_DIR}")
    legacy = _load_module(
        "_fh_gate_ablation_legacy", LEGACY_STUDY_DIR / "study.py"
    )
    static_runner = _load_module(
        "_fh_gate_ablation_static", SOURCE_STUDY_DIR / "run_static_backtest.py"
    )
    extension_runner = _load_module(
        "_fh_gate_ablation_extension",
        SOURCE_STUDY_DIR / "extend_static_backtest.py",
    )

    frozen_calendar = static_runner._read_calendar()
    frozen_cohort, _, frozen_counts, _ = static_runner.build_anchor_cohort(
        frozen_calendar
    )
    extension_cohort = _extension_cohort(
        legacy, static_runner, extension_runner
    )
    cohort = pl.concat(
        [frozen_cohort, extension_cohort.select(frozen_cohort.columns)],
        how="vertical_relaxed",
    )
    if cohort.select(pl.struct(KEY_COLUMNS).is_duplicated().any()).item():
        raise ValueError("combined cohort contains duplicate event keys")
    bucketed = legacy.add_threshold_buckets(cohort)
    extension_calendar = sorted(extension_cohort["Date"].unique().to_list())
    calendar = [*frozen_calendar, *extension_calendar]
    if len(calendar) != len(set(calendar)) or calendar[-1] != "20261013":
        raise ValueError("invalid combined calendar")

    periods = {
        "full": calendar,
        "year_2025": _calendar_between(calendar, "20250211", "20251231"),
        "pre_freeze_2026": _calendar_between(
            calendar, "20260102", "20260825"
        ),
        "extension_all": _calendar_between(
            calendar, "20260826", "20261013"
        ),
        "first_live": _calendar_between(calendar, "20260828", "20261013"),
        "year_2026_ytd": _calendar_between(
            calendar, "20260102", "20261013"
        ),
    }
    trial_rows: list[dict[str, Any]] = []
    daily_frames: list[pl.DataFrame] = []
    funnel_rows: list[dict[str, Any]] = []
    ledgers: list[pl.DataFrame] = []
    baseline: pl.DataFrame | None = None

    for use_toref, use_l1, use_spread in product((False, True), repeat=3):
        variant = f"T{int(use_toref)}L{int(use_l1)}S{int(use_spread)}"
        pre_cap, accepted = _apply_variant(
            bucketed,
            legacy=legacy,
            use_toref=use_toref,
            use_l1=use_l1,
            use_spread=use_spread,
        )
        accepted = accepted.with_columns(pl.lit(variant).alias("variant"))
        if variant == "T1L1S1":
            baseline = accepted
        ledgers.append(
            accepted.select(
                "variant",
                *KEY_COLUMNS,
                "TransTime",
                "BidPrice1",
                "label_raw_bp",
                "label_residual_bp",
                "ToRef",
                "MD_L1Rate_30_re",
                "SpreadPairElapsed",
                "accLots",
                "stock_position_wan",
            )
        )
        for period, dates in periods.items():
            summary = static_runner._period_summary(
                accepted,
                period=period,
                calendar=dates,
                fee_bp=FEE_BP,
            )
            summary.update(
                {
                    "variant": variant,
                    "use_toref_gate": use_toref,
                    "use_l1_gate": use_l1,
                    "use_spread_gate": use_spread,
                }
            )
            trial_rows.append(summary)
            daily_frames.append(
                static_runner._daily_frame(accepted, dates, FEE_BP)
                .with_columns(
                    pl.lit(variant).alias("variant"),
                    pl.lit(period).alias("period"),
                )
            )
        for period, dates in {
            "full": periods["full"],
            "first_live": periods["first_live"],
            "date_20261013": ["20261013"],
        }.items():
            funnel_rows.append(
                {
                    "variant": variant,
                    "period": period,
                    "condition_rows": cohort.filter(
                        pl.col("Date").is_in(dates)
                    ).height,
                    "pre_capacity_rows": pre_cap.filter(
                        pl.col("Date").is_in(dates)
                    ).height,
                    "accepted_rows": accepted.filter(
                        pl.col("Date").is_in(dates)
                    ).height,
                }
            )

    if baseline is None:
        raise RuntimeError("baseline factorial cell missing")
    reference = pl.concat(
        [
            pl.read_parquet(FROZEN_DIR / "accepted_signals.parquet").select(
                KEY_COLUMNS
            ),
            pl.read_parquet(
                EXTENSION_DIR / "accepted_signals_extension.parquet"
            ).select(KEY_COLUMNS),
        ],
        how="vertical",
    )
    baseline_keys = baseline.select(KEY_COLUMNS)
    missing = reference.join(
        baseline_keys, on=list(KEY_COLUMNS), how="anti"
    ).height
    unexpected = baseline_keys.join(
        reference, on=list(KEY_COLUMNS), how="anti"
    ).height
    if missing or unexpected:
        raise RuntimeError(
            f"baseline parity failed: missing={missing}, unexpected={unexpected}"
        )

    summary_frame = pl.DataFrame(trial_rows).drop("signal_variant").sort(
        ["period", "variant"]
    )
    daily_frame = pl.concat(daily_frames, how="diagonal_relaxed").sort(
        ["period", "variant", "Date"]
    )
    funnel_frame = pl.DataFrame(funnel_rows).sort(["period", "variant"])
    ledger = pl.concat(ledgers, how="vertical_relaxed").sort(
        ["variant", "Date", "QuoteCode", "TransTime", "ChannelSeq"]
    )
    _atomic_parquet(summary_frame, OUTPUT_DIR / "period_summary.parquet")
    _atomic_parquet(daily_frame, OUTPUT_DIR / "daily_summary.parquet")
    _atomic_parquet(funnel_frame, OUTPUT_DIR / "funnel_summary.parquet")
    _atomic_parquet(ledger, OUTPUT_DIR / "accepted_signals.parquet")
    _atomic_json(
        {
            "contract": "fh_neg_rejection_execution_gate_factorial/v1",
            "status": "RESEARCH_DIAGNOSTIC_ONLY",
            "raw_trials": 8,
            "selection_mode": "complete_factorial_no_promotion",
            "fee_bp": FEE_BP,
            "calendar_start": calendar[0],
            "calendar_end": calendar[-1],
            "calendar_sessions": len(calendar),
            "frozen_cohort_rows": frozen_counts["labeled_anchor_rows"],
            "extension_cohort_rows": extension_cohort.height,
            "combined_cohort_rows": cohort.height,
            "baseline_parity": {
                "expected_rows": reference.height,
                "actual_rows": baseline.height,
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "key_sha256": _key_sha256(baseline),
            },
            "artifacts": {
                "period_summary": "period_summary.parquet",
                "daily_summary": "daily_summary.parquet",
                "funnel_summary": "funnel_summary.parquet",
                "accepted_signals": "accepted_signals.parquet",
            },
            "note": (
                "Outcome-informed execution-gate sensitivity only; no production "
                "change or winner promotion is authorized."
            ),
        },
        OUTPUT_DIR / "run_manifest.json",
    )
    print(
        summary_frame.filter(
            pl.col("period").is_in(
                ["year_2025", "pre_freeze_2026", "first_live"]
            )
        ).select(
            "period",
            "variant",
            "trades",
            "cap_weighted_net_bp",
            "net_pnl",
            "calendar_daily_sharpe",
            "max_drawdown",
        )
    )
    print("baseline parity", reference.height, baseline.height, missing, unexpected)


if __name__ == "__main__":
    run()
