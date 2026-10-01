"""Extend the static fh_neg_rejection snapshot with post-freeze sessions.

This runner does not touch any frozen artifact.  It rebuilds the identical
selector -> feature -> label -> static-gate -> capacity chain for sessions
after the frozen END (20260825) and writes everything into a versioned
study-local output directory.  The first extension day is the
frozen END itself, used purely as a pipeline-parity check against the frozen
``accepted_signals.parquet`` and then excluded from the extension P&L.

Known approximation: the frozen cohort additionally passed the factorial
lineage fail-closed audit, which removed 0.28% of model-input rows over the
whole retrospective window.  The extension is selector-only and therefore may
keep a handful of rows per day that the audit would have quarantined.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[5]
STUDY_DIR = Path(__file__).resolve().parent
FROZEN_DIR = STUDY_DIR / "artifacts" / "static_backtest_retrospective_v1"
OUTPUT_DIR = STUDY_DIR / "artifacts" / "static_backtest_extension_v1"
LEGACY_DIR = PROJECT_ROOT / "src/research/intradaySignal/experiments/20260826_filllots_at_high_signal"

FROZEN_END = "20260825"
PRIMARY_FEE_BP = 19.3
KEY_COLUMNS = ("Date", "QuoteCode", "ChannelSeq")

# Snapshot of run_model_backtest.BACKTEST_COLUMNS (that module is not
# importable without chdir into the legacy study).
BACKTEST_COLUMNS = (
    "Date",
    "QuoteCode",
    "ChannelSeq",
    "TransTime",
    "BidPrice1",
    "Close",
    "FutureHigh",
    "RefPrice",
    "TickSize",
    "label_raw_bp",
    "label_residual_bp",
    "label_z",
    "label_z_within_stock_day",
    "label_z_within_day",
    "OTC",
    "MD_L1Rate_30_re",
    "SpreadPairElapsed",
    "RemainSeconds",
    "ToRef",
    "AmountRank_canDayTrade",
    "day_amount_rank",
    "avg_bidLots1",
    "avg_askLots1",
    "hft_strick_makerSpreadBP",
)

PLOT_COLUMNS = (
    "Date",
    "QuoteCode",
    "ChannelSeq",
    "TransTime",
    "BidPrice1",
    "TakerSell_CloseBP",
    "FutureHigh",
    "RefPrice",
    "label_raw_bp",
    "label_residual_bp",
)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _evidence_class(date: str) -> str:
    if date <= "20260825":
        return "retrospective_diagnostic"
    if date == "20260826":
        return "artifact_oos_non_pristine"
    if date == "20260827":
        return "freeze_day_excluded"
    return "prospective_first_live"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", required=True, help="last session YYYYMMDD")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="new study-local artifact directory; existing runs are preserved",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    artifact_root = (STUDY_DIR / "artifacts").resolve()
    if not output_dir.is_relative_to(artifact_root):
        raise ValueError("output-dir must be inside this study's artifacts")
    if output_dir == FROZEN_DIR.resolve():
        raise ValueError("cannot write into the frozen static BackTest directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"{output_dir} already contains artifacts; choose a new version"
        )

    legacy = _load_module("_fh_signal_study_20260826_ext", LEGACY_DIR / "study.py")
    rsb = _load_module(
        "_fh_static_backtest_ext", STUDY_DIR / "run_static_backtest.py"
    )
    from src.research.intradaySignal.framework.dataset import build_event_dataset
    from src.research.intradaySignal.framework.events import build_event_index
    from src.research.intradaySignal.framework.labels import (
        add_centered_labels,
        add_directional_labels,
    )
    from src.research.intradaySignal.framework.profiles.negfill import (
        SHORT_TAKER_STOP108_V1,
    )

    roots = legacy.data_roots()
    from src.research.intradaySignal.framework.events import EventDaySources

    source_calendar = sorted(
        path.name[:8]
        for path in roots.tick_data_dir.glob("*_StockTick.parquet")
        if FROZEN_END <= path.name[:8] <= args.end
    )
    if not source_calendar or source_calendar[-1] != args.end:
        raise FileNotFoundError(
            f"requested end {args.end} is not a complete tick-data session"
        )
    for date in source_calendar:
        EventDaySources(date, roots).require(
            "tick_data", "tick_feature", "pre_market", "tick_bar", "txf_tick"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Event index + causal dataset for FROZEN_END..end -------------
    index_spec = dataclasses.replace(
        legacy.event_index_spec(),
        index_path=output_dir / "event_index" / "fh_neg_rejection.parquet",
    )
    report = build_event_index(
        index_spec,
        roots=roots,
        force=args.force,
        start=FROZEN_END,
        end=args.end,
    )
    print(
        f"[index] rows={report.row_count} dates={report.event_date_count} "
        f"scanned={len(report.scanned_dates)}",
        flush=True,
    )
    if list(report.scanned_dates) != source_calendar:
        raise RuntimeError("event-index scanned calendar differs from complete sources")
    dataset_spec = dataclasses.replace(
        legacy.event_dataset_spec(),
        index_path=index_spec.index_path,
        output_dir=output_dir / "event_dataset",
    )
    dataset_report = build_event_dataset(
        dataset_spec,
        roots=roots,
        force=args.force,
        start=FROZEN_END,
        end=args.end,
    )
    print(
        f"[dataset] saved={len(dataset_report.saved_dates)} "
        f"skipped={len(dataset_report.skipped_dates)}",
        flush=True,
    )

    # ---- 2. Model-input reconstruction (same as build_model_input) -------
    partitions = sorted(
        path
        for path in (output_dir / "event_dataset").glob("*.parquet")
        if FROZEN_END <= path.stem <= args.end
    )
    if not partitions:
        raise FileNotFoundError("no extension event-dataset partitions")
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
            (*BACKTEST_COLUMNS, *legacy.FEATURE_MANIFEST.ordered_names)
        )
    )
    model_input = labeled.select(keep).sort(
        ["Date", "QuoteCode", "TransTime", "ChannelSeq"]
    )
    if model_input.select(
        pl.struct(list(KEY_COLUMNS)).is_duplicated().any()
    ).item():
        raise ValueError("extension model input contains duplicate keys")

    # ---- 3. Robustness pool boundary + legal-tick stop108 label ----------
    cohort = rsb.apply_robustness_pool_boundary(model_input)
    cohort = rsb.recompute_robustness_labels(cohort)

    # ---- 4. Static gates + capacity (frozen execution contract) ----------
    bucketed = legacy.add_threshold_buckets(cohort)
    pre_cap = legacy.static_execution_filter(bucketed).sort(
        ["Date", "QuoteCode", "TransTime", "ChannelSeq"]
    )
    accepted = (
        legacy.apply_capacity(pre_cap)
        .with_columns(
            pl.lit("static_anchor_no_model_gate_ext").alias("signal_variant"),
            pl.col("label_raw_bp").alias("TakerSell_CloseBP"),
            pl.lit(1, dtype=pl.Int8).alias("lots"),
        )
        .sort(["Date", "QuoteCode", "TransTime", "ChannelSeq"])
    )

    # ---- 5. Parity check on FROZEN_END against the frozen snapshot -------
    frozen_signals = pl.read_parquet(FROZEN_DIR / "accepted_signals.parquet")
    frozen_day = frozen_signals.filter(pl.col("Date") == FROZEN_END)
    ext_day = accepted.filter(pl.col("Date") == FROZEN_END)
    joined = ext_day.select(KEY_COLUMNS).join(
        frozen_day.select(KEY_COLUMNS), on=list(KEY_COLUMNS), how="inner"
    )
    parity = {
        "parity_date": FROZEN_END,
        "frozen_rows": len(frozen_day),
        "extension_rows": len(ext_day),
        "matched_keys": len(joined),
        "frozen_net_pnl": float(
            (
                frozen_day["BidPrice1"]
                * (frozen_day["label_raw_bp"] - PRIMARY_FEE_BP)
                / 10.0
            ).sum()
        ),
        "extension_net_pnl": float(
            (
                ext_day["BidPrice1"]
                * (ext_day["label_raw_bp"] - PRIMARY_FEE_BP)
                / 10.0
            ).sum()
        ),
    }
    print(f"[parity] {json.dumps(parity)}", flush=True)
    if (
        len(frozen_day) == 0
        or parity["extension_rows"] != parity["frozen_rows"]
        or parity["matched_keys"] != parity["frozen_rows"]
        or abs(parity["extension_net_pnl"] - parity["frozen_net_pnl"]) > 1e-6
    ):
        raise RuntimeError("frozen-end pipeline parity failed")

    # ---- 6. Extension daily P&L (parity day excluded) --------------------
    ext_signals = accepted.filter(pl.col("Date") > FROZEN_END)
    ext_calendar = [date for date in source_calendar if date > FROZEN_END]
    ext_daily = rsb._daily_frame(ext_signals, ext_calendar, PRIMARY_FEE_BP)
    ext_daily = ext_daily.with_columns(
        pl.col("Date")
        .map_elements(_evidence_class, return_dtype=pl.String)
        .alias("evidence_class")
    )

    frozen_daily = pl.read_parquet(FROZEN_DIR / "daily.parquet").with_columns(
        pl.lit("retrospective_diagnostic").alias("evidence_class")
    )
    combined = pl.concat(
        [frozen_daily, ext_daily.select(frozen_daily.columns)],
        how="vertical",
    ).sort("Date")
    cumulative = np.cumsum(combined["net_pnl"].to_numpy())
    high_water = np.maximum.accumulate(cumulative)
    combined = combined.with_columns(
        pl.Series("cumulative_net_pnl", cumulative),
        pl.Series("high_water_mark", high_water),
        pl.Series("drawdown", cumulative - high_water),
    )

    ext_signals.write_parquet(output_dir / "accepted_signals_extension.parquet")
    ext_daily.write_parquet(output_dir / "daily_extension.parquet")
    combined.write_parquet(output_dir / "daily_combined.parquet")

    # ---- 7. Project BackTest plots over the combined signal set ----------
    combined_signals = pl.concat(
        [
            frozen_signals.select(PLOT_COLUMNS),
            ext_signals.select(PLOT_COLUMNS),
        ],
        how="vertical",
    )
    rsb._save_project_backtest_plot(
        combined_signals,
        path=output_dir / "backtest_full_extended.png",
        title=(
            "fh_neg_rejection static anchor (no model gate) "
            f"20250211-{args.end} · extension after {FROZEN_END}"
        ),
        fee_bp=PRIMARY_FEE_BP,
    )
    rsb._save_project_backtest_plot(
        combined_signals.filter(pl.col("Date") >= "20260101"),
        path=output_dir / "backtest_2026_extended.png",
        title=(
            "fh_neg_rejection static anchor (no model gate) "
            f"2026 YTD through {args.end}"
        ),
        fee_bp=PRIMARY_FEE_BP,
    )

    manifest = {
        "contract": "fh_neg_rejection_static_backtest_extension/v1",
        "note": (
            "Selector-only extension of the frozen retrospective snapshot; "
            "no factorial lineage audit; RESEARCH_SHADOW_ONLY; not "
            "deployment evidence."
        ),
        "frozen_end": FROZEN_END,
        "requested_end": args.end,
        "output_dir": str(output_dir),
        "source_calendar": source_calendar,
        "extension_calendar": ext_calendar,
        "evidence_classes": {
            date: _evidence_class(date) for date in ext_calendar
        },
        "extension_accepted_rows": len(ext_signals),
        "parity": parity,
        "fee_bp": PRIMARY_FEE_BP,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        ext_daily.select(
            "Date",
            "evidence_class",
            "trades",
            "stocks",
            "gross_pnl",
            "net_pnl",
        )
    )
    total = float(ext_daily["net_pnl"].sum())
    print(f"[extension] net_pnl_total={total:,.0f} TWD", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
