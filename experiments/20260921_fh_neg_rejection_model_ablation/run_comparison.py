"""Compare all-condition, Ridge, and ElasticNet on the frozen FH signal study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from src.research.intradaySignal.framework.model import train_and_predict_causal_v2
from src.research.intradaySignal.framework.prior_thresholds import (
    PriorQuantileSpec,
    WindowBasis,
    apply_prior_threshold,
    build_prior_session_thresholds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[5]
STUDY_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = STUDY_DIR / "artifacts" / "model_family_v1"
SOURCE_DIR = (
    PROJECT_ROOT
    / "src/research/intradaySignal/experiments/20260826_filllots_at_high_signal"
)
KEY_COLUMNS = ("Date", "QuoteCode", "ChannelSeq")
MODEL_ID = "elastic_net_l1_050"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_source_modules() -> tuple[ModuleType, ModuleType]:
    study = _load_module("study", SOURCE_DIR / "study.py")
    runner = _load_module("_fh_source_runner_20260921", SOURCE_DIR / "run_model_backtest.py")
    return study, runner


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _model_audit(result: dict[str, object]) -> dict[str, object]:
    return {
        "feature_manifest": result["feature_manifest"],
        "feature_names": result["feature_names"],
        "screening_log": result["screening_log"],
        "coeff_log": result["coeff_log"],
        "pls_log": result["pls_log"],
    }


def train_elastic_net(
    model_input: pl.DataFrame,
    *,
    study: ModuleType,
    runner: ModuleType,
    n_jobs: int,
) -> pl.DataFrame:
    columns = tuple(
        dict.fromkeys((*runner.BACKTEST_COLUMNS, *study.FEATURE_MANIFEST.ordered_names))
    )
    frame = model_input.select(columns).to_pandas()
    frame["Date"] = pd.to_datetime(frame["Date"], format="%Y%m%d")
    common = {
        "n_jobs": n_jobs,
        "feature_screening": False,
        "max_position_per_stock": study.TRAINING_POSITION_CAP_WAN,
        "position_sample_mode": "hard",
        "model_family": "elastic_net",
        "elastic_net_l1_ratio": 0.5,
        "elastic_net_alpha_mode": "ridge_l2_equivalent",
        "linear_model_max_iter": 50_000,
        "linear_model_tol": 1e-4,
    }
    timing = train_and_predict_causal_v2(
        frame,
        study.FEATURE_MANIFEST,
        study.TIMING_TARGET,
        n_training_days=study.TIMING_LOOKBACK_DAYS,
        alpha=study.RIDGE_ALPHA,
        **common,
    )
    _atomic_json(
        _jsonable(_model_audit(timing)), OUTPUT_DIR / "timing_model_audit.json"
    )
    selection = train_and_predict_causal_v2(
        timing["df"],
        study.FEATURE_MANIFEST,
        study.SELECTION_TARGET,
        n_training_days=study.SELECTION_LOOKBACK_DAYS,
        alpha=study.RIDGE_ALPHA,
        **common,
    )
    _atomic_json(
        _jsonable(_model_audit(selection)),
        OUTPUT_DIR / "selection_model_audit.json",
    )
    timing_pred = f"{study.TIMING_TARGET}_pred"
    selection_pred = f"{study.SELECTION_TARGET}_pred"
    output_columns = tuple(
        dict.fromkeys((*runner.BACKTEST_COLUMNS, timing_pred, selection_pred))
    )
    output = pl.from_pandas(selection["df"][list(output_columns)]).with_columns(
        pl.col("Date").dt.strftime("%Y%m%d")
    )
    output = output.sort(["Date", "QuoteCode", "TransTime", "ChannelSeq"])
    if output.select(pl.struct(KEY_COLUMNS).is_duplicated().any()).item():
        raise ValueError("ElasticNet predictions contain duplicate event keys")
    _atomic_parquet(output, OUTPUT_DIR / "predictions.parquet")
    return output


def apply_thresholds(
    predictions: pl.DataFrame,
    *,
    study: ModuleType,
) -> pl.DataFrame:
    timing_pred = f"{study.TIMING_TARGET}_pred"
    selection_pred = f"{study.SELECTION_TARGET}_pred"
    bucketed = study.add_threshold_buckets(predictions)
    sessions = sorted(
        path.name[:8]
        for path in study.data_roots().tick_data_dir.glob("*_StockTick.parquet")
        if study.STUDY_START <= path.name[:8] <= study.STUDY_END
    )
    event_dates = bucketed["Date"].unique().sort().to_list()
    model_warmup_start = event_dates[study.TIMING_LOOKBACK_DAYS]
    scores = bucketed.filter(pl.col("Date") >= model_warmup_start)
    bucket_cols = ("time_bucket", "to_ref_bucket")
    common = {
        "bucket_cols": bucket_cols,
        "lookback_sessions": study.THRESHOLD_LOOKBACK_SESSIONS,
        "min_sessions": study.THRESHOLD_MIN_SESSIONS,
        "min_observations": study.THRESHOLD_MIN_OBSERVATIONS,
        "window_basis": WindowBasis.CALENDAR_SESSIONS,
        "score_unit": "residual_z_centered_model_score",
    }
    timing_spec = PriorQuantileSpec(
        score_col=timing_pred,
        threshold_col="timing_threshold",
        q=study.TIMING_QUANTILE,
        **common,
    )
    selection_spec = PriorQuantileSpec(
        score_col=selection_pred,
        threshold_col="selection_threshold",
        q=study.SELECTION_QUANTILE,
        **common,
    )
    timing_thresholds = build_prior_session_thresholds(
        scores, sessions=sessions, spec=timing_spec
    )
    selection_thresholds = build_prior_session_thresholds(
        scores, sessions=sessions, spec=selection_spec
    )
    _atomic_parquet(timing_thresholds, OUTPUT_DIR / "timing_thresholds.parquet")
    _atomic_parquet(selection_thresholds, OUTPUT_DIR / "selection_thresholds.parquet")
    output = apply_prior_threshold(
        bucketed,
        timing_thresholds.select("Date", *bucket_cols, "timing_threshold"),
        score_col=timing_pred,
        threshold_col="timing_threshold",
        bucket_cols=bucket_cols,
        output_col="passes_timing",
    )
    output = apply_prior_threshold(
        output,
        selection_thresholds.select("Date", *bucket_cols, "selection_threshold"),
        score_col=selection_pred,
        threshold_col="selection_threshold",
        bucket_cols=bucket_cols,
        output_col="passes_selection",
    ).with_columns(
        (
            pl.col(timing_pred).is_finite()
            & pl.col(selection_pred).is_finite()
            & (pl.col("Date") >= model_warmup_start)
        ).alias("model_era"),
        (pl.col("passes_timing") & pl.col("passes_selection")).alias(
            "passes_dual"
        ),
    )
    _atomic_parquet(output, OUTPUT_DIR / "thresholded_events.parquet")
    return output


def build_signals(
    thresholded: pl.DataFrame, *, study: ModuleType
) -> dict[str, pl.DataFrame]:
    gates = {
        "static_only": pl.col("model_era"),
        "timing_only": pl.col("model_era") & pl.col("passes_timing"),
        "selection_only": pl.col("model_era") & pl.col("passes_selection"),
        "dual": pl.col("model_era") & pl.col("passes_dual"),
    }
    output: dict[str, pl.DataFrame] = {}
    for name, gate in gates.items():
        pre_cap = study.static_execution_filter(
            thresholded.filter(gate.fill_null(False))
        )
        output[name] = study.apply_capacity(pre_cap).with_columns(
            pl.lit(name).alias("signal_variant"),
            pl.col("label_raw_bp").alias("TakerSell_CloseBP"),
        )
    return output


def summarize(
    signals: dict[str, pl.DataFrame],
    *,
    study: ModuleType,
    runner: ModuleType,
) -> tuple[pl.DataFrame, dict[str, object]]:
    sessions = sorted(
        path.name[:8]
        for path in study.data_roots().tick_data_dir.glob("*_StockTick.parquet")
        if study.STUDY_START <= path.name[:8] <= study.STUDY_END
    )
    model_dates = signals["static_only"]["Date"].unique().sort().to_list()
    model_start = model_dates[0]
    rows: list[dict[str, object]] = []
    for variant, signal in signals.items():
        for period in ("development", "holdout_2026_ytd", "full_model_era"):
            row = runner._summary_row(
                signal,
                variant=variant,
                period=period,
                calendar=runner._period_calendar(sessions, period, model_start),
                fee_bp=study.TAKER_FEE_BP,
            )
            row["model_family"] = MODEL_ID if variant != "static_only" else "none"
            rows.append(row)
    summary = pl.DataFrame(rows)
    decision = runner._select_variant(summary, model_start=model_start)
    decision["model_family"] = MODEL_ID
    _atomic_parquet(summary, OUTPUT_DIR / "elastic_net_variant_summary.parquet")
    _atomic_json(_jsonable(decision), OUTPUT_DIR / "elastic_net_decision.json")
    selected = signals[str(decision["selected_variant"])]
    _atomic_parquet(selected, OUTPUT_DIR / "elastic_net_selected_signals.parquet")
    _atomic_parquet(signals["static_only"], OUTPUT_DIR / "static_only_signals.parquet")
    return summary, decision


def _daily_net(
    signals: pl.DataFrame, sessions: list[str], fee_bp: float, name: str
) -> pl.DataFrame:
    aggregated = (
        signals.filter(pl.col("Date").is_in(sessions))
        .with_columns(
            (
                pl.col("BidPrice1")
                * (pl.col("label_raw_bp") - fee_bp)
                / 10.0
            ).alias(name)
        )
        .group_by("Date")
        .agg(pl.col(name).sum())
    )
    return (
        pl.DataFrame({"Date": sessions})
        .join(aggregated, on="Date", how="left")
        .with_columns(pl.col(name).fill_null(0.0))
    )


def compare_and_plot(
    *,
    elastic_summary: pl.DataFrame,
    elastic_decision: dict[str, object],
    elastic_signals: dict[str, pl.DataFrame],
    study: ModuleType,
    runner: ModuleType,
) -> dict[str, object]:
    ridge_summary = pl.read_parquet(runner.VARIANT_SUMMARY_PATH).with_columns(
        pl.when(pl.col("signal_variant") == "static_only")
        .then(pl.lit("none"))
        .otherwise(pl.lit("ridge_original"))
        .alias("model_family")
    )
    combined_summary = pl.concat(
        [
            ridge_summary,
            elastic_summary.filter(pl.col("signal_variant") != "static_only").select(
                ridge_summary.columns
            ),
        ],
        how="vertical_relaxed",
    )
    _atomic_parquet(combined_summary, OUTPUT_DIR / "model_family_summary.parquet")

    frozen_static = ridge_summary.filter(
        (pl.col("signal_variant") == "static_only")
        & (pl.col("period") == "holdout_2026_ytd")
    ).row(0, named=True)
    rebuilt_static = elastic_summary.filter(
        (pl.col("signal_variant") == "static_only")
        & (pl.col("period") == "holdout_2026_ytd")
    ).row(0, named=True)
    for column in ("trades", "active_days"):
        if frozen_static[column] != rebuilt_static[column]:
            raise RuntimeError(f"static parity failed for {column}")
    for column in ("net_pnl", "cap_weighted_net_bp", "daily_sharpe"):
        if abs(float(frozen_static[column]) - float(rebuilt_static[column])) >= 1e-8:
            raise RuntimeError(f"static parity failed for {column}")

    ridge_decision = json.loads(runner.SELECTION_DECISION_PATH.read_text())
    ridge_variant = str(ridge_decision["selected_variant"])
    elastic_variant = str(elastic_decision["selected_variant"])
    ridge_selected = pl.read_parquet(runner.SELECTED_SIGNALS_PATH)
    elastic_selected = elastic_signals[elastic_variant]
    sessions = sorted(
        path.name[:8]
        for path in study.data_roots().tick_data_dir.glob("*_StockTick.parquet")
        if study.HOLDOUT_START <= path.name[:8] <= study.STUDY_END
    )
    daily = _daily_net(
        elastic_signals["static_only"], sessions, study.TAKER_FEE_BP, "static_only"
    )
    daily = daily.join(
        _daily_net(ridge_selected, sessions, study.TAKER_FEE_BP, "ridge_selected"),
        on="Date",
    ).join(
        _daily_net(
            elastic_selected, sessions, study.TAKER_FEE_BP, "elastic_net_selected"
        ),
        on="Date",
    )
    _atomic_parquet(daily, OUTPUT_DIR / "holdout_daily_comparison.parquet")

    fig, axis = plt.subplots(figsize=(12, 6))
    for column, label in (
        ("static_only", "All condition / no model gate"),
        ("ridge_selected", f"Ridge selected: {ridge_variant}"),
        ("elastic_net_selected", f"ElasticNet selected: {elastic_variant}"),
    ):
        axis.plot(
            pd.to_datetime(daily["Date"].to_list()),
            np.cumsum(daily[column].to_numpy()) / 10_000.0,
            label=label,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("fh_neg_rejection 2026 evaluation: cumulative net PnL")
    axis.set_ylabel("TWD 10k; fixed 19.3bp cost")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "holdout_cumulative_comparison.png", dpi=160)
    plt.close(fig)

    ridge_predictions = pl.read_parquet(runner.PREDICTION_PATH).select(
        *KEY_COLUMNS,
        f"{study.TIMING_TARGET}_pred",
        f"{study.SELECTION_TARGET}_pred",
    )
    elastic_predictions = pl.read_parquet(OUTPUT_DIR / "predictions.parquet").select(
        *KEY_COLUMNS,
        f"{study.TIMING_TARGET}_pred",
        f"{study.SELECTION_TARGET}_pred",
    )
    joined = ridge_predictions.join(
        elastic_predictions,
        on=list(KEY_COLUMNS),
        suffix="_elastic",
        validate="1:1",
    ).filter(pl.col("Date") >= study.HOLDOUT_START)
    score_correlations: dict[str, dict[str, float]] = {}
    for target in (study.TIMING_TARGET, study.SELECTION_TARGET):
        left = f"{target}_pred"
        right = f"{target}_pred_elastic"
        pair = joined.select(left, right).drop_nulls().to_pandas()
        score_correlations[target] = {
            "rows": len(pair),
            "pearson": float(pair[left].corr(pair[right], method="pearson")),
            "spearman": float(pair[left].corr(pair[right], method="spearman")),
        }

    ridge_keys = ridge_selected.filter(pl.col("Date") >= study.HOLDOUT_START).select(
        KEY_COLUMNS
    )
    elastic_keys = elastic_selected.filter(
        pl.col("Date") >= study.HOLDOUT_START
    ).select(KEY_COLUMNS)
    overlap = ridge_keys.join(elastic_keys, on=list(KEY_COLUMNS), how="inner")
    union_count = len(ridge_keys) + len(elastic_keys) - len(overlap)

    diagnostics = {
        "ridge_selected_variant": ridge_variant,
        "elastic_net_selected_variant": elastic_variant,
        "static_holdout_parity": True,
        "score_correlations_2026": score_correlations,
        "selected_signal_overlap_2026": {
            "ridge_rows": len(ridge_keys),
            "elastic_net_rows": len(elastic_keys),
            "intersection_rows": len(overlap),
            "jaccard": len(overlap) / union_count if union_count else None,
        },
    }
    _atomic_json(_jsonable(diagnostics), OUTPUT_DIR / "comparison_diagnostics.json")
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()) and not args.force:
        raise FileExistsError(f"{OUTPUT_DIR} already exists; use a new version or --force")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    study, runner = _load_source_modules()
    model_input_path = runner.MODEL_INPUT_PATH
    model_input = pl.read_parquet(model_input_path)
    predictions = train_elastic_net(
        model_input, study=study, runner=runner, n_jobs=args.n_jobs
    )
    thresholded = apply_thresholds(predictions, study=study)
    signals = build_signals(thresholded, study=study)
    summary, decision = summarize(
        signals, study=study, runner=runner
    )
    diagnostics = compare_and_plot(
        elastic_summary=summary,
        elastic_decision=decision,
        elastic_signals=signals,
        study=study,
        runner=runner,
    )
    manifest = {
        "study_id": STUDY_DIR.name,
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "source_model_input": str(model_input_path),
        "source_model_input_sha256": _sha256(model_input_path),
        "source_ridge_summary_sha256": _sha256(runner.VARIANT_SUMMARY_PATH),
        "source_ridge_predictions_sha256": _sha256(runner.PREDICTION_PATH),
        "feature_manifest": study.FEATURE_MANIFEST.canonical_dict(),
        "feature_manifest_hash": study.FEATURE_MANIFEST.fingerprint,
        "feature_screening": False,
        "elastic_net": {
            "l1_ratio": 0.5,
            "requested_alpha": study.RIDGE_ALPHA,
            "alpha_mode": "ridge_l2_equivalent",
            "max_iter": 50_000,
            "tolerance": 1e-4,
        },
        "elastic_net_decision": decision,
        "diagnostics": diagnostics,
    }
    _atomic_json(_jsonable(manifest), OUTPUT_DIR / "run_manifest.json")
    print(summary.sort(["period", "signal_variant"]))
    print(json.dumps(_jsonable(diagnostics), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
