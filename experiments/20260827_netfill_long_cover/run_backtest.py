"""Materialize and keep the frozen signed-FillLots long-cover backtest.

The candidate registry and all gates are pinned in ``study_contract.json``.
This runner deliberately has no model, score threshold, or outcome-dependent
candidate selection.  It rebuilds selector keys from the complete raw stream,
attaches the long taker label, applies one identical execution/capacity
contract to every candidate, and preserves every member of the six-trial
family.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from src.dataloader.txfDataLoader import TxfDataLoader
from src.research.backTest import BackTest
from src.research.intradaySignal.framework.dataset import (
    MergeOptions,
    UniverseSpec,
    materialize_event_day,
)
from src.research.intradaySignal.framework.events import DataRoots, EventDaySources
from src.research.intradaySignal.framework.labels import add_directional_labels
from src.research.intradaySignal.framework.profiles.benchmark import (
    LONG_TAKER_TP109_V1,
)
from src.research.intradaySignal.framework.profiles.filllots_at_high import (
    select_filllots_at_high_feature_candidates,
)
from src.research.intradaySignal.framework.profiles.filllots_at_low import (
    select_filllots_at_low_candidates,
)

from analysis_utils import (
    calendar_pnl_metrics,
    cover_portfolio_metrics,
    deflated_sharpe_ratio,
    moving_block_max_t,
)


PROJECT_ROOT = Path(__file__).resolve().parents[5]
STUDY_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = STUDY_DIR / "study_contract.json"
OUTPUT_DIR = STUDY_DIR / "artifacts" / "static_long_cover_v2"
DAILY_SIGNAL_DIR = OUTPUT_DIR / "daily_signals"
DAILY_COUNT_DIR = OUTPUT_DIR / "daily_counts"

SHORT_SNAPSHOT_DIR = (
    PROJECT_ROOT
    / "src/research/intradaySignal/experiments/20260827_fh_neg_rejection_robustness"
    / "artifacts/static_backtest_retrospective_v1"
)
SHORT_DAILY_PATH = SHORT_SNAPSHOT_DIR / "daily.parquet"
SHORT_MANIFEST_PATH = SHORT_SNAPSHOT_DIR / "run_manifest.json"
SHORT_VERIFICATION_PATH = SHORT_SNAPSHOT_DIR / "verification.json"
HISTORICAL_TRIAL_EXPOSURE_PATH = (
    PROJECT_ROOT
    / "src/research/intradaySignal/experiments/20260827_fh_neg_rejection_robustness"
    / "historical_trial_exposure.json"
)
BACKTEST_SOURCE_PATH = PROJECT_ROOT / "src/research/backTest.py"
HIGH_PROFILE_PATH = (
    PROJECT_ROOT
    / "src/research/intradaySignal/framework/profiles/filllots_at_high.py"
)
LOW_PROFILE_PATH = (
    PROJECT_ROOT
    / "src/research/intradaySignal/framework/profiles/filllots_at_low.py"
)
LABEL_SOURCE_PATH = (
    PROJECT_ROOT / "src/research/intradaySignal/framework/labels.py"
)

HIGH_CANDIDATES = (
    "fh_pos_sample",
    "fh_pos_depth2",
    "fh_pos_breakout",
)
LOW_CANDIDATES = (
    "depth_gt_pos1_epoch_entry",
    "depth_gt_pos2_epoch_entry",
    "depth_gt_pos4_epoch_entry",
)
CANDIDATES = HIGH_CANDIDATES + LOW_CANDIDATES
CANDIDATE_FAMILY = {
    **{candidate: "FillLots_atHigh" for candidate in HIGH_CANDIDATES},
    **{candidate: "FillLots_atLow" for candidate in LOW_CANDIDATES},
}
KEY_COLUMNS = ("Date", "QuoteCode", "ChannelSeq")
DAY_KEY_COLUMNS = ("QuoteCode", "ChannelSeq")
SOURCE_NAMES = (
    "tick_data",
    "tick_feature",
    "pre_market",
    "tick_bar",
    "txf_tick",
)
REQUIRED_SOURCE_NAMES = tuple(
    source for source in SOURCE_NAMES if source != "txf_tick"
)

PRIMARY_FEE_BP = 19.3
FEE_GRID_BP = (0.0, 10.0, 19.3, 25.0, 35.0)
RAW_LABEL_ABS_MAX_BP = 10_000.0
PER_STOCK_POSITION_CAP_WAN = 200.0
PRICE_CAP = 1000.0
UNIVERSE_START_TIME = dt.time(9, 0, 30)
UNIVERSE_END_TIME = dt.time(12, 30)
UNIVERSE_PRIOR_RANK_MAX = 150
UNIVERSE_BID_REF_LOWER = -0.015
UNIVERSE_BID_REF_UPPER = 0.05
UNIVERSE_BID_REF_ROUND_DECIMALS = 6
TIME_BUCKET_CUTS = (3300.0, 6900.0, 10500.0, 14100.0)
TO_REF_BUCKET_CUTS = (0.0, 0.01, 0.02, 0.03, 0.05)
MD_L1_RATE_MIN = 0.25
SPREAD_PAIR_ELAPSED_MIN = 0.1
EXECUTION_RANK_MAX = 100
MAKER_SPREAD_BP_MIN = -70.0
PERIODS = {
    "full_retrospective": ("20250211", "20260825"),
    "year_2025": ("20250211", "20251231"),
    "year_2026_historical": ("20260101", "20260825"),
}

ACCEPTED_PATH = OUTPUT_DIR / "accepted_signals.parquet"
DAILY_PATH = OUTPUT_DIR / "daily.parquet"
DAILY_COUNTS_PATH = OUTPUT_DIR / "daily_counts.parquet"
PERIOD_SUMMARY_PATH = OUTPUT_DIR / "period_summary.parquet"
MONTHLY_SUMMARY_PATH = OUTPUT_DIR / "monthly_summary.parquet"
FEE_SENSITIVITY_PATH = OUTPUT_DIR / "fee_sensitivity.parquet"
FAMILY_BIAS_PATH = OUTPUT_DIR / "family_bias_adjustment.json"
TRIAL_LEDGER_PATH = OUTPUT_DIR / "trial_ledger.parquet"
COVER_SUMMARY_PATH = OUTPUT_DIR / "cover_summary.parquet"
DECISION_PATH = OUTPUT_DIR / "decision.json"
VERIFICATION_PATH = OUTPUT_DIR / "verification.json"
RUN_MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
COMPARISON_PLOT_PATH = OUTPUT_DIR / "comparison_full.png"
COVER_PLOT_PATH = OUTPUT_DIR / "cover_comparison.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path, *, content_hash: bool = True) -> dict[str, Any]:
    stat = path.stat()
    output: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if content_hash:
        output["sha256"] = _sha256(path)
    return output


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
        json.dump(
            _json_value(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        frame.write_parquet(temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _assert_unique_keys(
    frame: pl.DataFrame,
    keys: Iterable[str],
    context: str,
) -> None:
    key_list = list(keys)
    missing = sorted(set(key_list) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{context}: missing keys {missing}")
    if frame.is_empty():
        return
    nulls = frame.select(
        pl.any_horizontal([pl.col(column).is_null() for column in key_list])
        .sum()
        .alias("n")
    ).item()
    duplicates = frame.select(pl.struct(key_list).is_duplicated().sum()).item()
    if nulls or duplicates:
        raise RuntimeError(
            f"{context}: null keys={nulls}, duplicate keys={duplicates}"
        )


def _eligible_long_universe(frame: pl.DataFrame) -> pl.DataFrame:
    bid_to_ref = (
        (pl.col("BidPrice1") - pl.col("RefPrice")) / pl.col("RefPrice")
    ).round(UNIVERSE_BID_REF_ROUND_DECIMALS)
    return frame.filter(
        (pl.col("AskPrice1") > 0.0)
        & (pl.col("RefPrice") > 0.0)
        & (pl.col("TrialMatch") == 0)
        & (pl.col("TransTime").dt.time() > UNIVERSE_START_TIME)
        & (pl.col("TransTime").dt.time() < UNIVERSE_END_TIME)
        & (pl.col("nextday_allow_day_trade_mark") == "X")
        & (pl.col("day_amount_rank") <= UNIVERSE_PRIOR_RANK_MAX)
        & (bid_to_ref > UNIVERSE_BID_REF_LOWER)
        & (bid_to_ref < UNIVERSE_BID_REF_UPPER)
    )


LONG_UNIVERSE = UniverseSpec(
    universe_id="netfill_long_daytrade_rank150_090030_123000_bidref",
    universe_version="1",
    filter_fn=_eligible_long_universe,
)
MERGE_OPTIONS = MergeOptions(
    universe=LONG_UNIVERSE,
    require_pre_market=True,
    attach_tick_bar=True,
    require_tick_bar=True,
    tick_bar_alignment="causal_completed_slot",
    tick_bar_columns=("AmountRank", "AmountRank_canDayTrade"),
    attach_txf=True,
    require_txf=False,
    strict_cardinality=True,
    schema_policy="union",
)


def _add_threshold_buckets(frame: pl.DataFrame) -> pl.DataFrame:
    """Exact inherited four-by-four positive-ToRef execution cells."""

    return frame.with_columns(
        pl.col("RemainSeconds")
        .cut(
            list(TIME_BUCKET_CUTS),
            labels=["outside_low", "t0", "t1", "t2", "t3"],
            left_closed=False,
        )
        .alias("_time_cut"),
        pl.col("ToRef")
        .cut(
            list(TO_REF_BUCKET_CUTS),
            labels=[
                "outside_low",
                "r0",
                "r1",
                "r2",
                "r3",
                "outside_high",
            ],
            left_closed=False,
        )
        .alias("_ref_cut"),
    ).with_columns(
        pl.when(pl.col("_time_cut").is_in(["t0", "t1", "t2", "t3"]))
        .then(pl.col("_time_cut").cast(pl.String))
        .alias("time_bucket"),
        pl.when(pl.col("_ref_cut").is_in(["r0", "r1", "r2", "r3"]))
        .then(pl.col("_ref_cut").cast(pl.String))
        .alias("to_ref_bucket"),
    ).drop("_time_cut", "_ref_cut")


def _static_execution_filter(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(
        pl.col("time_bucket").is_not_null()
        & pl.col("to_ref_bucket").is_not_null()
        & ((pl.col("OTC") == True) | (pl.col("MD_L1Rate_30_re") > MD_L1_RATE_MIN))
        & (pl.col("SpreadPairElapsed") > SPREAD_PAIR_ELAPSED_MIN)
        & (
            (pl.col("AmountRank_canDayTrade") <= EXECUTION_RANK_MAX)
            | (pl.col("day_amount_rank") <= EXECUTION_RANK_MAX)
        )
        & (pl.col("AskPrice1") <= PRICE_CAP)
        & (
            pl.col("hft_strick_makerSpreadBP").is_null()
            | (pl.col("hft_strick_makerSpreadBP") > MAKER_SPREAD_BP_MIN)
        )
    )


def _apply_capacity(frame: pl.DataFrame) -> pl.DataFrame:
    ordered = frame.sort(
        ["candidate_id", "Date", "QuoteCode", "TransTime", "ChannelSeq"]
    )
    groups = ["candidate_id", "Date", "QuoteCode"]
    ordered = ordered.with_columns(
        pl.col("AskPrice1").cum_count().over(groups).alias("accLots"),
        (pl.col("AskPrice1").cum_sum().over(groups) / 10.0).alias(
            "stock_position_wan"
        ),
    )
    return ordered.filter(
        (pl.col("accLots") < pl.col("avg_bidLots1") + pl.col("avg_askLots1"))
        & (pl.col("stock_position_wan") < PER_STOCK_POSITION_CAP_WAN)
    )


def _read_contract() -> tuple[dict[str, Any], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract["candidate_family"]["trial_count"] != len(CANDIDATES):
        raise RuntimeError("contract candidate trial count drifted")
    contract_candidates = tuple(
        item["candidate_id"]
        for section in ("at_high_nested_ablation", "at_low_prior_depth_neighborhood")
        for item in contract["candidate_family"][section]
    )
    if contract_candidates != CANDIDATES:
        raise RuntimeError(
            f"contract candidate order drift: {contract_candidates} != {CANDIDATES}"
        )
    runtime = contract["runtime_assertions"]
    expected_runtime = {
        "artifact_version": OUTPUT_DIR.name,
        "candidate_ids": list(CANDIDATES),
        "source_names": list(SOURCE_NAMES),
        "label_profile_id": LONG_TAKER_TP109_V1.profile_id,
        "label_profile_sha256": LONG_TAKER_TP109_V1.fingerprint,
        "raw_label_abs_max_bp_strict": RAW_LABEL_ABS_MAX_BP,
        "universe": {
            "id": LONG_UNIVERSE.universe_id,
            "version": LONG_UNIVERSE.universe_version,
            "start_time_strict": UNIVERSE_START_TIME.isoformat(),
            "end_time_strict": UNIVERSE_END_TIME.isoformat(),
            "prior_rank_max": UNIVERSE_PRIOR_RANK_MAX,
            "bid_to_ref_lower_strict": UNIVERSE_BID_REF_LOWER,
            "bid_to_ref_upper_strict": UNIVERSE_BID_REF_UPPER,
            "bid_to_ref_round_decimals": UNIVERSE_BID_REF_ROUND_DECIMALS,
        },
        "merge_options": {
            "require_pre_market": MERGE_OPTIONS.require_pre_market,
            "attach_tick_bar": MERGE_OPTIONS.attach_tick_bar,
            "require_tick_bar": MERGE_OPTIONS.require_tick_bar,
            "tick_bar_alignment": MERGE_OPTIONS.tick_bar_alignment,
            "tick_bar_columns": list(MERGE_OPTIONS.tick_bar_columns or ()),
            "attach_txf": MERGE_OPTIONS.attach_txf,
            "require_txf": MERGE_OPTIONS.require_txf,
            "strict_cardinality": MERGE_OPTIONS.strict_cardinality,
            "schema_policy": MERGE_OPTIONS.schema_policy,
        },
        "execution": {
            "md_l1_rate_min_strict": MD_L1_RATE_MIN,
            "spread_pair_elapsed_min_strict": SPREAD_PAIR_ELAPSED_MIN,
            "current_or_prior_rank_max": EXECUTION_RANK_MAX,
            "ask_price_max": PRICE_CAP,
            "maker_spread_bp_min_strict_if_nonnull": MAKER_SPREAD_BP_MIN,
            "to_ref_positive_upper_inclusive": TO_REF_BUCKET_CUTS[-1],
            "time_bucket_cuts": list(TIME_BUCKET_CUTS),
            "to_ref_bucket_cuts": list(TO_REF_BUCKET_CUTS),
        },
        "capacity": {
            "lots_per_event": 1,
            "per_stock_position_cap_wan_strict": PER_STOCK_POSITION_CAP_WAN,
            "prior_depth_cap_strict": True,
        },
        "cost": {
            "primary_fee_bp": PRIMARY_FEE_BP,
            "fee_grid_bp": list(FEE_GRID_BP),
        },
        "cover": {
            "native_multiplier_grid": contract["cover_evaluation"][
                "fixed_native_long_multiplier_grid"
            ],
            "anchor_native_multiplier": contract["cover_evaluation"][
                "anchor_native_multiplier"
            ],
            "correlation_gate": "both_pearson_and_spearman_negative",
        },
    }
    if runtime != expected_runtime:
        raise RuntimeError(
            "frozen runtime assertions drifted: "
            f"contract={runtime!r}, code={expected_runtime!r}"
        )
    return contract, _sha256(CONTRACT_PATH)


def _read_calendar(contract: dict[str, Any]) -> list[str]:
    path = PROJECT_ROOT / contract["calendar"]["source"]
    frame = pl.read_parquet(path)
    dates = (
        frame.filter(
            (pl.col("Date") >= contract["calendar"]["start"])
            & (pl.col("Date") <= contract["calendar"]["end"])
        )["Date"]
        .unique()
        .sort()
        .to_list()
    )
    if len(dates) != contract["calendar"]["requested_sessions"]:
        raise RuntimeError(
            f"calendar drift: {len(dates)} != "
            f"{contract['calendar']['requested_sessions']}"
        )
    if dates[0] != contract["calendar"]["start"] or dates[-1] != contract[
        "calendar"
    ]["end"]:
        raise RuntimeError("calendar endpoint drift")
    return dates


def _source_identities(day: EventDaySources) -> str:
    day.require(*REQUIRED_SOURCE_NAMES)
    identities: dict[str, dict[str, Any]] = {}
    for source in SOURCE_NAMES:
        path = day.path(source)
        if not path.exists():
            identities[source] = {
                "path": str(path.resolve()),
                "exists": False,
            }
            continue
        stat = path.stat()
        identities[source] = {
            "path": str(path.resolve()),
            "exists": True,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return json.dumps(
        identities,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _candidate_markers(day: EventDaySources) -> pl.DataFrame:
    high = select_filllots_at_high_feature_candidates(
        day, candidate_ids=HIGH_CANDIDATES
    )
    low = select_filllots_at_low_candidates(day, candidate_ids=LOW_CANDIDATES)
    selections = {**high, **low}
    markers = pl.concat(
        [
            selections[candidate]
            .with_columns(
                pl.lit(candidate).alias("candidate_id"),
                pl.lit(CANDIDATE_FAMILY[candidate]).alias("candidate_family"),
            )
            .select(*DAY_KEY_COLUMNS, "candidate_id", "candidate_family")
            for candidate in CANDIDATES
        ],
        how="vertical",
    )
    _assert_unique_keys(
        markers,
        ("candidate_id", *DAY_KEY_COLUMNS),
        f"{day.date} candidate markers",
    )
    return markers


def _count_by_candidate(
    frame: pl.DataFrame,
    *,
    count_name: str,
) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(
            {"candidate_id": [], count_name: []},
            schema={"candidate_id": pl.String, count_name: pl.Int64},
        )
    return frame.group_by("candidate_id").len(name=count_name)


ACCEPTED_COLUMNS = (
    "contract_sha256",
    "materialization_identity_sha256",
    "Date",
    "candidate_id",
    "candidate_family",
    "QuoteCode",
    "ChannelSeq",
    "TransTime",
    "BidPrice1",
    "AskPrice1",
    "Close",
    "FutureHigh",
    "RefPrice",
    "label_entry_price",
    "label_exit_price",
    "label_raw_bp",
    "label_market_component_bp",
    "label_hedge_pnl_bp",
    "label_residual_bp",
    "label_z",
    "label_profile_id",
    "OTC",
    "MD_L1Rate_30_re",
    "SpreadPairElapsed",
    "RemainSeconds",
    "ToRef",
    "time_bucket",
    "to_ref_bucket",
    "AmountRank_canDayTrade",
    "day_amount_rank",
    "avg_bidLots1",
    "avg_askLots1",
    "hft_strick_makerSpreadBP",
    "accLots",
    "stock_position_wan",
    "lots",
    "TakerSell_CloseBP",
    "BackTestEntryPrice",
    "signal_variant",
)


def _empty_accepted() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "contract_sha256": pl.String,
            "materialization_identity_sha256": pl.String,
            "Date": pl.String,
            "candidate_id": pl.String,
            "candidate_family": pl.String,
            "QuoteCode": pl.String,
            "ChannelSeq": pl.UInt64,
            "TransTime": pl.Datetime("us"),
            "BidPrice1": pl.Float64,
            "AskPrice1": pl.Float64,
            "Close": pl.Float64,
            "FutureHigh": pl.Float64,
            "RefPrice": pl.Float64,
            "label_entry_price": pl.Float64,
            "label_exit_price": pl.Float64,
            "label_raw_bp": pl.Float64,
            "label_market_component_bp": pl.Float64,
            "label_hedge_pnl_bp": pl.Float64,
            "label_residual_bp": pl.Float64,
            "label_z": pl.Float64,
            "label_profile_id": pl.String,
            "OTC": pl.Boolean,
            "MD_L1Rate_30_re": pl.Float64,
            "SpreadPairElapsed": pl.Float64,
            "RemainSeconds": pl.Int64,
            "ToRef": pl.Float64,
            "time_bucket": pl.String,
            "to_ref_bucket": pl.String,
            "AmountRank_canDayTrade": pl.Int64,
            "day_amount_rank": pl.Int64,
            "avg_bidLots1": pl.Float64,
            "avg_askLots1": pl.Float64,
            "hft_strick_makerSpreadBP": pl.Float64,
            "accLots": pl.UInt32,
            "stock_position_wan": pl.Float64,
            "lots": pl.Int8,
            "TakerSell_CloseBP": pl.Float64,
            "BackTestEntryPrice": pl.Float64,
            "signal_variant": pl.String,
        }
    )


def _cached_day_valid(
    signal_path: Path,
    count_path: Path,
    *,
    date: str,
    contract_sha256: str,
    materialization_identity_sha256: str,
    source_identities_json: str,
) -> bool:
    if not signal_path.exists() or not count_path.exists():
        return False
    counts = pl.read_parquet(count_path)
    checks = {
        "Date": date,
        "contract_sha256": contract_sha256,
        "materialization_identity_sha256": materialization_identity_sha256,
        "source_identities_json": source_identities_json,
    }
    for column, expected in checks.items():
        if column not in counts.columns or counts[column].unique().to_list() != [expected]:
            return False
    if set(counts["candidate_id"].to_list()) != set(CANDIDATES) or len(counts) != len(
        CANDIDATES
    ):
        return False
    signals = pl.read_parquet(signal_path)
    if signals.is_empty():
        return set(signals.columns) == set(ACCEPTED_COLUMNS)
    return (
        signals["Date"].unique().to_list() == [date]
        and signals["contract_sha256"].unique().to_list() == [contract_sha256]
        and signals["materialization_identity_sha256"].unique().to_list()
        == [materialization_identity_sha256]
        and set(signals["candidate_id"].unique().to_list()).issubset(CANDIDATES)
    )


def _materialize_day(
    date: str,
    *,
    roots: DataRoots,
    txf_loader: TxfDataLoader,
    contract_sha256: str,
    materialization_identity_sha256: str,
    materialization_identity_json: str,
    source_identities_json: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    day = EventDaySources(date, roots)
    markers = _candidate_markers(day)
    indexed = _count_by_candidate(markers, count_name="indexed_rows")
    union_keys = markers.select(DAY_KEY_COLUMNS).unique().sort(DAY_KEY_COLUMNS)

    materialized = materialize_event_day(
        date,
        union_keys,
        roots=roots,
        options=MERGE_OPTIONS,
        txf_loader=txf_loader,
    ).frame
    eligible_marked = materialized.join(
        markers,
        on=list(DAY_KEY_COLUMNS),
        how="inner",
        validate="1:m",
    )
    eligible = _count_by_candidate(eligible_marked, count_name="eligible_rows")

    all_labels = add_directional_labels(
        materialized,
        spec=LONG_TAKER_TP109_V1,
    )
    labeled = all_labels.filter(
        pl.col("label_raw_bp").is_finite()
        & (pl.col("label_raw_bp").abs() < RAW_LABEL_ABS_MAX_BP)
    )
    labeled_marked = labeled.join(
        markers,
        on=list(DAY_KEY_COLUMNS),
        how="inner",
        validate="1:m",
    )
    labeled_counts = _count_by_candidate(
        labeled_marked, count_name="labeled_rows"
    )
    diagnostic_valid_counts = _count_by_candidate(
        all_labels.filter(pl.col("label_valid"))
        .join(
            markers,
            on=list(DAY_KEY_COLUMNS),
            how="inner",
            validate="1:m",
        ),
        count_name="diagnostic_label_valid_rows",
    )
    execution_ready = _add_threshold_buckets(
        labeled_marked.with_columns(
            (
                (pl.col("BidPrice1") - pl.col("RefPrice"))
                / pl.col("RefPrice")
            )
            .round(6)
            .alias("ToRef"),
            pl.col("MD_L1Rate_30").cast(pl.Float64).alias(
                "MD_L1Rate_30_re"
            ),
        )
    )
    pre_capacity = _static_execution_filter(execution_ready)
    pre_capacity_counts = _count_by_candidate(
        pre_capacity, count_name="pre_capacity_rows"
    )
    accepted = _apply_capacity(pre_capacity).with_columns(
        pl.lit(contract_sha256).alias("contract_sha256"),
        pl.lit(materialization_identity_sha256).alias(
            "materialization_identity_sha256"
        ),
        pl.lit(1, dtype=pl.Int8).alias("lots"),
        pl.col("label_raw_bp").alias("TakerSell_CloseBP"),
        pl.col("AskPrice1").alias("BackTestEntryPrice"),
        pl.col("candidate_id").alias("signal_variant"),
    )
    if accepted.is_empty():
        accepted = _empty_accepted()
    else:
        accepted = accepted.select(ACCEPTED_COLUMNS).cast(
            _empty_accepted().schema, strict=False
        )
    accepted_counts = _count_by_candidate(
        accepted, count_name="accepted_rows"
    )

    counts = (
        pl.DataFrame(
            {
                "candidate_id": list(CANDIDATES),
                "candidate_family": [
                    CANDIDATE_FAMILY[candidate] for candidate in CANDIDATES
                ],
            }
        )
        .join(indexed, on="candidate_id", how="left")
        .join(eligible, on="candidate_id", how="left")
        .join(labeled_counts, on="candidate_id", how="left")
        .join(diagnostic_valid_counts, on="candidate_id", how="left")
        .join(pre_capacity_counts, on="candidate_id", how="left")
        .join(accepted_counts, on="candidate_id", how="left")
        .with_columns(
            pl.lit(date).alias("Date"),
            pl.lit(contract_sha256).alias("contract_sha256"),
            pl.lit(materialization_identity_sha256).alias(
                "materialization_identity_sha256"
            ),
            pl.lit(materialization_identity_json).alias(
                "materialization_identity_json"
            ),
            pl.lit(source_identities_json).alias("source_identities_json"),
            pl.col("indexed_rows").fill_null(0),
            pl.col("eligible_rows").fill_null(0),
            pl.col("labeled_rows").fill_null(0),
            pl.col("diagnostic_label_valid_rows").fill_null(0),
            pl.col("pre_capacity_rows").fill_null(0),
            pl.col("accepted_rows").fill_null(0),
        )
        .select(
            "contract_sha256",
            "materialization_identity_sha256",
            "Date",
            "candidate_id",
            "candidate_family",
            "indexed_rows",
            "eligible_rows",
            "labeled_rows",
            "diagnostic_label_valid_rows",
            "pre_capacity_rows",
            "accepted_rows",
            "materialization_identity_json",
            "source_identities_json",
        )
        .sort("candidate_id")
    )
    return accepted, counts


def _materialization_identity(
    contract_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    """Bind resumable partitions to every directly output-affecting input."""

    local_functions = (
        _assert_unique_keys,
        _eligible_long_universe,
        _add_threshold_buckets,
        _static_execution_filter,
        _apply_capacity,
        _candidate_markers,
        _count_by_candidate,
        _empty_accepted,
        _materialize_day,
    )
    dependency_paths = (
        PROJECT_ROOT / "src/dataloader/txfDataLoader.py",
        PROJECT_ROOT / "src/research/intradaySignal/framework/dataset.py",
        PROJECT_ROOT / "src/research/intradaySignal/framework/events.py",
        PROJECT_ROOT / "src/features/market_beta.py",
        LABEL_SOURCE_PATH,
        PROJECT_ROOT / "src/research/intradaySignal/framework/profiles/benchmark.py",
        HIGH_PROFILE_PATH,
        LOW_PROFILE_PATH,
    )
    local_source = "\n\n".join(
        inspect.getsource(function) for function in local_functions
    )
    payload = {
        "protocol": "netfill_long_cover_materialization_identity/v1",
        "contract_sha256": contract_sha256,
        "runtime": {
            "python": platform.python_version(),
            "polars": importlib.metadata.version("polars"),
        },
        "local_function_names": [
            function.__name__ for function in local_functions
        ],
        "local_source_sha256": hashlib.sha256(
            local_source.encode("utf-8")
        ).hexdigest(),
        "dependency_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in dependency_paths
        },
        "constants": {
            "candidates": list(CANDIDATES),
            "candidate_family": CANDIDATE_FAMILY,
            "source_names": list(SOURCE_NAMES),
            "price_cap": PRICE_CAP,
            "per_stock_position_cap_wan": PER_STOCK_POSITION_CAP_WAN,
            "universe_start_time": UNIVERSE_START_TIME.isoformat(),
            "universe_end_time": UNIVERSE_END_TIME.isoformat(),
            "universe_prior_rank_max": UNIVERSE_PRIOR_RANK_MAX,
            "universe_bid_ref_lower": UNIVERSE_BID_REF_LOWER,
            "universe_bid_ref_upper": UNIVERSE_BID_REF_UPPER,
            "universe_bid_ref_round_decimals": (
                UNIVERSE_BID_REF_ROUND_DECIMALS
            ),
            "time_bucket_cuts": list(TIME_BUCKET_CUTS),
            "to_ref_bucket_cuts": list(TO_REF_BUCKET_CUTS),
            "md_l1_rate_min": MD_L1_RATE_MIN,
            "spread_pair_elapsed_min": SPREAD_PAIR_ELAPSED_MIN,
            "execution_rank_max": EXECUTION_RANK_MAX,
            "maker_spread_bp_min": MAKER_SPREAD_BP_MIN,
            "accepted_columns": list(ACCEPTED_COLUMNS),
            "universe_id": LONG_UNIVERSE.universe_id,
            "universe_version": LONG_UNIVERSE.universe_version,
            "merge_options": {
                "require_pre_market": MERGE_OPTIONS.require_pre_market,
                "attach_tick_bar": MERGE_OPTIONS.attach_tick_bar,
                "require_tick_bar": MERGE_OPTIONS.require_tick_bar,
                "tick_bar_alignment": MERGE_OPTIONS.tick_bar_alignment,
                "tick_bar_columns": list(MERGE_OPTIONS.tick_bar_columns or ()),
                "attach_txf": MERGE_OPTIONS.attach_txf,
                "require_txf": MERGE_OPTIONS.require_txf,
                "strict_cardinality": MERGE_OPTIONS.strict_cardinality,
                "feature_suffix": MERGE_OPTIONS.feature_suffix,
                "schema_policy": MERGE_OPTIONS.schema_policy,
                "preserve_tick_data_column_order": (
                    MERGE_OPTIONS.preserve_tick_data_column_order
                ),
            },
            "label_profile_id": LONG_TAKER_TP109_V1.profile_id,
            "label_profile_sha256": LONG_TAKER_TP109_V1.fingerprint,
            "raw_label_abs_max_bp_strict": RAW_LABEL_ABS_MAX_BP,
        },
    }
    identity_json = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    identity_sha256 = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
    return payload, identity_json, identity_sha256


def _calendar_sharpe(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return math.nan
    std = float(np.std(values, ddof=1))
    if not math.isfinite(std) or std <= 0.0:
        return math.nan
    return float(np.mean(values) / std * math.sqrt(252.0))


def _drawdown(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    cumulative = np.cumsum(values)
    if len(cumulative):
        high_water = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))[
            1:
        ]
    else:
        high_water = cumulative.copy()
    return cumulative, high_water, cumulative - high_water


def _daily_frame(
    signals: pl.DataFrame,
    calendar: list[str],
    *,
    fee_bp: float,
) -> pl.DataFrame:
    calendar_frame = pl.DataFrame({"Date": calendar})
    rows: list[pl.DataFrame] = []
    for candidate in CANDIDATES:
        subset = signals.filter(pl.col("candidate_id") == candidate)
        if subset.is_empty():
            aggregated = pl.DataFrame(
                schema={
                    "Date": pl.String,
                    "gross_pnl": pl.Float64,
                    "net_pnl": pl.Float64,
                    "residual_gross_pnl": pl.Float64,
                    "residual_net_pnl": pl.Float64,
                    "residual_position": pl.Float64,
                    "residual_trades": pl.UInt32,
                    "position": pl.Float64,
                    "trades": pl.UInt32,
                    "stocks": pl.UInt32,
                }
            )
        else:
            aggregated = (
                subset.with_columns(
                    (
                        pl.col("AskPrice1") * pl.col("label_raw_bp") / 10.0
                    ).alias("gross_pnl"),
                    (
                        pl.col("AskPrice1")
                        * (pl.col("label_raw_bp") - fee_bp)
                        / 10.0
                    ).alias("net_pnl"),
                    pl.when(pl.col("label_residual_bp").is_finite())
                    .then(
                        pl.col("AskPrice1")
                        * pl.col("label_residual_bp")
                        / 10.0
                    )
                    .otherwise(0.0)
                    .alias("residual_gross_pnl"),
                    pl.when(pl.col("label_residual_bp").is_finite())
                    .then(
                        pl.col("AskPrice1")
                        * (pl.col("label_residual_bp") - fee_bp)
                        / 10.0
                    )
                    .otherwise(0.0)
                    .alias("residual_net_pnl"),
                    pl.when(pl.col("label_residual_bp").is_finite())
                    .then(pl.col("AskPrice1") * 1000.0)
                    .otherwise(0.0)
                    .alias("residual_position"),
                    pl.col("label_residual_bp")
                    .is_finite()
                    .fill_null(False)
                    .cast(pl.UInt32)
                    .alias("residual_trades"),
                    (pl.col("AskPrice1") * 1000.0).alias("position"),
                )
                .group_by("Date")
                .agg(
                    pl.col("gross_pnl").sum(),
                    pl.col("net_pnl").sum(),
                    pl.col("residual_gross_pnl").sum(),
                    pl.col("residual_net_pnl").sum(),
                    pl.col("residual_position").sum(),
                    pl.col("residual_trades").sum(),
                    pl.col("position").sum(),
                    pl.len().cast(pl.UInt32).alias("trades"),
                    pl.col("QuoteCode").n_unique().cast(pl.UInt32).alias("stocks"),
                )
            )
        daily = (
            calendar_frame.join(aggregated, on="Date", how="left")
            .with_columns(
                pl.lit(candidate).alias("candidate_id"),
                pl.lit(CANDIDATE_FAMILY[candidate]).alias("candidate_family"),
                pl.lit(float(fee_bp)).alias("fee_bp"),
                pl.col("gross_pnl").fill_null(0.0),
                pl.col("net_pnl").fill_null(0.0),
                pl.col("residual_gross_pnl").fill_null(0.0),
                pl.col("residual_net_pnl").fill_null(0.0),
                pl.col("residual_position").fill_null(0.0),
                pl.col("residual_trades").fill_null(0),
                pl.col("position").fill_null(0.0),
                pl.col("trades").fill_null(0),
                pl.col("stocks").fill_null(0),
            )
            .sort("Date")
        )
        cumulative, high_water, drawdown = _drawdown(
            daily["net_pnl"].to_numpy()
        )
        daily = daily.with_columns(
            pl.Series("cumulative_net_pnl", cumulative),
            pl.Series("high_water_mark", high_water),
            pl.Series("drawdown", drawdown),
        )
        rows.append(daily)
    return pl.concat(rows, how="vertical").sort("candidate_id", "Date")


def _period_dates(calendar: list[str], bounds: tuple[str, str]) -> list[str]:
    start, end = bounds
    dates = [date for date in calendar if start <= date <= end]
    if not dates:
        raise RuntimeError(f"period {bounds} has no dates")
    return dates


def _summarize_period(
    signals: pl.DataFrame,
    daily: pl.DataFrame,
    *,
    candidate: str,
    period: str,
    dates: list[str],
    fee_bp: float,
) -> dict[str, Any]:
    subset = signals.filter(
        (pl.col("candidate_id") == candidate) & pl.col("Date").is_in(dates)
    )
    day = daily.filter(
        (pl.col("candidate_id") == candidate) & pl.col("Date").is_in(dates)
    ).sort("Date")
    if len(day) != len(dates):
        raise RuntimeError(f"{candidate}/{period}: calendar rows drifted")
    position = float(day["position"].sum())
    gross = float(day["gross_pnl"].sum())
    net = float(day["net_pnl"].sum())
    residual_gross = float(day["residual_gross_pnl"].sum())
    residual_net = float(day["residual_net_pnl"].sum())
    residual_position = float(day["residual_position"].sum())
    residual_trades = int(day["residual_trades"].sum())
    pnl = day["net_pnl"].to_numpy()
    _, _, drawdown = _drawdown(pnl)
    return {
        "candidate_id": candidate,
        "candidate_family": CANDIDATE_FAMILY[candidate],
        "period": period,
        "start": dates[0],
        "end": dates[-1],
        "fee_bp": float(fee_bp),
        "calendar_days": len(dates),
        "active_days": int((day["trades"] > 0).sum()),
        "zero_signal_days": int((day["trades"] == 0).sum()),
        "trades": int(day["trades"].sum()),
        "distinct_stocks": int(subset["QuoteCode"].n_unique())
        if not subset.is_empty()
        else 0,
        "stock_days": int(
            subset.select(pl.struct(["Date", "QuoteCode"]).n_unique()).item()
        )
        if not subset.is_empty()
        else 0,
        "gross_pnl": gross,
        "cost_pnl": gross - net,
        "net_pnl": net,
        "residual_gross_pnl": residual_gross,
        "residual_net_pnl": residual_net,
        "residual_valid_trades": residual_trades,
        "residual_coverage_rate": residual_trades / int(day["trades"].sum())
        if int(day["trades"].sum()) > 0
        else math.nan,
        "residual_position": residual_position,
        "cap_weighted_gross_bp": gross / position * 10_000.0
        if position > 0.0
        else math.nan,
        "cap_weighted_net_bp": net / position * 10_000.0
        if position > 0.0
        else math.nan,
        "cap_weighted_residual_gross_bp": residual_gross
        / residual_position
        * 10_000.0
        if residual_position > 0.0
        else math.nan,
        "cap_weighted_residual_net_bp": residual_net
        / residual_position
        * 10_000.0
        if residual_position > 0.0
        else math.nan,
        "mean_event_raw_bp": float(subset["label_raw_bp"].mean())
        if not subset.is_empty()
        else math.nan,
        "mean_event_net_bp": float(subset["label_raw_bp"].mean() - fee_bp)
        if not subset.is_empty()
        else math.nan,
        "calendar_daily_sharpe": _calendar_sharpe(pnl),
        "calendar_daily_win_rate": float(np.mean(pnl > 0.0)),
        "max_drawdown": float(np.min(drawdown)) if len(drawdown) else math.nan,
        "mean_daily_position": float(day["position"].mean()),
        "max_daily_position": float(day["position"].max()),
    }


def _build_period_summary(
    signals: pl.DataFrame,
    daily: pl.DataFrame,
    calendar: list[str],
) -> pl.DataFrame:
    rows = [
        _summarize_period(
            signals,
            daily,
            candidate=candidate,
            period=period,
            dates=_period_dates(calendar, bounds),
            fee_bp=PRIMARY_FEE_BP,
        )
        for candidate in CANDIDATES
        for period, bounds in PERIODS.items()
    ]
    return pl.DataFrame(rows).sort("candidate_id", "period")


def _build_fee_sensitivity(
    signals: pl.DataFrame,
    calendar: list[str],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for fee_bp in FEE_GRID_BP:
        fee_daily = _daily_frame(signals, calendar, fee_bp=fee_bp)
        for candidate in CANDIDATES:
            rows.append(
                _summarize_period(
                    signals,
                    fee_daily,
                    candidate=candidate,
                    period="full_retrospective",
                    dates=calendar,
                    fee_bp=fee_bp,
                )
            )
    return pl.DataFrame(rows).sort("candidate_id", "fee_bp")


def _build_monthly_summary(daily: pl.DataFrame) -> pl.DataFrame:
    return (
        daily.with_columns(pl.col("Date").str.slice(0, 6).alias("month"))
        .group_by("candidate_id", "candidate_family", "month")
        .agg(
            pl.len().alias("calendar_days"),
            (pl.col("trades") > 0).sum().alias("active_days"),
            pl.col("trades").sum(),
            pl.col("gross_pnl").sum(),
            pl.col("net_pnl").sum(),
            pl.col("position").sum(),
        )
        .with_columns(
            pl.lit(PRIMARY_FEE_BP).alias("fee_bp"),
            pl.when(pl.col("position") > 0.0)
            .then(pl.col("gross_pnl") / pl.col("position") * 10_000.0)
            .alias("cap_weighted_gross_bp"),
            pl.when(pl.col("position") > 0.0)
            .then(pl.col("net_pnl") / pl.col("position") * 10_000.0)
            .alias("cap_weighted_net_bp"),
        )
        .sort("candidate_id", "month")
    )


def _save_project_backtest_plot(
    signals: pl.DataFrame,
    daily: pl.DataFrame,
    *,
    candidate: str,
    dates: list[str],
    path: Path,
    title: str,
) -> dict[str, Any]:
    subset = signals.filter(
        (pl.col("candidate_id") == candidate) & pl.col("Date").is_in(dates)
    )
    if subset.is_empty():
        expected = 0.0
        actual = 0.0
        stats: dict[str, Any] = {
            "status": "not_run_all_zero_candidate_period"
        }
    else:
        pdf = subset.to_pandas()
        # Project BackTest taker mode names the entry column BidPrice1
        # regardless of direction.  For this endpoint adapter only, replace it
        # with the frozen long Ask entry; the canonical ledger keeps both.
        pdf["BidPrice1"] = pdf["BackTestEntryPrice"]
        backtest = BackTest(pdf, fee=PRIMARY_FEE_BP)
        stats, _ = backtest.run_analysis(return_daily=True, plot=False)
        expected = float(
            (
                pdf["BackTestEntryPrice"]
                * (pdf["TakerSell_CloseBP"] - PRIMARY_FEE_BP)
                / 10.0
            ).sum()
        )
        actual = float(backtest.df["PnL"].sum())
    if not np.isclose(expected, actual, rtol=1e-12, atol=1e-8):
        raise RuntimeError(
            f"BackTest parity failed for {candidate}: {expected} != {actual}"
        )
    day = daily.filter(
        (pl.col("candidate_id") == candidate) & pl.col("Date").is_in(dates)
    ).sort("Date")
    if day["Date"].to_list() != dates:
        raise RuntimeError(f"{candidate}: plot calendar is not dense and exact")
    daily_pnl = day["net_pnl"].to_numpy()
    cumulative, _, drawdown = _drawdown(daily_pnl)
    canonical_total = float(np.sum(daily_pnl))
    if not np.isclose(expected, canonical_total, rtol=1e-12, atol=1e-8):
        raise RuntimeError(
            f"canonical daily parity failed for {candidate}: "
            f"{expected} != {canonical_total}"
        )
    plot_dates = pd.to_datetime(day["Date"].to_list())
    figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(plot_dates, cumulative / 10_000.0, color="#2457A7")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Cumulative net PnL\n(萬 TWD)")
    axes[0].set_title(title, fontsize=13, fontweight="bold")
    axes[0].text(
        0.99,
        0.02,
        f"Ask taker · 1 lot/event · {PRIMARY_FEE_BP:.1f}bp · "
        f"calendar Sharpe {_calendar_sharpe(daily_pnl):.2f}",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    axes[1].fill_between(
        plot_dates,
        drawdown / 10_000.0,
        0.0,
        color="#B43C39",
        alpha=0.55,
    )
    axes[1].set_ylabel("Drawdown\n(萬 TWD)")
    axes[2].bar(
        plot_dates,
        day["position"].to_numpy() / 10_000.0,
        width=1.0,
        color="#5B8E7D",
        alpha=0.65,
    )
    axes[2].set_ylabel("Daily entry notional\n(萬 TWD)")
    axes[2].set_xlabel("Date")
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, dpi=150, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    return {
        "candidate_id": candidate,
        "dates": [dates[0], dates[-1]],
        "rows": len(subset),
        "calendar_rows": len(day),
        "zero_signal_days": int((day["trades"] == 0).sum()),
        "expected_net_pnl": expected,
        "backtest_net_pnl": actual,
        "canonical_daily_net_pnl": canonical_total,
        "canonical_calendar_sharpe": _calendar_sharpe(daily_pnl),
        "canonical_max_drawdown_initial_zero": float(np.min(drawdown)),
        "project_backtest_event_day_stats_diagnostic_only": _json_value(stats),
        "path": str(path.relative_to(STUDY_DIR)),
    }


def _save_comparison_plot(daily: pl.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for candidate in CANDIDATES:
        frame = daily.filter(pl.col("candidate_id") == candidate).sort("Date")
        dates = pd.to_datetime(frame["Date"].to_list())
        axes[0].plot(
            dates,
            frame["cumulative_net_pnl"].to_numpy() / 10_000.0,
            label=candidate,
            linewidth=1.4,
        )
        axes[1].plot(
            dates,
            frame["position"].to_numpy() / 10_000.0,
            label=candidate,
            linewidth=0.9,
            alpha=0.75,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Cumulative net PnL (萬 TWD)")
    axes[0].set_title(
        "Frozen signed-FillLots long family · 19.3bp · no winner selection",
        fontweight="bold",
    )
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].set_ylabel("Daily cumulative entry notional (萬 TWD)")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        fig.savefig(temporary, dpi=150, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)


def _build_family_bias_adjustment(
    daily: pl.DataFrame,
    short: pl.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    arrays = {
        candidate: daily.filter(pl.col("candidate_id") == candidate)
        .sort("Date")["net_pnl"]
        .to_numpy()
        for candidate in CANDIDATES
    }
    selection_contract = contract["selection_bias_accounting"]
    max_stat = moving_block_max_t(
        arrays,
        block_length=selection_contract["moving_block_length_sessions"],
        repetitions=selection_contract["bootstrap_repetitions"],
        seed=selection_contract["bootstrap_seed"],
        family_alpha=0.05,
    )
    block_sensitivity = {
        str(block_length): moving_block_max_t(
            arrays,
            block_length=int(block_length),
            repetitions=selection_contract[
                "block_sensitivity_repetitions"
            ],
            seed=selection_contract["bootstrap_seed"] + int(block_length),
            family_alpha=0.05,
        )
        for block_length in selection_contract[
            "block_length_sensitivity_sessions"
        ]
    }
    short_pnl = short.sort("Date")["net_pnl"].to_numpy()
    worst_count = max(1, math.ceil(0.10 * len(short_pnl)))
    worst_indices = np.argsort(short_pnl, kind="stable")[:worst_count]
    worst_mask = np.zeros(len(short_pnl), dtype=float)
    worst_mask[worst_indices] = 1.0
    tail_arrays = {
        candidate: arrays[candidate] * worst_mask for candidate in CANDIDATES
    }
    cover_tail_max_stat = moving_block_max_t(
        tail_arrays,
        block_length=selection_contract["moving_block_length_sessions"],
        repetitions=selection_contract["bootstrap_repetitions"],
        seed=selection_contract["bootstrap_seed"] + 10_000,
        family_alpha=0.05,
    )
    cover_tail_block_sensitivity = {
        str(block_length): moving_block_max_t(
            tail_arrays,
            block_length=int(block_length),
            repetitions=selection_contract[
                "block_sensitivity_repetitions"
            ],
            seed=(
                selection_contract["bootstrap_seed"]
                + 20_000
                + int(block_length)
            ),
            family_alpha=0.05,
        )
        for block_length in selection_contract[
            "block_length_sensitivity_sessions"
        ]
    }
    for candidate in CANDIDATES:
        cover_tail_max_stat["trials"][candidate][
            "conditional_mean_native_long_pnl_on_worst_short_days"
        ] = float(np.mean(arrays[candidate][worst_indices]))

    dependence_candidates = [
        candidate
        for candidate in CANDIDATES
        if np.std(arrays[candidate], ddof=1) > np.finfo(float).eps
    ]
    if len(dependence_candidates) == 1:
        correlation = np.ones((1, 1), dtype=float)
    elif dependence_candidates:
        pnl_matrix = np.column_stack(
            [arrays[candidate] for candidate in dependence_candidates]
        )
        correlation = np.corrcoef(pnl_matrix, rowvar=False)
        correlation = np.nan_to_num(
            correlation,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        np.fill_diagonal(correlation, 1.0)
    else:
        correlation = np.empty((0, 0), dtype=float)
    eigenvalues = (
        np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
        if dependence_candidates
        else np.asarray([], dtype=float)
    )
    effective_trials = (
        float(
            np.square(np.sum(eigenvalues))
            / np.sum(np.square(eigenvalues))
        )
        if len(eigenvalues)
        else None
    )
    annualized_sharpes = np.asarray(
        [_calendar_sharpe(arrays[candidate]) for candidate in CANDIDATES],
        dtype=float,
    )
    finite_sharpes = annualized_sharpes[np.isfinite(annualized_sharpes)]
    empirical_trial_std = (
        float(np.std(finite_sharpes, ddof=1))
        if len(finite_sharpes) > 1
        else None
    )
    trial_counts = selection_contract["dsr_like_trial_counts"]
    dsr: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        dsr[candidate] = {}
        for trial_count in trial_counts:
            try:
                result = deflated_sharpe_ratio(
                    arrays[candidate],
                    trial_count=int(trial_count),
                    trial_sharpe_std_annualized=empirical_trial_std,
                )
            except ValueError as error:
                result = {
                    "estimable": False,
                    "reason": str(error),
                    "dsr_probability": None,
                }
            else:
                result["estimable"] = True
            dsr[candidate][str(trial_count)] = result
    historical_exposure = json.loads(
        HISTORICAL_TRIAL_EXPOSURE_PATH.read_text(encoding="utf-8")
    )
    documented_lower_bound = int(
        historical_exposure["arithmetic_inventory_lower_bounds"]
        ["condition_selector_configurations_across_distinct_historical_families"]
    )
    if documented_lower_bound != int(
        selection_contract["historical_trial_lower_bound"]
    ):
        raise RuntimeError("historical trial exposure lower bound drifted")
    return _json_value(
        {
            "contract": "netfill_long_cover_family_bias/v2",
            "family_candidates": list(CANDIDATES),
            "current_family_trial_count": len(CANDIDATES),
            "historical_trial_lower_bound": selection_contract[
                "historical_trial_lower_bound"
            ],
            "moving_block_max_t": max_stat,
            "moving_block_max_t_block_sensitivity": block_sensitivity,
            "cover_tail_moving_block_max_t": cover_tail_max_stat,
            "cover_tail_moving_block_max_t_block_sensitivity": (
                cover_tail_block_sensitivity
            ),
            "cover_tail_definition": {
                "short_calendar_days": len(short_pnl),
                "worst_short_day_count": worst_count,
                "worst_short_dates": [
                    short.sort("Date")["Date"][int(index)]
                    for index in worst_indices
                ],
                "test_series": "native long daily PnL on fixed worst-short-decile dates and zero elsewhere",
            },
            "current_family_dependence": {
                "method": "eigenvalue_participation_ratio_of_daily_pnl_correlation",
                "raw_trial_count": len(CANDIDATES),
                "effective_trial_count": effective_trials,
                "estimable_candidate_order": dependence_candidates,
                "non_estimable_candidates": [
                    candidate
                    for candidate in CANDIDATES
                    if candidate not in dependence_candidates
                ],
                "correlation_matrix": correlation.tolist(),
                "eigenvalues": eigenvalues.tolist(),
            },
            "historical_trial_exposure_lineage": {
                "identity": _file_identity(HISTORICAL_TRIAL_EXPOSURE_PATH),
                "contract": historical_exposure["contract"],
                "selector_configuration_lower_bound": documented_lower_bound,
            },
            "dsr_like": {
                "empirical_annualized_sharpe_std_across_current_family": empirical_trial_std,
                "note": (
                    "DSR is secondary and approximate.  The 28-trial count is "
                    "a documented historical lower bound, while cross-trial "
                    "Sharpe dispersion is estimated only from the current six. "
                    "Neither DSR value is used as a hard acceptance gate."
                ),
                "candidates": dsr,
            },
        }
    )


def _load_short_daily(calendar: list[str]) -> tuple[pl.DataFrame, dict[str, Any]]:
    required_paths = (
        SHORT_DAILY_PATH,
        SHORT_MANIFEST_PATH,
        SHORT_VERIFICATION_PATH,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing short snapshot inputs: {missing}")
    manifest = json.loads(SHORT_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("contract") != "fh_neg_rejection_static_backtest/retrospective_v1":
        raise RuntimeError("unexpected short snapshot manifest contract")
    manifest_outputs = manifest.get("outputs", {})
    for path in (SHORT_DAILY_PATH, SHORT_VERIFICATION_PATH):
        matches = [
            identity
            for relative, identity in manifest_outputs.items()
            if relative.endswith(f"/{path.name}")
        ]
        if len(matches) != 1 or matches[0].get("sha256") != _sha256(path):
            raise RuntimeError(
                f"short snapshot hash is not bound by manifest: {path}"
            )
    short = pl.read_parquet(SHORT_DAILY_PATH).sort("Date")
    required_columns = {
        "Date",
        "net_pnl",
        "gross_pnl",
        "position",
        "trades",
        "stocks",
        "fee_bp",
    }
    missing_columns = sorted(required_columns - set(short.columns))
    if missing_columns:
        raise RuntimeError(f"short daily missing columns {missing_columns}")
    if len(short) != len(calendar):
        raise RuntimeError(
            f"short calendar length drift: {len(short)} != {len(calendar)}"
        )
    if short["Date"].to_list() != calendar:
        raise RuntimeError("short daily Date calendar is not exact")
    _assert_unique_keys(short, ("Date",), "short daily")
    if short["fee_bp"].unique().to_list() != [PRIMARY_FEE_BP]:
        raise RuntimeError("short daily fee is not the frozen 19.3bp")
    verification = json.loads(SHORT_VERIFICATION_PATH.read_text(encoding="utf-8"))
    if verification.get("all_passed") is not True:
        raise RuntimeError("short snapshot verification is not all_passed")
    if not verification.get("checks"):
        raise RuntimeError("short snapshot verification has no checks")
    failing = [
        key
        for key, value in verification.get("checks", {}).items()
        if value is not True
    ]
    if failing:
        raise RuntimeError(f"short snapshot verification has failures: {failing}")
    identities = {
        "daily": _file_identity(SHORT_DAILY_PATH),
        "manifest": _file_identity(SHORT_MANIFEST_PATH),
        "verification": _file_identity(SHORT_VERIFICATION_PATH),
    }
    return short, identities


def _build_cover_summary(
    daily: pl.DataFrame,
    short: pl.DataFrame,
    contract: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    short = short.sort("Date")
    short_pnl = short["net_pnl"].to_numpy()
    short_position = short["position"].to_numpy()
    cover_contract = contract["cover_evaluation"]
    multipliers = tuple(
        float(value)
        for value in cover_contract["fixed_native_long_multiplier_grid"]
    )
    anchor = float(cover_contract["anchor_native_multiplier"])
    if anchor not in multipliers:
        raise RuntimeError("cover anchor is absent from fixed multiplier grid")
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    short_loss = short_pnl < 0.0
    worst_count = max(1, math.ceil(0.10 * len(short_pnl)))
    worst_indices = np.argsort(short_pnl, kind="stable")[:worst_count]

    for candidate in CANDIDATES:
        long_day = daily.filter(pl.col("candidate_id") == candidate).sort("Date")
        if long_day["Date"].to_list() != short["Date"].to_list():
            raise RuntimeError(f"{candidate}: long/short calendar mismatch")
        long_pnl = long_day["net_pnl"].to_numpy()
        long_position = long_day["position"].to_numpy()
        if float(np.sum(long_position)) > 0.0:
            result = cover_portfolio_metrics(
                short_pnl,
                long_pnl,
                short_position=short_position,
                long_position=long_position,
                native_long_multipliers=multipliers,
                normalized_long_multipliers=multipliers,
            )
        else:
            short_metrics = calendar_pnl_metrics(short_pnl)
            zero_long_metrics = calendar_pnl_metrics(long_pnl)

            def zero_frontier(field: str) -> list[dict[str, Any]]:
                return [
                    {
                        field: multiplier,
                        **short_metrics,
                        "incremental_total_pnl_vs_short": 0.0,
                        "calendar_sharpe_delta_vs_short": 0.0,
                        "max_drawdown_delta_vs_short": 0.0,
                        "expected_shortfall_5pct_delta_vs_short": 0.0,
                    }
                    for multiplier in multipliers
                ]

            result = {
                "capital_normalization_method": (
                    "unavailable_zero_long_aggregate_position"
                ),
                "short_position_total": float(np.sum(short_position)),
                "long_position_total": 0.0,
                "long_position_normalization_scale": None,
                "primary_frontier_definition": (
                    "short_pnl_plus_h_times_native_long_pnl"
                ),
                "position_normalization_role": "unavailable_zero_support",
                "short": short_metrics,
                "long_raw": zero_long_metrics,
                "long_normalized_to_short_position": None,
                "native_cover_diagnostics": {
                    "pearson_net_daily_pnl": math.nan,
                    "spearman_net_daily_pnl": math.nan,
                    "downside_beta_long_on_short_loss_days": math.nan,
                    "short_loss_days": int(np.count_nonzero(short_loss)),
                    "mean_native_long_pnl_on_short_loss_days": 0.0,
                    "native_long_positive_rate_on_short_loss_days": 0.0,
                    "worst_short_days": int(worst_count),
                    "mean_native_long_pnl_on_worst_short_days": 0.0,
                    "native_long_positive_rate_on_worst_short_days": 0.0,
                },
                "position_normalized_cover_diagnostics": None,
                "native_cash_frontier": zero_frontier(
                    "native_long_multiplier_h"
                ),
                "position_normalized_frontier": zero_frontier(
                    "normalized_long_multiplier_h"
                ),
            }
        details[candidate] = _json_value(result)
        diagnostics = result["native_cover_diagnostics"]
        normalization_value = result["long_position_normalization_scale"]
        normalization = (
            None
            if normalization_value is None
            else float(normalization_value)
        )
        for mode, frontier, multiplier_field, scale in (
            (
                "native_cash_add_on_primary",
                result["native_cash_frontier"],
                "native_long_multiplier_h",
                1.0,
            ),
            (
                "position_normalized_secondary",
                result["position_normalized_frontier"],
                "normalized_long_multiplier_h",
                normalization,
            ),
        ):
            for portfolio in frontier:
                multiplier = float(portfolio[multiplier_field])
                effective_scale = 0.0 if scale is None else scale
                combined = short_pnl + multiplier * effective_scale * long_pnl
                rows.append(
                    {
                        "candidate_id": candidate,
                        "candidate_family": CANDIDATE_FAMILY[candidate],
                        "accounting_mode": mode,
                        "long_multiplier_h": multiplier,
                        "long_pnl_scale": scale,
                        "calendar_days": int(portfolio["calendar_days"]),
                        "total_pnl": float(portfolio["total_pnl"]),
                        "mean_daily_pnl": float(portfolio["mean_daily_pnl"]),
                        "calendar_sharpe": float(portfolio["calendar_sharpe"]),
                        "max_drawdown": float(portfolio["max_drawdown"]),
                        "expected_shortfall_5pct": float(
                            portfolio["expected_shortfall_5pct"]
                        ),
                        "incremental_total_pnl_vs_short": float(
                            portfolio["incremental_total_pnl_vs_short"]
                        ),
                        "calendar_sharpe_delta_vs_short": float(
                            portfolio["calendar_sharpe_delta_vs_short"]
                        ),
                        "max_drawdown_delta_vs_short": float(
                            portfolio["max_drawdown_delta_vs_short"]
                        ),
                        "expected_shortfall_5pct_delta_vs_short": float(
                            portfolio[
                                "expected_shortfall_5pct_delta_vs_short"
                            ]
                        ),
                        "pearson_net_daily_pnl": float(
                            diagnostics["pearson_net_daily_pnl"]
                        ),
                        "spearman_net_daily_pnl": float(
                            diagnostics["spearman_net_daily_pnl"]
                        ),
                        "native_downside_beta_long_on_short_loss_days": float(
                            diagnostics[
                                "downside_beta_long_on_short_loss_days"
                            ]
                        ),
                        "effective_scaled_downside_beta_on_short_loss_days": (
                            float(
                                diagnostics[
                                    "downside_beta_long_on_short_loss_days"
                                ]
                            )
                            * effective_scale
                            * multiplier
                        ),
                        "short_loss_days": int(diagnostics["short_loss_days"]),
                        "mean_native_long_pnl_on_short_loss_days": float(
                            diagnostics[
                                "mean_native_long_pnl_on_short_loss_days"
                            ]
                        ),
                        "native_long_positive_rate_on_short_loss_days": float(
                            diagnostics[
                                "native_long_positive_rate_on_short_loss_days"
                            ]
                        ),
                        "worst_short_days": int(
                            diagnostics["worst_short_days"]
                        ),
                        "mean_native_long_pnl_on_worst_short_days": float(
                            diagnostics[
                                "mean_native_long_pnl_on_worst_short_days"
                            ]
                        ),
                        "native_long_positive_rate_on_worst_short_days": float(
                            diagnostics[
                                "native_long_positive_rate_on_worst_short_days"
                            ]
                        ),
                        "mean_combined_pnl_on_short_loss_days": float(
                            np.mean(combined[short_loss])
                        ),
                        "mean_combined_pnl_on_worst_short_days": float(
                            np.mean(combined[worst_indices])
                        ),
                        "rescued_short_loss_days": int(
                            np.count_nonzero(short_loss & (combined >= 0.0))
                        ),
                        "new_loss_days": int(
                            np.count_nonzero((short_pnl >= 0.0) & (combined < 0.0))
                        ),
                        "is_anchor": bool(
                            mode == "native_cash_add_on_primary"
                            and np.isclose(multiplier, anchor)
                        ),
                    }
                )
    return pl.DataFrame(rows).sort(
        "candidate_id", "accounting_mode", "long_multiplier_h"
    ), details


def _save_cover_plot(
    daily: pl.DataFrame,
    short: pl.DataFrame,
    *,
    anchor: float,
    path: Path,
) -> None:
    short = short.sort("Date")
    dates = pd.to_datetime(short["Date"].to_list())
    short_pnl = short["net_pnl"].to_numpy()
    short_cumulative = np.cumsum(short_pnl)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(
        dates,
        short_cumulative / 10_000.0,
        color="black",
        linewidth=2.2,
        label="fh_neg_rejection short only",
    )
    axes[1].plot(
        dates,
        short_pnl / 10_000.0,
        color="black",
        linewidth=0.8,
        alpha=0.65,
        label="short daily",
    )
    for candidate in CANDIDATES:
        long_pnl = (
            daily.filter(pl.col("candidate_id") == candidate)
            .sort("Date")["net_pnl"]
            .to_numpy()
        )
        combined = short_pnl + anchor * long_pnl
        axes[0].plot(
            dates,
            np.cumsum(combined) / 10_000.0,
            linewidth=1.2,
            label=f"short + {anchor:g}×{candidate}",
        )
        axes[1].plot(
            dates,
            anchor * long_pnl / 10_000.0,
            linewidth=0.7,
            alpha=0.55,
            label=candidate,
        )
    axes[0].axhline(0.0, color="#777777", linewidth=0.8)
    axes[0].set_ylabel("Cumulative net PnL (萬 TWD)")
    axes[0].set_title(
        f"Native cash cover frontier anchor h={anchor:g} · 19.3bp both sleeves",
        fontweight="bold",
    )
    axes[0].legend(ncol=2, fontsize=7)
    axes[0].grid(alpha=0.25)
    axes[1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[1].set_ylabel("Long add-on daily net PnL (萬 TWD)")
    axes[1].set_xlabel("Date")
    axes[1].legend(ncol=3, fontsize=7)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        fig.savefig(temporary, dpi=150, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)


def _classify_decision(
    qualifying_standalone: Sequence[str],
    qualifying_cover: Sequence[str],
) -> tuple[str, list[str]]:
    """Classify gates without treating different candidates as one strategy."""
    qualifying_both = sorted(
        set(qualifying_standalone).intersection(qualifying_cover)
    )
    if qualifying_both:
        decision = "RETROSPECTIVE_JOINT_LONG_COVER_SCREEN_PASSED"
    elif qualifying_standalone and qualifying_cover:
        decision = "RETROSPECTIVE_STANDALONE_AND_COVER_SCREENS_ARE_DISJOINT"
    elif qualifying_standalone:
        decision = "RETROSPECTIVE_STANDALONE_SCREEN_ONLY"
    elif qualifying_cover:
        decision = "RETROSPECTIVE_COVER_SCREEN_ONLY"
    else:
        decision = "NO_RETROSPECTIVE_STANDALONE_OR_COVER_SCREEN_PASS"
    return decision, qualifying_both


def _build_decision(
    period_summary: pl.DataFrame,
    family_bias: dict[str, Any],
    cover_summary: pl.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    break_even_candidates: list[str] = []
    qualifying_standalone: list[str] = []
    qualifying_cover: list[str] = []
    tail_help_but_unprofitable: list[str] = []
    anchor = float(contract["cover_evaluation"]["anchor_native_multiplier"])
    max_trials = max(contract["selection_bias_accounting"]["dsr_like_trial_counts"])
    for candidate in CANDIDATES:
        rows = period_summary.filter(pl.col("candidate_id") == candidate)
        full = rows.filter(pl.col("period") == "full_retrospective").row(
            0, named=True
        )
        year_2025 = rows.filter(pl.col("period") == "year_2025").row(
            0, named=True
        )
        year_2026 = rows.filter(
            pl.col("period") == "year_2026_historical"
        ).row(0, named=True)
        max_stat = family_bias["moving_block_max_t"]["trials"][candidate]
        dsr = family_bias["dsr_like"]["candidates"][candidate][str(max_trials)]
        anchor_row = cover_summary.filter(
            (pl.col("candidate_id") == candidate)
            & (pl.col("accounting_mode") == "native_cash_add_on_primary")
            & pl.col("long_multiplier_h").is_close(anchor)
        ).row(0, named=True)
        break_even = float(full["cap_weighted_net_bp"]) >= 0.0
        cross_year = (
            float(year_2025["cap_weighted_net_bp"]) >= 0.0
            and float(year_2026["cap_weighted_net_bp"]) >= 0.0
        )
        current_family_bias_screen = (
            float(max_stat["max_t_adjusted_p"]) <= 0.10
        )
        standalone_pass = (
            break_even and cross_year and current_family_bias_screen
        )
        cover_tail_max_stat = family_bias[
            "cover_tail_moving_block_max_t"
        ]["trials"][candidate]
        descriptive_cover_screen = (
            float(anchor_row["mean_native_long_pnl_on_worst_short_days"])
            > 0.0
            and float(anchor_row["pearson_net_daily_pnl"]) < 0.0
            and float(anchor_row["spearman_net_daily_pnl"]) < 0.0
            and float(anchor_row["calendar_sharpe_delta_vs_short"]) >= 0.0
            and float(anchor_row["max_drawdown_delta_vs_short"]) >= 0.0
        )
        cover_gate = (
            descriptive_cover_screen
            and float(cover_tail_max_stat["max_t_adjusted_p"]) <= 0.10
        )
        if break_even:
            break_even_candidates.append(candidate)
        if standalone_pass:
            qualifying_standalone.append(candidate)
        if cover_gate:
            qualifying_cover.append(candidate)
        if cover_gate and not break_even:
            tail_help_but_unprofitable.append(candidate)
        candidates[candidate] = {
            "full_cap_weighted_net_bp": float(full["cap_weighted_net_bp"]),
            "year_2025_net_bp": float(year_2025["cap_weighted_net_bp"]),
            "year_2026_net_bp": float(year_2026["cap_weighted_net_bp"]),
            "max_t_adjusted_p": float(max_stat["max_t_adjusted_p"]),
            "cover_tail_max_t_adjusted_p": float(
                cover_tail_max_stat["max_t_adjusted_p"]
            ),
            f"dsr_like_probability_{max_trials}_trials": (
                None
                if dsr["dsr_probability"] is None
                else float(dsr["dsr_probability"])
            ),
            "break_even": break_even,
            "cross_year_nonnegative": cross_year,
            "current_six_family_max_t_screen": current_family_bias_screen,
            "dsr_is_diagnostic_only": True,
            "standalone_research_screen": standalone_pass,
            "descriptive_native_anchor_cover_screen": (
                descriptive_cover_screen
            ),
            "family_adjusted_native_anchor_cover_screen": cover_gate,
            "parameter_plateau_status": (
                "NOT_EVALUATED"
                if candidate in HIGH_CANDIDATES
                else "LOW_THRESHOLD_LADDER_REPORTED"
            ),
            "anchor_metrics": _json_value(anchor_row),
        }
    decision, qualifying_both = _classify_decision(
        qualifying_standalone,
        qualifying_cover,
    )
    return {
        "contract": "netfill_long_cover_decision/v2",
        "decision": decision,
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "promotion_authorized": False,
        "deployment_handoff_authorized": False,
        "break_even_candidates": break_even_candidates,
        "standalone_research_screen_candidates": qualifying_standalone,
        "cover_research_screen_candidates": qualifying_cover,
        "joint_research_screen_candidates": qualifying_both,
        "tail_help_but_unprofitable_candidates": tail_help_but_unprofitable,
        "winner_selected": False,
        "selection_bias_claim": (
            "Current-six standalone and fixed-tail screens are max-t adjusted; "
            "historical search exposure and the multi-parameter fh_pos_breakout "
            "plateau are not fully controlled."
        ),
        "candidate_results": candidates,
        "limitations": [
            "retrospective and historically outcome-informed",
            "no joint portfolio cap, inventory netting, latency, impact, or fill model",
            "native cover multiplier is cash accounting, not an equal-capital allocation",
            "position-normalized frontier is secondary because position is cumulative entry notional",
        ],
    }


def _build_trial_ledger(
    counts: pl.DataFrame,
    daily: pl.DataFrame,
    period_summary: pl.DataFrame,
    family_bias: dict[str, Any],
    decision: dict[str, Any],
    contract: dict[str, Any],
    *,
    materialization_identity_sha256: str,
) -> pl.DataFrame:
    definitions = {
        item["candidate_id"]: item["definition"]
        for section in (
            "at_high_nested_ablation",
            "at_low_prior_depth_neighborhood",
        )
        for item in contract["candidate_family"][section]
    }
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        profile_path = (
            HIGH_PROFILE_PATH if candidate in HIGH_CANDIDATES else LOW_PROFILE_PATH
        )
        selector_identity = {
            "candidate_id": candidate,
            "definition": definitions[candidate],
            "profile_path": str(profile_path.relative_to(PROJECT_ROOT)),
            "profile_sha256": _sha256(profile_path),
        }
        selector_fingerprint = hashlib.sha256(
            json.dumps(
                selector_identity,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        count = counts.filter(pl.col("candidate_id") == candidate)
        day = daily.filter(pl.col("candidate_id") == candidate)
        periods = period_summary.filter(pl.col("candidate_id") == candidate)
        full = periods.filter(pl.col("period") == "full_retrospective").row(
            0, named=True
        )
        year_2025 = periods.filter(pl.col("period") == "year_2025").row(
            0, named=True
        )
        year_2026 = periods.filter(
            pl.col("period") == "year_2026_historical"
        ).row(0, named=True)
        result = decision["candidate_results"][candidate]
        rows.append(
            {
                "trial_id": candidate,
                "candidate_family": CANDIDATE_FAMILY[candidate],
                "objective": contract["objective"],
                "selector_definition": definitions[candidate],
                "selector_profile_path": selector_identity["profile_path"],
                "selector_profile_sha256": selector_identity["profile_sha256"],
                "selector_fingerprint": selector_fingerprint,
                "pool_label_execution_fingerprint": (
                    materialization_identity_sha256
                ),
                "indexed_rows": int(count["indexed_rows"].sum()),
                "eligible_rows": int(count["eligible_rows"].sum()),
                "raw_labeled_rows": int(count["labeled_rows"].sum()),
                "diagnostic_label_valid_rows": int(
                    count["diagnostic_label_valid_rows"].sum()
                ),
                "pre_capacity_rows": int(count["pre_capacity_rows"].sum()),
                "accepted_rows": int(count["accepted_rows"].sum()),
                "active_days": int((day["trades"] > 0).sum()),
                "zero_signal_days": int((day["trades"] == 0).sum()),
                "full_net_bp": float(full["cap_weighted_net_bp"]),
                "year_2025_net_bp": float(year_2025["cap_weighted_net_bp"]),
                "year_2026_net_bp": float(year_2026["cap_weighted_net_bp"]),
                "standalone_max_t_adjusted_p": float(
                    family_bias["moving_block_max_t"]["trials"][candidate]
                    ["max_t_adjusted_p"]
                ),
                "cover_tail_max_t_adjusted_p": float(
                    family_bias["cover_tail_moving_block_max_t"]["trials"]
                    [candidate]["max_t_adjusted_p"]
                ),
                "break_even": bool(result["break_even"]),
                "standalone_research_screen": bool(
                    result["standalone_research_screen"]
                ),
                "cover_research_screen": bool(
                    result["family_adjusted_native_anchor_cover_screen"]
                ),
                "parameter_plateau_status": result[
                    "parameter_plateau_status"
                ],
                "selected": False,
                "promotion_authorized": False,
            }
        )
    return pl.DataFrame(rows).sort("trial_id")


def _daily_partition_paths(directory: Path, calendar: list[str]) -> list[Path]:
    paths = [directory / f"{date}.parquet" for date in calendar]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} daily partitions; sample={missing[:5]}"
        )
    return paths


def _materialize_all(
    calendar: list[str],
    *,
    contract_sha256: str,
    materialization_identity_sha256: str,
    materialization_identity_json: str,
) -> None:
    if RUN_MANIFEST_PATH.exists():
        raise FileExistsError(
            f"kept artifact already finalized: {RUN_MANIFEST_PATH}; "
            "daily partitions are immutable"
        )
    roots = DataRoots.from_project_root(PROJECT_ROOT)
    txf_loader = TxfDataLoader(roots.txf_tick_dir)
    DAILY_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    DAILY_COUNT_DIR.mkdir(parents=True, exist_ok=True)
    for index, date in enumerate(calendar, start=1):
        day = EventDaySources(date, roots)
        source_json = _source_identities(day)
        signal_path = DAILY_SIGNAL_DIR / f"{date}.parquet"
        count_path = DAILY_COUNT_DIR / f"{date}.parquet"
        if signal_path.exists() or count_path.exists():
            if _cached_day_valid(
                signal_path,
                count_path,
                date=date,
                contract_sha256=contract_sha256,
                materialization_identity_sha256=(
                    materialization_identity_sha256
                ),
                source_identities_json=source_json,
            ):
                print(f"[{index}/{len(calendar)}] {date}: cached", flush=True)
                continue
            raise RuntimeError(
                f"{date}: stale or partial daily cache exists; preserve this "
                "artifact version and use a new output version"
            )
        accepted, counts = _materialize_day(
            date,
            roots=roots,
            txf_loader=txf_loader,
            contract_sha256=contract_sha256,
            materialization_identity_sha256=(
                materialization_identity_sha256
            ),
            materialization_identity_json=materialization_identity_json,
            source_identities_json=source_json,
        )
        _atomic_parquet(accepted, signal_path)
        _atomic_parquet(counts, count_path)
        print(
            f"[{index}/{len(calendar)}] {date}: "
            f"indexed={int(counts['indexed_rows'].sum()):,} "
            f"labeled={int(counts['labeled_rows'].sum()):,} "
            f"accepted={len(accepted):,}",
            flush=True,
        )


def _verify_outputs(
    *,
    accepted: pl.DataFrame,
    counts: pl.DataFrame,
    daily: pl.DataFrame,
    period_summary: pl.DataFrame,
    cover_summary: pl.DataFrame,
    family_bias: dict[str, Any],
    decision: dict[str, Any],
    trial_ledger: pl.DataFrame,
    short: pl.DataFrame,
    contract: dict[str, Any],
    materialization_identity_sha256: str,
    calendar: list[str],
    plot_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    _assert_unique_keys(
        accepted,
        ("candidate_id", *KEY_COLUMNS),
        "combined accepted signals",
    )
    _assert_unique_keys(counts, ("Date", "candidate_id"), "daily counts")
    _assert_unique_keys(daily, ("Date", "candidate_id"), "daily PnL")
    _assert_unique_keys(trial_ledger, ("trial_id",), "trial ledger")

    expected_keys = {
        (date, candidate) for date in calendar for candidate in CANDIDATES
    }
    count_keys = set(counts.select("Date", "candidate_id").iter_rows())
    daily_keys = set(daily.select("Date", "candidate_id").iter_rows())
    accepted_by_key = {
        (row["Date"], row["candidate_id"]): int(row["len"])
        for row in accepted.group_by("Date", "candidate_id")
        .len()
        .iter_rows(named=True)
    }
    count_accepted_by_key = {
        (row["Date"], row["candidate_id"]): int(row["accepted_rows"])
        for row in counts.select(
            "Date", "candidate_id", "accepted_rows"
        ).iter_rows(named=True)
    }
    daily_trades_by_key = {
        (row["Date"], row["candidate_id"]): int(row["trades"])
        for row in daily.select("Date", "candidate_id", "trades").iter_rows(
            named=True
        )
    }
    accepted_count_reconciled = all(
        accepted_by_key.get(key, 0) == count_accepted_by_key.get(key)
        for key in expected_keys
    )
    daily_trade_reconciled = all(
        accepted_by_key.get(key, 0) == daily_trades_by_key.get(key)
        for key in expected_keys
    )

    funnel_valid = all(
        int(row["indexed_rows"]) >= int(row["eligible_rows"])
        >= int(row["labeled_rows"])
        >= int(row["pre_capacity_rows"])
        >= int(row["accepted_rows"])
        and int(row["eligible_rows"])
        >= int(row["diagnostic_label_valid_rows"])
        for row in counts.iter_rows(named=True)
    )
    source_identity_one_per_day = all(
        int(row["n"] or 0) == 1
        for row in counts.group_by("Date")
        .agg(pl.col("source_identities_json").n_unique().alias("n"))
        .iter_rows(named=True)
    )
    identity_json_values = counts[
        "materialization_identity_json"
    ].unique().to_list()
    identity_json_hashes_match = all(
        hashlib.sha256(value.encode("utf-8")).hexdigest()
        == materialization_identity_sha256
        and json.loads(value)["contract_sha256"] == _sha256(CONTRACT_PATH)
        for value in identity_json_values
    )
    accepted_identity_values = set(
        accepted["materialization_identity_sha256"].unique().to_list()
    )
    accepted_contract_values = set(
        accepted["contract_sha256"].unique().to_list()
    )

    if accepted.is_empty():
        label_exit_parity = True
        label_raw_parity = True
        expected_net = 0.0
        expected_daily_map: dict[tuple[str, str], dict[str, float | int]] = {}
    else:
        expected_exit = np.where(
            accepted["FutureHigh"].to_numpy()
            > accepted["RefPrice"].to_numpy() * 1.09,
            accepted["RefPrice"].to_numpy() * 1.09,
            accepted["Close"].to_numpy(),
        )
        expected_raw = (
            (expected_exit - accepted["AskPrice1"].to_numpy())
            / accepted["AskPrice1"].to_numpy()
            * 10_000.0
        )
        label_exit_parity = bool(
            np.allclose(
                accepted["label_exit_price"].to_numpy(),
                expected_exit,
                rtol=1e-12,
                atol=1e-9,
            )
        )
        label_raw_parity = bool(
            np.allclose(
                accepted["label_raw_bp"].to_numpy(),
                expected_raw,
                rtol=1e-12,
                atol=1e-8,
            )
        )
        expected_net = float(
            accepted.select(
                (
                    pl.col("AskPrice1")
                    * (pl.col("label_raw_bp") - PRIMARY_FEE_BP)
                    / 10.0
                ).sum()
            ).item()
        )
        expected_daily = (
            accepted.with_columns(
                (
                    pl.col("AskPrice1")
                    * (pl.col("label_raw_bp") - PRIMARY_FEE_BP)
                    / 10.0
                ).alias("_net"),
                pl.when(pl.col("label_residual_bp").is_finite())
                .then(
                    pl.col("AskPrice1")
                    * pl.col("label_residual_bp")
                    / 10.0
                )
                .otherwise(0.0)
                .alias("_residual_gross"),
                pl.when(pl.col("label_residual_bp").is_finite())
                .then(
                    pl.col("AskPrice1")
                    * (pl.col("label_residual_bp") - PRIMARY_FEE_BP)
                    / 10.0
                )
                .otherwise(0.0)
                .alias("_residual_net"),
                pl.when(pl.col("label_residual_bp").is_finite())
                .then(pl.col("AskPrice1") * 1000.0)
                .otherwise(0.0)
                .alias("_residual_position"),
                pl.col("label_residual_bp")
                .is_finite()
                .fill_null(False)
                .cast(pl.Int64)
                .alias("_residual_trades"),
            )
            .group_by("Date", "candidate_id")
            .agg(
                pl.col("_net").sum().alias("net_pnl"),
                pl.col("_residual_gross").sum().alias(
                    "residual_gross_pnl"
                ),
                pl.col("_residual_net").sum().alias("residual_net_pnl"),
                pl.col("_residual_position").sum().alias("residual_position"),
                pl.col("_residual_trades").sum().alias("residual_trades"),
            )
        )
        expected_daily_map = {
            (row["Date"], row["candidate_id"]): row
            for row in expected_daily.iter_rows(named=True)
        }

    def close(left: Any, right: Any, *, atol: float = 1e-6) -> bool:
        if left is None or right is None:
            return left is None and right is None
        return bool(
            np.isclose(
                float(left),
                float(right),
                rtol=1e-12,
                atol=atol,
                equal_nan=True,
            )
        )

    daily_pnl_reconciled = True
    residual_support_reconciled = True
    for row in daily.iter_rows(named=True):
        key = (row["Date"], row["candidate_id"])
        expected = expected_daily_map.get(key)
        daily_pnl_reconciled &= close(
            row["net_pnl"], 0.0 if expected is None else expected["net_pnl"]
        )
        residual_support_reconciled &= all(
            (
                int(row["residual_trades"])
                == (0 if expected is None else int(expected["residual_trades"])),
                close(
                    row["residual_position"],
                    0.0 if expected is None else expected["residual_position"],
                ),
                close(
                    row["residual_gross_pnl"],
                    0.0
                    if expected is None
                    else expected["residual_gross_pnl"],
                ),
                close(
                    row["residual_net_pnl"],
                    0.0 if expected is None else expected["residual_net_pnl"],
                ),
                int(row["residual_trades"]) <= int(row["trades"]),
                float(row["residual_position"]) <= float(row["position"])
                + 1e-6,
            )
        )

    daily_net = float(daily["net_pnl"].sum())
    period_full_net = float(
        period_summary.filter(pl.col("period") == "full_retrospective")[
            "net_pnl"
        ].sum()
    )
    candidate_period_parity = all(
        close(
            daily.filter(pl.col("candidate_id") == candidate)["net_pnl"].sum(),
            period_summary.filter(
                (pl.col("candidate_id") == candidate)
                & (pl.col("period") == "full_retrospective")
            )["net_pnl"].item(),
        )
        for candidate in CANDIDATES
    )
    residual_period_denominator_parity = True
    for candidate in CANDIDATES:
        candidate_day = daily.filter(pl.col("candidate_id") == candidate)
        full_row = period_summary.filter(
            (pl.col("candidate_id") == candidate)
            & (pl.col("period") == "full_retrospective")
        ).row(0, named=True)
        residual_position = float(candidate_day["residual_position"].sum())
        residual_net = float(candidate_day["residual_net_pnl"].sum())
        expected_residual_bp = (
            residual_net / residual_position * 10_000.0
            if residual_position > 0.0
            else math.nan
        )
        residual_period_denominator_parity &= all(
            (
                close(full_row["residual_position"], residual_position),
                close(full_row["residual_net_pnl"], residual_net),
                close(
                    full_row["cap_weighted_residual_net_bp"],
                    expected_residual_bp,
                ),
            )
        )

    cover_grid = sorted(
        float(value)
        for value in contract["cover_evaluation"][
            "fixed_native_long_multiplier_grid"
        ]
    )
    anchor = float(contract["cover_evaluation"]["anchor_native_multiplier"])
    short_day = short.sort("Date")
    short_pnl = short_day["net_pnl"].to_numpy()
    short_position = short_day["position"].to_numpy()
    short_sharpe = _calendar_sharpe(short_pnl)
    _, _, short_drawdown = _drawdown(short_pnl)
    short_mdd = float(np.min(short_drawdown))
    short_tail_count = max(1, math.ceil(0.05 * len(short_pnl)))
    short_es = float(np.mean(np.sort(short_pnl)[:short_tail_count]))
    cover_formula_parity = True
    cover_grid_exact = True
    cover_anchor_exact = True
    primary_scale_exact = True
    h_zero_short_parity = True
    decision_uses_primary_only = True
    for candidate in CANDIDATES:
        long_day = daily.filter(pl.col("candidate_id") == candidate).sort("Date")
        long_pnl = long_day["net_pnl"].to_numpy()
        long_position_total = float(long_day["position"].sum())
        normalized_scale = (
            float(np.sum(short_position)) / long_position_total
            if long_position_total > 0.0
            else math.nan
        )
        candidate_rows = cover_summary.filter(
            pl.col("candidate_id") == candidate
        )
        for mode in (
            "native_cash_add_on_primary",
            "position_normalized_secondary",
        ):
            mode_rows = candidate_rows.filter(
                pl.col("accounting_mode") == mode
            ).sort("long_multiplier_h")
            cover_grid_exact &= (
                [float(value) for value in mode_rows["long_multiplier_h"]]
                == cover_grid
            )
            for row in mode_rows.iter_rows(named=True):
                multiplier = float(row["long_multiplier_h"])
                scale = 1.0 if mode == "native_cash_add_on_primary" else normalized_scale
                if not math.isfinite(scale):
                    cover_formula_parity &= row["long_pnl_scale"] is None
                    continue
                primary_scale_exact &= (
                    mode != "native_cash_add_on_primary"
                    or close(row["long_pnl_scale"], 1.0)
                )
                cover_formula_parity &= close(row["long_pnl_scale"], scale)
                combined = short_pnl + multiplier * scale * long_pnl
                _, _, combined_drawdown = _drawdown(combined)
                expected_mdd = float(np.min(combined_drawdown))
                expected_es = float(
                    np.mean(np.sort(combined)[:short_tail_count])
                )
                expected_metrics = {
                    "total_pnl": float(np.sum(combined)),
                    "mean_daily_pnl": float(np.mean(combined)),
                    "calendar_sharpe": _calendar_sharpe(combined),
                    "max_drawdown": expected_mdd,
                    "expected_shortfall_5pct": expected_es,
                    "incremental_total_pnl_vs_short": float(
                        np.sum(combined) - np.sum(short_pnl)
                    ),
                    "calendar_sharpe_delta_vs_short": (
                        _calendar_sharpe(combined) - short_sharpe
                    ),
                    "max_drawdown_delta_vs_short": expected_mdd - short_mdd,
                    "expected_shortfall_5pct_delta_vs_short": (
                        expected_es - short_es
                    ),
                }
                cover_formula_parity &= all(
                    close(row[column], value)
                    for column, value in expected_metrics.items()
                )
                if np.isclose(multiplier, 0.0):
                    h_zero_short_parity &= all(
                        close(row[column], value)
                        for column, value in {
                            "total_pnl": float(np.sum(short_pnl)),
                            "calendar_sharpe": short_sharpe,
                            "max_drawdown": short_mdd,
                            "expected_shortfall_5pct": short_es,
                        }.items()
                    )
        anchor_rows = candidate_rows.filter(pl.col("is_anchor") == True)
        cover_anchor_exact &= (
            len(anchor_rows) == 1
            and anchor_rows["accounting_mode"].item()
            == "native_cash_add_on_primary"
            and close(anchor_rows["long_multiplier_h"].item(), anchor)
        )
        decision_anchor = decision["candidate_results"][candidate][
            "anchor_metrics"
        ]
        decision_uses_primary_only &= (
            decision_anchor["accounting_mode"]
            == "native_cash_add_on_primary"
            and close(decision_anchor["long_pnl_scale"], 1.0)
            and close(decision_anchor["long_multiplier_h"], anchor)
        )

    trial_support_parity = all(
        int(row["accepted_rows"])
        == int(
            counts.filter(pl.col("candidate_id") == row["trial_id"])[
                "accepted_rows"
            ].sum()
        )
        for row in trial_ledger.iter_rows(named=True)
    )
    checks = {
        "calendar_count_exact": len(calendar) == 377,
        "daily_counts_keys_dense_exact": count_keys == expected_keys,
        "daily_keys_dense_exact": daily_keys == expected_keys,
        "period_rows_exact": len(period_summary) == len(CANDIDATES) * len(PERIODS),
        "cover_rows_exact": len(cover_summary)
        == len(CANDIDATES)
        * 2
        * len(
            contract["cover_evaluation"]["fixed_native_long_multiplier_grid"]
        ),
        "all_candidates_retained_in_counts_daily_ledger": (
            set(counts["candidate_id"].unique().to_list()) == set(CANDIDATES)
            and set(daily["candidate_id"].unique().to_list()) == set(CANDIDATES)
            and set(trial_ledger["trial_id"].to_list()) == set(CANDIDATES)
            and set(accepted["candidate_id"].unique().to_list()).issubset(CANDIDATES)
        ),
        "accepted_counts_reconcile_by_day_candidate": accepted_count_reconciled,
        "daily_trades_reconcile_by_day_candidate": daily_trade_reconciled,
        "daily_net_pnl_reconciles_by_day_candidate": daily_pnl_reconciled,
        "count_funnel_is_monotone": funnel_valid,
        "one_source_identity_per_day": source_identity_one_per_day,
        "materialization_identity_json_hash_exact": identity_json_hashes_match,
        "materialization_identity_columns_exact": (
            counts["materialization_identity_sha256"].unique().to_list()
            == [materialization_identity_sha256]
            and accepted_identity_values.issubset(
                {materialization_identity_sha256}
            )
        ),
        "contract_sha256_columns_exact": (
            counts["contract_sha256"].unique().to_list()
            == [_sha256(CONTRACT_PATH)]
            and accepted_contract_values.issubset({_sha256(CONTRACT_PATH)})
        ),
        "long_exit_formula_parity": label_exit_parity,
        "long_raw_formula_parity": label_raw_parity,
        "raw_label_is_finite_and_bounded": (
            accepted.is_empty()
            or bool(
                accepted.select(
                    (
                        pl.col("label_raw_bp").is_finite()
                        & (
                            pl.col("label_raw_bp").abs()
                            < RAW_LABEL_ABS_MAX_BP
                        )
                    ).all()
                ).item()
            )
        ),
        "residual_diagnostic_support_reconciled": residual_support_reconciled,
        "residual_period_uses_own_position_denominator": (
            residual_period_denominator_parity
        ),
        "accepted_to_daily_net_pnl_parity": bool(
            np.isclose(expected_net, daily_net, rtol=1e-12, atol=1e-6)
        ),
        "daily_to_period_net_pnl_parity": bool(
            np.isclose(daily_net, period_full_net, rtol=1e-12, atol=1e-6)
        ),
        "candidate_daily_to_period_net_pnl_parity": candidate_period_parity,
        "max_stat_six_trials": family_bias["moving_block_max_t"]["trial_count"]
        == len(CANDIDATES),
        "cover_tail_max_stat_six_trials": family_bias[
            "cover_tail_moving_block_max_t"
        ]["trial_count"]
        == len(CANDIDATES),
        "block_length_sensitivity_complete": (
            set(family_bias["moving_block_max_t_block_sensitivity"])
            == {
                str(value)
                for value in contract["selection_bias_accounting"]
                ["block_length_sensitivity_sessions"]
            }
            and set(
                family_bias[
                    "cover_tail_moving_block_max_t_block_sensitivity"
                ]
            )
            == {
                str(value)
                for value in contract["selection_bias_accounting"]
                ["block_length_sensitivity_sessions"]
            }
        ),
        "historical_trial_lineage_hash_exact": (
            family_bias["historical_trial_exposure_lineage"]["identity"]
            ["sha256"]
            == _sha256(HISTORICAL_TRIAL_EXPOSURE_PATH)
        ),
        "trial_ledger_support_parity": trial_support_parity,
        "trial_ledger_never_selects_winner": not bool(
            trial_ledger["selected"].any()
        ),
        "high_parameter_plateau_explicitly_not_evaluated": set(
            trial_ledger.filter(pl.col("trial_id").is_in(HIGH_CANDIDATES))[
                "parameter_plateau_status"
            ].to_list()
        )
        == {"NOT_EVALUATED"},
        "cover_grid_exact": cover_grid_exact,
        "cover_formula_parity_all_rows": cover_formula_parity,
        "cover_h_zero_equals_short": h_zero_short_parity,
        "cover_anchor_unique_and_primary": cover_anchor_exact,
        "primary_cover_scale_is_one": primary_scale_exact,
        "decision_anchor_is_primary_contract_anchor": decision_uses_primary_only,
        "project_backtest_plot_count": len(plot_audits) == len(CANDIDATES) * 2,
        "project_backtest_pnl_parity": all(
            close(audit["expected_net_pnl"], audit["backtest_net_pnl"])
            and close(
                audit["expected_net_pnl"],
                audit["canonical_daily_net_pnl"],
            )
            for audit in plot_audits
        ),
        "all_required_plots_exist": all(
            path.exists() and path.stat().st_size > 0
            for path in (
                COMPARISON_PLOT_PATH,
                COVER_PLOT_PATH,
                *[
                    OUTPUT_DIR / f"backtest_full_{candidate}.png"
                    for candidate in CANDIDATES
                ],
                *[
                    OUTPUT_DIR / f"backtest_2026_{candidate}.png"
                    for candidate in CANDIDATES
                ],
            )
        ),
    }
    return {
        "contract": "netfill_long_cover_verification/v2",
        "all_passed": all(checks.values()),
        "checks": checks,
        "details": {
            "accepted_rows": len(accepted),
            "accepted_candidate_counts": {
                row["candidate_id"]: int(row["len"])
                for row in accepted.group_by("candidate_id")
                .len()
                .sort("candidate_id")
                .iter_rows(named=True)
            },
            "accepted_net_pnl": expected_net,
            "daily_net_pnl": daily_net,
            "period_full_net_pnl": period_full_net,
            "plot_audits": plot_audits,
        },
    }


def _finalize(
    calendar: list[str],
    *,
    contract: dict[str, Any],
    contract_sha256: str,
    materialization_identity: dict[str, Any],
    materialization_identity_sha256: str,
) -> None:
    if RUN_MANIFEST_PATH.exists():
        raise FileExistsError(
            f"kept artifact already finalized: {RUN_MANIFEST_PATH}; "
            "create a new artifact version instead of overwriting"
        )
    signal_paths = _daily_partition_paths(DAILY_SIGNAL_DIR, calendar)
    count_paths = _daily_partition_paths(DAILY_COUNT_DIR, calendar)
    accepted = pl.concat(
        [pl.read_parquet(path) for path in signal_paths],
        how="vertical",
    ).sort("candidate_id", "Date", "QuoteCode", "TransTime", "ChannelSeq")
    counts = pl.concat(
        [pl.read_parquet(path) for path in count_paths],
        # A fully zero funnel day is inferred as Int64 while populated days
        # retain UInt32 source counts.  Relax only the integer supertype at
        # final aggregation; row values and per-day lineage stay unchanged.
        how="vertical_relaxed",
    ).sort("Date", "candidate_id")
    accepted_identity_values = set(
        accepted["materialization_identity_sha256"].unique().to_list()
    )
    if (
        counts["materialization_identity_sha256"].unique().to_list()
        != [materialization_identity_sha256]
        or not accepted_identity_values.issubset(
            {materialization_identity_sha256}
        )
    ):
        raise RuntimeError("mixed or stale materialization implementation identity")
    _atomic_parquet(accepted, ACCEPTED_PATH)
    _atomic_parquet(counts, DAILY_COUNTS_PATH)

    daily = _daily_frame(accepted, calendar, fee_bp=PRIMARY_FEE_BP)
    period_summary = _build_period_summary(accepted, daily, calendar)
    fee_sensitivity = _build_fee_sensitivity(accepted, calendar)
    monthly_summary = _build_monthly_summary(daily)
    short, short_identities = _load_short_daily(calendar)
    family_bias = _build_family_bias_adjustment(daily, short, contract)
    cover_summary, cover_details = _build_cover_summary(daily, short, contract)
    decision = _build_decision(
        period_summary,
        family_bias,
        cover_summary,
        contract,
    )
    trial_ledger = _build_trial_ledger(
        counts,
        daily,
        period_summary,
        family_bias,
        decision,
        contract,
        materialization_identity_sha256=(
            materialization_identity_sha256
        ),
    )

    _atomic_parquet(daily, DAILY_PATH)
    _atomic_parquet(period_summary, PERIOD_SUMMARY_PATH)
    _atomic_parquet(monthly_summary, MONTHLY_SUMMARY_PATH)
    _atomic_parquet(fee_sensitivity, FEE_SENSITIVITY_PATH)
    _atomic_json(family_bias, FAMILY_BIAS_PATH)
    _atomic_parquet(trial_ledger, TRIAL_LEDGER_PATH)
    _atomic_parquet(cover_summary, COVER_SUMMARY_PATH)
    _atomic_json(decision, DECISION_PATH)

    plot_audits: list[dict[str, Any]] = []
    full_dates = calendar
    dates_2026 = _period_dates(calendar, PERIODS["year_2026_historical"])
    for candidate in CANDIDATES:
        plot_audits.append(
            _save_project_backtest_plot(
                accepted,
                daily,
                candidate=candidate,
                dates=full_dates,
                path=OUTPUT_DIR / f"backtest_full_{candidate}.png",
                title=f"{candidate} · full retrospective",
            )
        )
        plot_audits.append(
            _save_project_backtest_plot(
                accepted,
                daily,
                candidate=candidate,
                dates=dates_2026,
                path=OUTPUT_DIR / f"backtest_2026_{candidate}.png",
                title=f"{candidate} · 2026 historical YTD",
            )
        )
    _save_comparison_plot(daily, COMPARISON_PLOT_PATH)
    _save_cover_plot(
        daily,
        short,
        anchor=float(contract["cover_evaluation"]["anchor_native_multiplier"]),
        path=COVER_PLOT_PATH,
    )

    verification = _verify_outputs(
        accepted=accepted,
        counts=counts,
        daily=daily,
        period_summary=period_summary,
        cover_summary=cover_summary,
        family_bias=family_bias,
        decision=decision,
        trial_ledger=trial_ledger,
        short=short,
        contract=contract,
        materialization_identity_sha256=(
            materialization_identity_sha256
        ),
        calendar=calendar,
        plot_audits=plot_audits,
    )
    _atomic_json(verification, VERIFICATION_PATH)
    if not verification["all_passed"]:
        failing = [
            name
            for name, passed in verification["checks"].items()
            if not passed
        ]
        raise RuntimeError(f"verification failed: {failing}")

    implementation_paths = (
        CONTRACT_PATH,
        Path(__file__),
        STUDY_DIR / "analysis_utils.py",
        HIGH_PROFILE_PATH,
        LOW_PROFILE_PATH,
        LABEL_SOURCE_PATH,
        HISTORICAL_TRIAL_EXPOSURE_PATH,
        BACKTEST_SOURCE_PATH,
    )
    artifact_paths = (
        ACCEPTED_PATH,
        DAILY_PATH,
        DAILY_COUNTS_PATH,
        PERIOD_SUMMARY_PATH,
        MONTHLY_SUMMARY_PATH,
        FEE_SENSITIVITY_PATH,
        FAMILY_BIAS_PATH,
        TRIAL_LEDGER_PATH,
        COVER_SUMMARY_PATH,
        DECISION_PATH,
        VERIFICATION_PATH,
        COMPARISON_PLOT_PATH,
        COVER_PLOT_PATH,
        *[
            OUTPUT_DIR / f"backtest_full_{candidate}.png"
            for candidate in CANDIDATES
        ],
        *[
            OUTPUT_DIR / f"backtest_2026_{candidate}.png"
            for candidate in CANDIDATES
        ],
    )
    source_identity_rows = counts.select(
        "Date", "source_identities_json"
    ).unique()
    if len(source_identity_rows) != len(calendar):
        raise RuntimeError("source identity calendar drift")
    manifest = {
        "contract": contract["contract"],
        "contract_sha256": contract_sha256,
        "materialization_identity_sha256": (
            materialization_identity_sha256
        ),
        "materialization_identity": materialization_identity,
        "status": "complete",
        "evidence_class": contract["evidence_class"],
        "candidate_ids": list(CANDIDATES),
        "calendar": calendar,
        "source_identities": {
            row["Date"]: json.loads(row["source_identities_json"])
            for row in source_identity_rows.sort("Date").iter_rows(named=True)
        },
        "short_snapshot_inputs": short_identities,
        "long_label_profile_id": LONG_TAKER_TP109_V1.profile_id,
        "long_label_profile_hash": LONG_TAKER_TP109_V1.fingerprint,
        "primary_fee_bp": PRIMARY_FEE_BP,
        "decision": decision["decision"],
        "implementation_files": {
            str(path.relative_to(PROJECT_ROOT)): _file_identity(path)
            for path in implementation_paths
        },
        "daily_partitions": {
            "signals": [_file_identity(path) for path in signal_paths],
            "counts": [_file_identity(path) for path in count_paths],
        },
        "artifacts": {
            str(path.relative_to(OUTPUT_DIR)): _file_identity(path)
            for path in artifact_paths
        },
        "cover_details": cover_details,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "polars": importlib.metadata.version("polars"),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scipy": importlib.metadata.version("scipy"),
            "matplotlib": importlib.metadata.version("matplotlib"),
        },
    }
    _atomic_json(_json_value(manifest), RUN_MANIFEST_PATH)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen signed-FillLots long-cover backtest"
    )
    parser.add_argument(
        "--stage",
        choices=("all", "materialize", "finalize"),
        default="all",
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract, contract_sha256 = _read_contract()
    (
        materialization_identity,
        materialization_identity_json,
        materialization_identity_sha256,
    ) = _materialization_identity(contract_sha256)
    calendar = _read_calendar(contract)
    if args.stage in {"all", "materialize"}:
        materialize_calendar = [
            date
            for date in calendar
            if (args.start is None or date >= args.start)
            and (args.end is None or date <= args.end)
        ]
        if not materialize_calendar:
            raise ValueError("requested materialization range has no sessions")
        _materialize_all(
            materialize_calendar,
            contract_sha256=contract_sha256,
            materialization_identity_sha256=(
                materialization_identity_sha256
            ),
            materialization_identity_json=materialization_identity_json,
        )
    if args.stage in {"all", "finalize"}:
        if args.start is not None or args.end is not None:
            raise ValueError("finalize always requires the complete frozen calendar")
        _finalize(
            calendar,
            contract=contract,
            contract_sha256=contract_sha256,
            materialization_identity=materialization_identity,
            materialization_identity_sha256=(
                materialization_identity_sha256
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
