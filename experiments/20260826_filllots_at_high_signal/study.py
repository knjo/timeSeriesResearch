"""Frozen contracts for the FillLots-at-high short signal study.

This module is intentionally import-safe.  It owns the event adapter, eligible
universe, fixed feature manifest, and deterministic feature/risk transforms.
Outcome columns are used only by the label builder in the runner.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from src.research.intradaySignal.framework.dataset import (
    EventDatasetSpec,
    MergeOptions,
    UniverseSpec,
)
from src.research.intradaySignal.framework.events import (
    DataRoots,
    EventDaySources,
    EventIndexSpec,
)
from src.research.intradaySignal.framework.features import FeatureManifest, FeatureSpec
from src.research.intradaySignal.framework.profiles.filllots_at_high import (
    FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION,
    select_filllots_at_high_feature_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[5]
STUDY_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = STUDY_DIR / "artifacts"
EVENT_INDEX_PATH = ARTIFACT_DIR / "event_index_v1" / "fh_neg_rejection.parquet"
EVENT_DATASET_DIR = ARTIFACT_DIR / "event_dataset_v1"
MODEL_DIR = ARTIFACT_DIR / "model_v1"
BACKTEST_DIR = ARTIFACT_DIR / "backtest_v1"

STUDY_START = "20250211"
STUDY_END = "20260825"
DEVELOPMENT_END = "20251231"
HOLDOUT_START = "20260101"

EVENT_ID = "filllots_at_high.fh_neg_rejection.short"
SELECTOR_VERSION = (
    f"filllots_at_high_feature_v{FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION}."
    "fh_neg_rejection"
)
DATASET_VERSION = "causal_union_v1"

TIMING_TARGET = "label_z_within_stock_day"
SELECTION_TARGET = "label_z_within_day"
TIMING_LOOKBACK_DAYS = 60
SELECTION_LOOKBACK_DAYS = 20
RIDGE_ALPHA = 0.1
TRAINING_POSITION_CAP_WAN = 500.0

TIMING_QUANTILE = 0.30
SELECTION_QUANTILE = 0.10
THRESHOLD_LOOKBACK_SESSIONS = 3
THRESHOLD_MIN_SESSIONS = 3
THRESHOLD_MIN_OBSERVATIONS = 30

TAKER_FEE_BP = 19.3
PER_STOCK_POSITION_CAP_WAN = 200.0
PRICE_CAP = 1000.0


def select_short_events(day: EventDaySources) -> pl.DataFrame:
    """Adapt the promoted fixed candidate to the canonical event selector."""

    return select_filllots_at_high_feature_candidates(
        day,
        candidate_ids=("fh_neg_rejection",),
    )["fh_neg_rejection"]


def event_index_spec(*, selector=select_short_events) -> EventIndexSpec:
    return EventIndexSpec(
        event_id=EVENT_ID,
        selector_version=SELECTOR_VERSION,
        selector=selector,
        index_path=EVENT_INDEX_PATH,
        required_sources=("tick_data", "tick_feature", "pre_market"),
        calendar_source="tick_data",
    )


def _eligible_universe(frame: pl.DataFrame) -> pl.DataFrame:
    entry_to_ref = (
        (pl.col("BidPrice1") - pl.col("RefPrice")) / pl.col("RefPrice")
    )
    return frame.filter(
        (pl.col("BidPrice1") > 0.0)
        & (pl.col("RefPrice") > 0.0)
        & (pl.col("Open") > 0.0)
        & (pl.col("TrialMatch") == 0)
        & (pl.col("TransTime").dt.time() > dt.time(9, 0, 30))
        & (pl.col("TransTime").dt.time() < dt.time(12, 30))
        & (pl.col("nextday_allow_day_trade_mark") == "X")
        & (pl.col("day_amount_rank") <= 150)
        & (entry_to_ref > -0.015)
        & (entry_to_ref < 0.05)
    )


ELIGIBLE_UNIVERSE_V1 = UniverseSpec(
    universe_id="fh_neg_rejection_daytrade_rank150_090030_123000_bidref",
    universe_version="1",
    filter_fn=_eligible_universe,
)


def event_dataset_spec() -> EventDatasetSpec:
    return EventDatasetSpec(
        event_id=EVENT_ID,
        dataset_version=DATASET_VERSION,
        index_path=EVENT_INDEX_PATH,
        output_dir=EVENT_DATASET_DIR,
        merge_options=MergeOptions(
            universe=ELIGIBLE_UNIVERSE_V1,
            require_pre_market=True,
            attach_tick_bar=True,
            require_tick_bar=True,
            tick_bar_alignment="causal_completed_slot",
            tick_bar_columns=("AmountRank", "AmountRank_canDayTrade"),
            attach_txf=True,
            require_txf=True,
            strict_cardinality=True,
            schema_policy="union",
        ),
    )


BASE_FEATURES = (
    "ToLow",
    "ToHigh",
    "Low_High",
    "RemainSeconds",
    "ToOpen",
    "ToRef",
    "B1_A1B1",
    "B1_B1B5",
    "A1_A1A5",
    "B2_Last",
    "A2_Last",
    "B45_AB45",
    "B1_B12",
    "A1_A12",
    "QL_BidHHI",
    "QL_AskHHI",
)

NORMALIZED_FEATURES = (
    "LOB_BidVelocity_30_re",
    "LOB_AskVelocity_30_re",
    "FillLots_atLow_re",
    "FillLots_atHigh_re",
    "L1_SellBiggestLots_30_re",
    "L1_BuyBiggestLots_30_re",
    "MD_ElaspeTime_30_re",
    "MD_L1Rate_30_re",
    "IA_BuyImpact_30_re",
    "IA_SellImpact_30_re",
    "LOB_NetPressure_30_re",
    "QL_Asymmetry_re",
)

HIGH_CONTEXT_FEATURES = (
    "MicroMomentum_30",
    "LOB_NetPressure_30_signed_log",
    "QL_Asymmetry_signed_log",
    "filllots_at_high_gap_ticks",
    "Spread_ticks",
    "fill_size_depth_ratio_log",
    "BidPreMove",
    "AskPreMove",
)

VOL_FEATURES = (
    "txf_residual_vol_to_1330",
    "txf_residual_vol_0900_1330_60d",
)

FEATURE_MANIFEST = FeatureManifest(
    name="filllots_at_high_short_context",
    version="1",
    features=tuple(
        FeatureSpec(
            name=name,
            role="tick",
            required=True,
            missing_policy="prior_median",
            fallback=0.0,
            source=(
                "study_event_transform"
                if name in NORMALIZED_FEATURES or name in HIGH_CONTEXT_FEATURES
                else "tick_feature_or_causal_market_context"
            ),
            availability="event",
            transform=(
                "precomputed"
                if name in BASE_FEATURES or name in VOL_FEATURES
                else "study_v1"
            ),
        )
        for name in BASE_FEATURES
        + NORMALIZED_FEATURES
        + HIGH_CONTEXT_FEATURES
        + VOL_FEATURES
    ),
)


REQUIRED_MODEL_INPUT_COLUMNS = tuple(
    dict.fromkeys(
        (
            "Date",
            "QuoteCode",
            "ChannelSeq",
            "TransTime",
            "TrialMatch",
            "BidPrice1",
            "Open",
            "Close",
            "FutureHigh",
            "RefPrice",
            "TickSize",
            "OTC",
            "BidLots1",
            "BidLots2",
            "BidLots4",
            "BidLots5",
            "AskLots1",
            "AskLots2",
            "AskLots4",
            "AskLots5",
            "FillLots",
            "InOut",
            "FillLots_atLow",
            "FillLots_atHigh",
            "LOB_BidVelocity_30",
            "LOB_AskVelocity_30",
            "L1_SellBiggestLots_30",
            "L1_BuyBiggestLots_30",
            "MD_ElaspeTime_30",
            "MD_L1Rate_30",
            "IA_BuyImpact_30",
            "IA_SellImpact_30",
            "LOB_NetPressure_30",
            "QL_Asymmetry",
            "MicroMomentum_30",
            "Spread",
            "SpreadPairElapsed",
            "BidPreMove",
            "AskPreMove",
            "ToLow",
            "ToHigh",
            "Low_High",
            "RemainSeconds",
            "B1_A1B1",
            "B1_B1B5",
            "A1_A1A5",
            "QL_BidHHI",
            "QL_AskHHI",
            "avg_bidLots1",
            "avg_askLots1",
            "big_buy_lots",
            "big_sell_lots",
            "hft_strick_makerSpreadBP",
            "day_amount_rank",
            "AmountRank_canDayTrade",
            "txf_beta_60d",
            "txf_to_1330_bp",
            "txf_residual_vol_to_1330",
            "txf_residual_vol_0900_1330_60d",
        )
    )
)


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    denominator = denominator.cast(pl.Float64, strict=False)
    numerator = numerator.cast(pl.Float64, strict=False)
    return pl.when(
        numerator.is_finite() & denominator.is_finite() & (denominator > 0.0)
    ).then(numerator / denominator)


def _abs_log1p(value: pl.Expr) -> pl.Expr:
    value = value.cast(pl.Float64, strict=False)
    return pl.when(value.is_finite()).then(value.abs().log1p())


def _signed_log1p(value: pl.Expr) -> pl.Expr:
    value = value.cast(pl.Float64, strict=False)
    return pl.when(value.is_finite()).then(value.sign() * value.abs().log1p())


def add_study_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach the fixed event-time feature transforms without future fields."""

    missing = sorted(set(REQUIRED_MODEL_INPUT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"FillHigh model input missing columns: {missing}")

    bid_depth = pl.col("avg_bidLots1")
    ask_depth = pl.col("avg_askLots1")
    total_depth = bid_depth + ask_depth
    book_45 = (
        pl.col("BidLots4")
        + pl.col("BidLots5")
        + pl.col("AskLots4")
        + pl.col("AskLots5")
    )
    tick_size = pl.col("TickSize").cast(pl.Float64, strict=False)
    ref_price = pl.col("RefPrice").cast(pl.Float64, strict=False)

    return frame.with_columns(
        (
            (pl.col("BidPrice1") - pl.col("RefPrice")) / pl.col("RefPrice")
        )
        .round(6)
        .alias("ToRef"),
        ((pl.col("BidPrice1") - pl.col("Open")) / pl.col("Open"))
        .round(6)
        .alias("ToOpen"),
        _safe_ratio(pl.col("BidLots2"), bid_depth).alias("B2_Last"),
        _safe_ratio(pl.col("AskLots2"), ask_depth).alias("A2_Last"),
        _safe_ratio(
            pl.col("BidLots4") + pl.col("BidLots5"), book_45
        ).alias("B45_AB45"),
        _safe_ratio(
            pl.col("BidLots1"), pl.col("BidLots1") + pl.col("BidLots2")
        ).alias("B1_B12"),
        _safe_ratio(
            pl.col("AskLots1"), pl.col("AskLots1") + pl.col("AskLots2")
        ).alias("A1_A12"),
        _safe_ratio(pl.col("LOB_BidVelocity_30").abs(), bid_depth).alias(
            "LOB_BidVelocity_30_re"
        ),
        _safe_ratio(pl.col("LOB_AskVelocity_30").abs(), ask_depth).alias(
            "LOB_AskVelocity_30_re"
        ),
        _safe_ratio(pl.col("FillLots_atLow"), total_depth).alias(
            "FillLots_atLow_re"
        ),
        _safe_ratio(pl.col("FillLots_atHigh"), total_depth).alias(
            "FillLots_atHigh_re"
        ),
        _safe_ratio(
            pl.col("L1_SellBiggestLots_30").abs(), pl.col("big_sell_lots")
        ).alias("L1_SellBiggestLots_30_re"),
        _safe_ratio(
            pl.col("L1_BuyBiggestLots_30").abs(), pl.col("big_buy_lots")
        ).alias("L1_BuyBiggestLots_30_re"),
        pl.when(
            pl.col("MD_ElaspeTime_30").is_finite()
            & (pl.col("MD_ElaspeTime_30") >= 0.0)
        )
        .then(pl.col("MD_ElaspeTime_30").cast(pl.Float64).log1p())
        .alias("MD_ElaspeTime_30_re"),
        pl.col("MD_L1Rate_30").cast(pl.Float64).alias("MD_L1Rate_30_re"),
        _abs_log1p(pl.col("IA_BuyImpact_30")).alias("IA_BuyImpact_30_re"),
        _abs_log1p(pl.col("IA_SellImpact_30")).alias("IA_SellImpact_30_re"),
        _abs_log1p(pl.col("LOB_NetPressure_30")).alias(
            "LOB_NetPressure_30_re"
        ),
        _abs_log1p(pl.col("QL_Asymmetry")).alias("QL_Asymmetry_re"),
        _signed_log1p(pl.col("LOB_NetPressure_30")).alias(
            "LOB_NetPressure_30_signed_log"
        ),
        _signed_log1p(pl.col("QL_Asymmetry")).alias(
            "QL_Asymmetry_signed_log"
        ),
        pl.when(
            pl.col("ToHigh").is_finite()
            & ref_price.is_finite()
            & tick_size.is_finite()
            & (ref_price > 0.0)
            & (tick_size > 0.0)
        )
        .then(pl.col("ToHigh") * ref_price / tick_size)
        .alias("filllots_at_high_gap_ticks"),
        _safe_ratio(pl.col("Spread"), tick_size).alias("Spread_ticks"),
        pl.when(_safe_ratio(pl.col("FillLots"), total_depth).is_not_null())
        .then(_safe_ratio(pl.col("FillLots"), total_depth).log1p())
        .alias("fill_size_depth_ratio_log"),
    )


def add_threshold_buckets(frame: pl.DataFrame) -> pl.DataFrame:
    """Reproduce the notebook's four-by-four positive-ToRef cells."""

    return frame.with_columns(
        pl.col("RemainSeconds")
        .cut(
            [3300.0, 6900.0, 10500.0, 14100.0],
            labels=["outside_low", "t0", "t1", "t2", "t3"],
            left_closed=False,
        )
        .alias("_time_cut"),
        pl.col("ToRef")
        .cut(
            [0.0, 0.01, 0.02, 0.03, 0.05],
            labels=["outside_low", "r0", "r1", "r2", "r3", "outside_high"],
            left_closed=False,
        )
        .alias("_ref_cut"),
    ).with_columns(
        pl.when(pl.col("_time_cut").is_in(["t0", "t1", "t2", "t3"]))
        .then(pl.col("_time_cut"))
        .alias("time_bucket"),
        pl.when(pl.col("_ref_cut").is_in(["r0", "r1", "r2", "r3"]))
        .then(pl.col("_ref_cut"))
        .alias("to_ref_bucket"),
    ).drop("_time_cut", "_ref_cut")


def static_execution_filter(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply causal/non-outcome gates before deterministic capacity shaping."""

    return frame.filter(
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


def apply_capacity(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply one-lot depth and single-stock notional caps after static gates."""

    ordered = frame.sort(["Date", "QuoteCode", "TransTime", "ChannelSeq"])
    groups = ["Date", "QuoteCode"]
    ordered = ordered.with_columns(
        pl.col("BidPrice1").cum_count().over(groups).alias("accLots"),
        (pl.col("BidPrice1").cum_sum().over(groups) / 10.0).alias(
            "stock_position_wan"
        ),
    )
    return ordered.filter(
        (pl.col("accLots") < pl.col("avg_bidLots1") + pl.col("avg_askLots1"))
        & (pl.col("stock_position_wan") < PER_STOCK_POSITION_CAP_WAN)
    )


def data_roots() -> DataRoots:
    return DataRoots.from_project_root(PROJECT_ROOT)


__all__ = [
    "ARTIFACT_DIR",
    "BACKTEST_DIR",
    "DATASET_VERSION",
    "DEVELOPMENT_END",
    "EVENT_DATASET_DIR",
    "EVENT_ID",
    "EVENT_INDEX_PATH",
    "FEATURE_MANIFEST",
    "HOLDOUT_START",
    "MODEL_DIR",
    "REQUIRED_MODEL_INPUT_COLUMNS",
    "RIDGE_ALPHA",
    "SELECTION_LOOKBACK_DAYS",
    "SELECTION_QUANTILE",
    "SELECTION_TARGET",
    "STUDY_END",
    "STUDY_START",
    "TAKER_FEE_BP",
    "THRESHOLD_LOOKBACK_SESSIONS",
    "THRESHOLD_MIN_OBSERVATIONS",
    "THRESHOLD_MIN_SESSIONS",
    "TIMING_LOOKBACK_DAYS",
    "TIMING_QUANTILE",
    "TIMING_TARGET",
    "TRAINING_POSITION_CAP_WAN",
    "add_study_features",
    "add_threshold_buckets",
    "apply_capacity",
    "data_roots",
    "event_dataset_spec",
    "event_index_spec",
    "select_short_events",
    "static_execution_filter",
]
