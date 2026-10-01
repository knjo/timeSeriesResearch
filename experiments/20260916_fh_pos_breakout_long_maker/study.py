"""Frozen contracts for the fh_pos_breakout maker-long signal study.

Fork of ``20260826_filllots_at_high_signal/study.py`` with four changes:

- selector is the promoted long candidate ``fh_pos_breakout`` (frozen thresholds);
- label is ``fhl.long_maker_tp109.v1`` (maker buy at BidPrice1, exit at
  ``1.09 * RefPrice`` when FutureHigh crosses it, otherwise 13:30 Close; beta/TXF
  hedge and residual-vol z stay in the label for model targets only);
- a maker fill gate from ``makerFill``: a Bid1 order placed 100ms after the
  event must fill within ``FILL_HORIZON_SECONDS`` or the row is not a trade;
- the D-1 defensive theme list (``DEFENSIVE_STOCK_SELECTION.md``) enters as two
  features (``in_top_list``, ``defensive_weight``) and as an execution-layer
  universe variant.

Everything else (eligible universe, 16-cell prior thresholds, static gates,
per-stock capacity) is inherited unchanged so the long line is comparable with
the short line.  Import-safe: no file discovery or fitting at import time.
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
from src.research.intradaySignal.framework.profiles.fhl import LONG_MAKER_TP109_V1
from src.research.intradaySignal.framework.profiles.filllots_at_high import (
    FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION,
    select_filllots_at_high_feature_candidates,
)
from src.pipeline_storage import load_pipeline_storage


PROJECT_ROOT = Path(__file__).resolve().parents[5]
STUDY_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = STUDY_DIR / "artifacts"
EVENT_INDEX_PATH = ARTIFACT_DIR / "event_index_v1" / "fh_pos_breakout.parquet"
EVENT_DATASET_DIR = ARTIFACT_DIR / "event_dataset_v1"
FILL_DIR = ARTIFACT_DIR / "maker_fill_v1"
MODEL_DIR = ARTIFACT_DIR / "model_v1"
BACKTEST_DIR = ARTIFACT_DIR / "backtest_v1"
DEFENSIVE_LIST_PATH = (
    STUDY_DIR.parent
    / "20260916_defensive_long_universe"
    / "artifacts"
    / "v1"
    / "defensive_stock_daily.parquet"
)

# Defensive list coverage ends 2026-08-10; the study calendar stops there so the
# universe features are defined on every session.
STUDY_START = "20250211"
STUDY_END = "20260810"
DEVELOPMENT_END = "20251231"
HOLDOUT_START = "20260101"

CANDIDATE_ID = "fh_pos_breakout"
EVENT_ID = "filllots_at_high.fh_pos_breakout.long"
SELECTOR_VERSION = (
    f"filllots_at_high_feature_v{FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION}."
    f"{CANDIDATE_ID}"
)
DATASET_VERSION = "causal_union_v1"
LABEL_SPEC = LONG_MAKER_TP109_V1

MAKER_FILL_DELAY_MS = 100
FILL_HORIZON_SECONDS = 600.0
FILL_HORIZON_SENSITIVITY = (60.0, 300.0, 600.0)

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

FEE_BP = 19.3
PER_STOCK_POSITION_CAP_WAN = 200.0
PRICE_CAP = 1000.0
ENTRY_PRICE_COL = "BidPrice1"


def select_long_events(day: EventDaySources) -> pl.DataFrame:
    """Adapt the promoted fixed long candidate to the canonical event selector."""

    return select_filllots_at_high_feature_candidates(
        day,
        candidate_ids=(CANDIDATE_ID,),
    )[CANDIDATE_ID]


def event_index_spec(*, selector=select_long_events) -> EventIndexSpec:
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
    universe_id="fh_pos_breakout_daytrade_rank150_090030_123000_bidref",
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

DEFENSIVE_FEATURES = (
    "in_top_list",
    "defensive_weight",
)


def _feature_spec(name: str) -> FeatureSpec:
    if name in DEFENSIVE_FEATURES:
        return FeatureSpec(
            name=name,
            role="tick",
            required=True,
            missing_policy="zero",
            fallback=0.0,
            source="defensive_stock_daily_point_in_time",
            availability="pre_market",
            transform="study_v1",
        )
    return FeatureSpec(
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
            "precomputed" if name in BASE_FEATURES or name in VOL_FEATURES else "study_v1"
        ),
    )


FEATURE_MANIFEST = FeatureManifest(
    name="fh_pos_breakout_long_maker_defensive",
    version="1",
    features=tuple(
        _feature_spec(name)
        for name in BASE_FEATURES
        + NORMALIZED_FEATURES
        + HIGH_CONTEXT_FEATURES
        + VOL_FEATURES
        + DEFENSIVE_FEATURES
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
            "AskPrice1",
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
        raise ValueError(f"model input missing columns: {missing}")

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
        ((pl.col("BidPrice1") - pl.col("RefPrice")) / pl.col("RefPrice")).round(6).alias("ToRef"),
        ((pl.col("BidPrice1") - pl.col("Open")) / pl.col("Open")).round(6).alias("ToOpen"),
        _safe_ratio(pl.col("BidLots2"), bid_depth).alias("B2_Last"),
        _safe_ratio(pl.col("AskLots2"), ask_depth).alias("A2_Last"),
        _safe_ratio(pl.col("BidLots4") + pl.col("BidLots5"), book_45).alias("B45_AB45"),
        _safe_ratio(pl.col("BidLots1"), pl.col("BidLots1") + pl.col("BidLots2")).alias("B1_B12"),
        _safe_ratio(pl.col("AskLots1"), pl.col("AskLots1") + pl.col("AskLots2")).alias("A1_A12"),
        _safe_ratio(pl.col("LOB_BidVelocity_30").abs(), bid_depth).alias("LOB_BidVelocity_30_re"),
        _safe_ratio(pl.col("LOB_AskVelocity_30").abs(), ask_depth).alias("LOB_AskVelocity_30_re"),
        _safe_ratio(pl.col("FillLots_atLow"), total_depth).alias("FillLots_atLow_re"),
        _safe_ratio(pl.col("FillLots_atHigh"), total_depth).alias("FillLots_atHigh_re"),
        _safe_ratio(pl.col("L1_SellBiggestLots_30").abs(), pl.col("big_sell_lots")).alias("L1_SellBiggestLots_30_re"),
        _safe_ratio(pl.col("L1_BuyBiggestLots_30").abs(), pl.col("big_buy_lots")).alias("L1_BuyBiggestLots_30_re"),
        pl.when(pl.col("MD_ElaspeTime_30").is_finite() & (pl.col("MD_ElaspeTime_30") >= 0.0))
        .then(pl.col("MD_ElaspeTime_30").cast(pl.Float64).log1p())
        .alias("MD_ElaspeTime_30_re"),
        pl.col("MD_L1Rate_30").cast(pl.Float64).alias("MD_L1Rate_30_re"),
        _abs_log1p(pl.col("IA_BuyImpact_30")).alias("IA_BuyImpact_30_re"),
        _abs_log1p(pl.col("IA_SellImpact_30")).alias("IA_SellImpact_30_re"),
        _abs_log1p(pl.col("LOB_NetPressure_30")).alias("LOB_NetPressure_30_re"),
        _abs_log1p(pl.col("QL_Asymmetry")).alias("QL_Asymmetry_re"),
        _signed_log1p(pl.col("LOB_NetPressure_30")).alias("LOB_NetPressure_30_signed_log"),
        _signed_log1p(pl.col("QL_Asymmetry")).alias("QL_Asymmetry_signed_log"),
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


def add_defensive_features(frame: pl.DataFrame, defensive_path: Path = DEFENSIVE_LIST_PATH) -> pl.DataFrame:
    """Join the point-in-time top-quintile list as (Date, QuoteCode) features.

    ``in_top_list`` is 1 when the stock belongs to a top-20% theme that day (D-1
    information only) and ``defensive_weight`` is the summed member weight; both
    are 0 when the stock is not in the list.  Row count must not change.
    """

    top = (
        pl.read_parquet(defensive_path)
        .filter(pl.col("side") == "top")
        .select(
            pl.col("decision_date").dt.strftime("%Y%m%d").alias("Date"),
            pl.col("quote_code").alias("QuoteCode"),
            pl.col("defensive_weight").cast(pl.Float64),
        )
    )
    if top.select(pl.struct(["Date", "QuoteCode"]).is_duplicated().any()).item():
        raise ValueError("defensive list is not unique on (Date, QuoteCode)")
    before = frame.height
    out = frame.join(top, on=["Date", "QuoteCode"], how="left").with_columns(
        pl.col("defensive_weight").is_not_null().cast(pl.Float64).alias("in_top_list"),
        pl.col("defensive_weight").fill_null(0.0),
    )
    if out.height != before:
        raise ValueError("defensive join changed the event row count")
    return out


def maker_fill_dir() -> Path:
    storage = load_pipeline_storage(project_root=PROJECT_ROOT)
    return Path(storage.maker_queue_dir)


def add_delayed_maker_fill_seconds(
    events: pl.DataFrame,
    *,
    date: str,
    roots: DataRoots,
    fill_dir: Path,
    delay_ms: int = MAKER_FILL_DELAY_MS,
) -> pl.DataFrame:
    """Seconds until a Bid1 maker buy placed ``delay_ms`` after the event fills.

    Reproduces the followHFTMaker / alpha_factory delayed lookup: find the tick
    at ``TransTime + delay``, check whether the event's BidPrice1 is still the
    best bid (Bid1 queue), has been undercut (immediate fill, 0s), or is now the
    second level (Bid2 queue); anything deeper is unknown and treated as no fill.
    Returns ``events`` with ``maker_buy_fill_seconds`` (null = not fillable).
    """

    tick_path = roots.tick_data_dir / f"{date}_StockTick.parquet"
    fill_path = fill_dir / f"{date}_makerFill.parquet"
    if not fill_path.exists():
        raise FileNotFoundError(fill_path)
    codes = events.select("QuoteCode").unique()
    lookup = (
        pl.scan_parquet(tick_path)
        .select("QuoteCode", "TransTime", "ChannelSeq", "BidPrice1", "BidPrice2")
        .join(codes.lazy(), on="QuoteCode", how="semi")
        .collect()
        .rename(
            {
                "TransTime": "_lookup_time",
                "ChannelSeq": "_d_ChannelSeq",
                "BidPrice1": "_d_BidPrice1",
                "BidPrice2": "_d_BidPrice2",
            }
        )
        .with_columns(
            pl.col("_d_BidPrice1").cast(pl.Float64),
            pl.col("_d_BidPrice2").cast(pl.Float64),
        )
        .sort(["QuoteCode", "_lookup_time"])
    )
    fills = (
        pl.scan_parquet(fill_path)
        .select("QuoteCode", "ChannelSeq", "Bid1_FillSeconds", "Bid2_FillSeconds")
        .join(codes.lazy(), on="QuoteCode", how="semi")
        .collect()
        .rename({"ChannelSeq": "_d_ChannelSeq"})
        .with_columns(
            pl.col("Bid1_FillSeconds").cast(pl.Float64),
            pl.col("Bid2_FillSeconds").cast(pl.Float64),
        )
    )
    out = (
        events.with_row_index("__row_id")
        .with_columns(
            (pl.col("TransTime") + pl.duration(milliseconds=delay_ms)).alias("_target_time"),
            pl.col("BidPrice1").cast(pl.Float64).alias("_bid"),
        )
        .sort(["QuoteCode", "_target_time"])
        .join_asof(lookup, left_on="_target_time", right_on="_lookup_time", by="QuoteCode", strategy="backward")
        .join(fills, on=["QuoteCode", "_d_ChannelSeq"], how="left")
    )
    eps = 1e-4
    same_bid1 = (pl.col("_d_BidPrice1") - pl.col("_bid")).abs() < eps
    undercut = pl.col("_d_BidPrice1") < pl.col("_bid")
    same_bid2 = (pl.col("_d_BidPrice2") - pl.col("_bid")).abs() < eps
    out = out.with_columns(
        pl.when(same_bid1)
        .then(pl.col("Bid1_FillSeconds"))
        .when(undercut)
        .then(0.0)
        .when(same_bid2)
        .then(pl.col("Bid2_FillSeconds"))
        .otherwise(None)
        .alias("maker_buy_fill_seconds"),
        # which branch produced the estimate; ``undercut_immediate`` is the
        # followHFTMaker convention (best bid fell below our price -> 0s) and is
        # the optimistic case, so it is kept separately for sensitivity.
        pl.when(same_bid1)
        .then(pl.lit("bid1_queue"))
        .when(undercut)
        .then(pl.lit("undercut_immediate"))
        .when(same_bid2)
        .then(pl.lit("bid2_queue"))
        .otherwise(pl.lit("unknown"))
        .alias("maker_fill_branch"),
        pl.col("_d_ChannelSeq").alias("maker_fill_lookup_channelseq"),
    )
    keep = [*events.columns, "maker_buy_fill_seconds", "maker_fill_branch", "maker_fill_lookup_channelseq"]
    result = out.sort("__row_id").select(keep)
    if result.height != events.height:
        raise ValueError(f"{date}: fill join changed row count")
    return result


def add_threshold_buckets(frame: pl.DataFrame) -> pl.DataFrame:
    """Reproduce the notebook's four-by-four positive-ToRef cells."""

    return frame.with_columns(
        pl.col("RemainSeconds")
        .cut([3300.0, 6900.0, 10500.0, 14100.0], labels=["outside_low", "t0", "t1", "t2", "t3"], left_closed=False)
        .alias("_time_cut"),
        pl.col("ToRef")
        .cut([0.0, 0.01, 0.02, 0.03, 0.05], labels=["outside_low", "r0", "r1", "r2", "r3", "outside_high"], left_closed=False)
        .alias("_ref_cut"),
    ).with_columns(
        pl.when(pl.col("_time_cut").is_in(["t0", "t1", "t2", "t3"])).then(pl.col("_time_cut")).alias("time_bucket"),
        pl.when(pl.col("_ref_cut").is_in(["r0", "r1", "r2", "r3"])).then(pl.col("_ref_cut")).alias("to_ref_bucket"),
    ).drop("_time_cut", "_ref_cut")


def static_execution_filter(frame: pl.DataFrame) -> pl.DataFrame:
    """Causal liquidity / universe gates shared with the short line."""

    return frame.filter(
        pl.col("time_bucket").is_not_null()
        & pl.col("to_ref_bucket").is_not_null()
        & ((pl.col("OTC") == True) | (pl.col("MD_L1Rate_30_re") > 0.25))
        & (pl.col("SpreadPairElapsed") > 0.1)
        & ((pl.col("AmountRank_canDayTrade") <= 100) | (pl.col("day_amount_rank") <= 100))
        & (pl.col("BidPrice1") <= PRICE_CAP)
        & (pl.col("hft_strick_makerSpreadBP").is_null() | (pl.col("hft_strick_makerSpreadBP") > -70.0))
    )


def apply_capacity(frame: pl.DataFrame) -> pl.DataFrame:
    """One-lot depth and single-stock notional caps after static gates."""

    ordered = frame.sort(["Date", "QuoteCode", "TransTime", "ChannelSeq"])
    groups = ["Date", "QuoteCode"]
    ordered = ordered.with_columns(
        pl.col(ENTRY_PRICE_COL).cum_count().over(groups).alias("accLots"),
        (pl.col(ENTRY_PRICE_COL).cum_sum().over(groups) / 10.0).alias("stock_position_wan"),
    )
    return ordered.filter(
        (pl.col("accLots") < pl.col("avg_bidLots1") + pl.col("avg_askLots1"))
        & (pl.col("stock_position_wan") < PER_STOCK_POSITION_CAP_WAN)
    )


def data_roots() -> DataRoots:
    return DataRoots.from_project_root(PROJECT_ROOT)
