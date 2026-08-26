"""FillLots-at-high feature-conditioned event family.

``FillLots_atHigh`` is rebuilt on the complete raw tick stream as cumulative
signed flow within each ``(QuoteCode, RecordHigh)`` epoch.  A persistent state
is never sampled by itself: every registered candidate requires either a
top-of-book price change or an economically material formal trade.  This keeps
unchanged-price quote rows from repeatedly representing the same state while
still allowing a large new fill at the same price to become a fresh event.

The bounded order-flow and book-state thresholds are fixed study hypotheses,
not outcome-fitted quantiles.  Candidate selection happens before the shared
eligible labeled pool is applied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from ..events import DAY_KEY_COLS, EventDaySources, EventIndexValidationError


FILLLOTS_AT_HIGH_FEATURE_FAMILY_ID = "filllots_at_high_feature"
FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION = "1"

MIN_LARGE_FILL_LOTS = 20.0
POSITIVE_DEPTH_RATIO = 2.0
NEGATIVE_DEPTH_RATIO = -8.0
NEAR_HIGH_TICKS = 1.5
FAR_HIGH_TICKS = 4.0
WEAK_BID_SHARE = 0.35
STRONG_BID_SHARE = 0.65
MICRO_MOMENTUM_THRESHOLD_BP = 0.10

CANDIDATE_IDS = (
    "fh_pos_sample",
    "fh_neg_sample",
    "fh_pos_depth2",
    "fh_neg_depth8",
    "fh_pos_near",
    "fh_pos_far",
    "fh_neg_near",
    "fh_neg_far",
    "fh_pos_breakout",
    "fh_pos_exhaustion",
    "fh_neg_rejection",
    "fh_neg_absorption",
)

CANDIDATE_DESCRIPTIONS = {
    "fh_pos_sample": "fresh sample trigger and F_high > 0",
    "fh_neg_sample": "fresh sample trigger and F_high < 0",
    "fh_pos_depth2": "fresh trigger and F_high / prior top depth >= 2",
    "fh_neg_depth8": "fresh trigger and F_high / prior top depth <= -8",
    "fh_pos_near": "fresh trigger, F_high > 0, and <= 1.5 ticks below high",
    "fh_pos_far": "fresh trigger, F_high > 0, and >= 4 ticks below high",
    "fh_neg_near": "fresh trigger, F_high < 0, and <= 1.5 ticks below high",
    "fh_neg_far": "fresh trigger, F_high < 0, and >= 4 ticks below high",
    "fh_pos_breakout": (
        "fresh trigger, depth ratio >= 2, <= 1.5 ticks below high, and "
        "bullish 2-of-3 tick-feature score"
    ),
    "fh_pos_exhaustion": (
        "fresh trigger, depth ratio >= 2, <= 1.5 ticks below high, and "
        "bearish 2-of-3 tick-feature score"
    ),
    "fh_neg_rejection": (
        "fresh trigger, depth ratio <= -8, >= 4 ticks below high, and "
        "bearish 2-of-3 tick-feature score"
    ),
    "fh_neg_absorption": (
        "fresh trigger, depth ratio <= -8, <= 1.5 ticks below high, and "
        "bullish 2-of-3 tick-feature score"
    ),
}

if tuple(CANDIDATE_DESCRIPTIONS) != CANDIDATE_IDS:  # pragma: no cover
    raise RuntimeError("FillLots-at-high candidate descriptions drifted")


def _require_columns(
    frame: pl.LazyFrame,
    columns: tuple[str, ...],
    *,
    source: str,
) -> None:
    available = set(frame.collect_schema().names())
    missing = [column for column in columns if column not in available]
    if missing:
        raise EventIndexValidationError(
            f"{source}: missing FillLots-at-high candidate columns {missing}"
        )


def _price_to_tick_index(price: pl.Expr) -> pl.Expr:
    """Map a positive TW-equity price onto a stable integer tick lattice."""

    return (
        pl.when(price <= 0.0)
        .then(None)
        .when(price < 10.0)
        .then(price / 0.01)
        .when(price < 50.0)
        .then(1000.0 + (price - 10.0) / 0.05)
        .when(price < 100.0)
        .then(1800.0 + (price - 50.0) / 0.10)
        .when(price < 500.0)
        .then(2300.0 + (price - 100.0) / 0.50)
        .when(price < 1000.0)
        .then(3100.0 + (price - 500.0))
        .otherwise(3600.0 + (price - 1000.0) / 5.0)
        .round(0)
        .cast(pl.Int64)
    )


def _tick_size(price: pl.Expr) -> pl.Expr:
    return (
        pl.when(price < 10.0)
        .then(0.01)
        .when(price < 50.0)
        .then(0.05)
        .when(price < 100.0)
        .then(0.10)
        .when(price < 500.0)
        .then(0.50)
        .when(price < 1000.0)
        .then(1.0)
        .otherwise(5.0)
    )


def build_filllots_at_high_candidate_frame(
    day: EventDaySources,
) -> pl.DataFrame:
    """Rebuild high-epoch flow and attach current causal tick features."""

    tick = day.scan_tick_data()
    tick_feature = day.scan_tick_feature()
    pre_market = day.scan_pre_market()
    _require_columns(
        tick,
        (
            *DAY_KEY_COLS,
            "FillLots",
            "InOut",
            "TrialMatch",
            "marketOpen",
            "RecordHigh",
            "RefPrice",
            "BidPrice1",
            "AskPrice1",
        ),
        source=str(day.path("tick_data")),
    )
    _require_columns(
        tick_feature,
        (
            *DAY_KEY_COLS,
            "B1_A1B1",
            "LOB_NetPressure_30",
            "MicroMomentum_30",
            "ToHigh",
            "TickSize",
        ),
        source=str(day.path("tick_feature")),
    )
    _require_columns(
        pre_market,
        (
            "QuoteCode",
            "avg_bidLots1",
            "avg_askLots1",
            "big_buy_lots",
            "big_sell_lots",
        ),
        source=str(day.path("pre_market")),
    )

    signed_flow = (
        pl.col("FillLots")
        * pl.col("InOut")
        * (pl.col("TrialMatch") == 0)
    )
    prior_depth = pl.col("avg_bidLots1") + pl.col("avg_askLots1")
    formal_trade = (
        pl.col("marketOpen")
        & (pl.col("TrialMatch") == 0)
        & (pl.col("FillLots") > 0)
        & pl.col("InOut").is_in([-1, 1])
    )
    current_price = (
        pl.when(pl.col("BidPrice1") > 0.0)
        .then(pl.col("BidPrice1"))
        .otherwise(pl.col("AskPrice1"))
    )

    frame = (
        tick.select(
            *DAY_KEY_COLS,
            "FillLots",
            "InOut",
            "TrialMatch",
            "marketOpen",
            "RecordHigh",
            "RefPrice",
            "BidPrice1",
            "AskPrice1",
        )
        .sort(list(DAY_KEY_COLS))
        .with_columns(
            signed_flow.cum_sum()
            .over(["QuoteCode", "RecordHigh"])
            .alias("FillLots_atHigh"),
            pl.col("BidPrice1")
            .diff()
            .over("QuoteCode")
            .alias("_bid_price_move"),
            pl.col("AskPrice1")
            .diff()
            .over("QuoteCode")
            .alias("_ask_price_move"),
            _price_to_tick_index(pl.col("BidPrice1")).alias(
                "_bid_tick_index"
            ),
            _price_to_tick_index(pl.col("AskPrice1")).alias(
                "_ask_tick_index"
            ),
            _tick_size(current_price).alias("_tick_size"),
        )
        .with_columns(
            pl.col("ChannelSeq")
            .cum_count()
            .over(
                [
                    "QuoteCode",
                    "RecordHigh",
                    "_bid_tick_index",
                    "_ask_tick_index",
                ]
            )
            .alias("_price_pair_visit")
        )
        .join(
            pre_market.select(
                "QuoteCode",
                "avg_bidLots1",
                "avg_askLots1",
                "big_buy_lots",
                "big_sell_lots",
            ),
            on="QuoteCode",
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.when(prior_depth.is_finite() & (prior_depth > 0.0))
            .then(pl.col("FillLots_atHigh") / prior_depth)
            .alias("filllots_at_high_depth_ratio"),
            pl.when(prior_depth.is_finite() & (prior_depth > 0.0))
            .then(pl.col("FillLots") / prior_depth)
            .alias("fill_size_depth_ratio"),
            formal_trade.alias("_formal_trade"),
            pl.when(pl.col("InOut") > 0)
            .then(
                pl.when(
                    pl.col("big_buy_lots").is_finite()
                    & (pl.col("big_buy_lots") > 0.0)
                )
                .then(
                    pl.max_horizontal(
                        pl.lit(MIN_LARGE_FILL_LOTS),
                        pl.col("big_buy_lots"),
                    )
                )
            )
            .when(pl.col("InOut") < 0)
            .then(
                pl.when(
                    pl.col("big_sell_lots").is_finite()
                    & (pl.col("big_sell_lots") > 0.0)
                )
                .then(
                    pl.max_horizontal(
                        pl.lit(MIN_LARGE_FILL_LOTS),
                        pl.col("big_sell_lots"),
                    )
                )
            )
            .otherwise(None)
            .alias("_large_fill_threshold"),
        )
        .with_columns(
            (
                (pl.col("_price_pair_visit") == 1)
                & pl.col("_bid_tick_index").is_not_null()
                & pl.col("_ask_tick_index").is_not_null()
                & (
                    (
                        pl.col("_bid_price_move").abs()
                        >= pl.col("_tick_size") * 0.5
                    )
                    | (
                        pl.col("_ask_price_move").abs()
                        >= pl.col("_tick_size") * 0.5
                    )
                )
            )
            .fill_null(False)
            .alias("_md_price_change"),
            (
                (pl.col("_price_pair_visit") == 1)
                & (
                    pl.col("_bid_price_move")
                    <= pl.col("_tick_size") * -0.5
                )
            )
            .fill_null(False)
            .alias("_bid_down"),
            (
                (pl.col("_price_pair_visit") == 1)
                & (
                    pl.col("_bid_price_move")
                    >= pl.col("_tick_size") * 0.5
                )
            )
            .fill_null(False)
            .alias("_bid_up"),
            (
                (pl.col("_price_pair_visit") == 1)
                & (
                    pl.col("_ask_price_move")
                    <= pl.col("_tick_size") * -0.5
                )
            )
            .fill_null(False)
            .alias("_ask_down"),
            (
                (pl.col("_price_pair_visit") == 1)
                & (
                    pl.col("_ask_price_move")
                    >= pl.col("_tick_size") * 0.5
                )
            )
            .fill_null(False)
            .alias("_ask_up"),
            (
                pl.col("_formal_trade")
                & (pl.col("FillLots") >= MIN_LARGE_FILL_LOTS)
                & (pl.col("FillLots") >= pl.col("_large_fill_threshold"))
            )
            .fill_null(False)
            .alias("_large_fill"),
        )
        .with_columns(
            (pl.col("_md_price_change") | pl.col("_large_fill")).alias(
                "_fresh_event"
            )
        )
        # State and price deltas above are complete-raw-stream calculations.
        # The derived feature source may omit rows, so attach it only afterward.
        .join(
            tick_feature.select(
                *DAY_KEY_COLS,
                "B1_A1B1",
                "LOB_NetPressure_30",
                "MicroMomentum_30",
                "ToHigh",
                "TickSize",
            ),
            on=list(DAY_KEY_COLS),
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.when(
                pl.col("ToHigh").is_finite()
                & pl.col("RefPrice").is_finite()
                & pl.col("TickSize").is_finite()
                & (pl.col("RefPrice") > 0.0)
                & (pl.col("TickSize") > 0.0)
            )
            .then(
                pl.col("ToHigh")
                * pl.col("RefPrice")
                / pl.col("TickSize")
            )
            .alias("filllots_at_high_gap_ticks")
        )
        .collect()
    )
    return frame


def _candidate_expressions() -> Mapping[str, pl.Expr]:
    state = pl.col("FillLots_atHigh")
    ratio = pl.col("filllots_at_high_depth_ratio")
    gap_ticks = pl.col("filllots_at_high_gap_ticks")
    fresh = pl.col("_fresh_event").fill_null(False)
    negative = fresh & (state < 0)
    positive = fresh & (state > 0)
    positive_depth = fresh & (ratio >= POSITIVE_DEPTH_RATIO)
    negative_depth = fresh & (ratio <= NEGATIVE_DEPTH_RATIO)
    near = gap_ticks <= NEAR_HIGH_TICKS
    far = gap_ticks >= FAR_HIGH_TICKS
    bull_score = (
        (pl.col("B1_A1B1") >= STRONG_BID_SHARE).cast(pl.UInt8)
        + (
            pl.col("MicroMomentum_30")
            >= MICRO_MOMENTUM_THRESHOLD_BP
        ).cast(pl.UInt8)
        + (pl.col("LOB_NetPressure_30") > 0.0).cast(pl.UInt8)
    )
    bear_score = (
        (pl.col("B1_A1B1") <= WEAK_BID_SHARE).cast(pl.UInt8)
        + (
            pl.col("MicroMomentum_30")
            <= -MICRO_MOMENTUM_THRESHOLD_BP
        ).cast(pl.UInt8)
        + (pl.col("LOB_NetPressure_30") < 0.0).cast(pl.UInt8)
    )

    return {
        "fh_pos_sample": positive,
        "fh_neg_sample": negative,
        "fh_pos_depth2": positive_depth,
        "fh_neg_depth8": negative_depth,
        "fh_pos_near": positive & near,
        "fh_pos_far": positive & far,
        "fh_neg_near": negative & near,
        "fh_neg_far": negative & far,
        "fh_pos_breakout": positive_depth & near & (bull_score >= 2),
        "fh_pos_exhaustion": positive_depth & near & (bear_score >= 2),
        "fh_neg_rejection": negative_depth & far & (bear_score >= 2),
        "fh_neg_absorption": negative_depth & near & (bull_score >= 2),
    }


def select_filllots_at_high_feature_candidates(
    day: EventDaySources,
    *,
    candidate_ids: Sequence[str] | None = None,
) -> Mapping[str, pl.DataFrame]:
    """Return unique canonical keys for the requested fixed candidates."""

    selected_ids = CANDIDATE_IDS if candidate_ids is None else tuple(candidate_ids)
    unknown = sorted(set(selected_ids) - set(CANDIDATE_IDS))
    if unknown:
        raise ValueError(f"unknown FillLots-at-high candidates: {unknown}")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("candidate_ids cannot contain duplicates")

    frame = build_filllots_at_high_candidate_frame(day)
    expressions = _candidate_expressions()
    return {
        candidate_id: (
            frame.filter(expressions[candidate_id].fill_null(False))
            .select(DAY_KEY_COLS)
            .sort(list(DAY_KEY_COLS))
        )
        for candidate_id in selected_ids
    }


__all__ = [
    "CANDIDATE_DESCRIPTIONS",
    "CANDIDATE_IDS",
    "FAR_HIGH_TICKS",
    "FILLLOTS_AT_HIGH_FEATURE_FAMILY_ID",
    "FILLLOTS_AT_HIGH_FEATURE_FAMILY_VERSION",
    "MICRO_MOMENTUM_THRESHOLD_BP",
    "MIN_LARGE_FILL_LOTS",
    "NEAR_HIGH_TICKS",
    "NEGATIVE_DEPTH_RATIO",
    "POSITIVE_DEPTH_RATIO",
    "STRONG_BID_SHARE",
    "WEAK_BID_SHARE",
    "build_filllots_at_high_candidate_frame",
    "select_filllots_at_high_feature_candidates",
]
