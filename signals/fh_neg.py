"""fh_neg — selling inside the intraday-high epoch (FillLots-at-high, short).

    layer 1  envelope   FillLots_atHigh < 0, ToRef in (-1.5%, 7%) (we never short at/above +7%),
                        and on this tick either a different spread pair begins (SpreadPairID
                        changes vs the previous tick of the same stock; opening/closing the same
                        pair does not count) or a formal fill at/above max(20 lots, D-1
                        big_sell_lots).  Carries pair_changed / large_fill / visit_index / high_time.
             condition  amount rank <= 150 plus the two levers that carry information: minutes
                        since the high (TransTime - high_time) and the ToOpen floor.  Grid re-picks
                        time x ToOpen daily; default <= 5 minutes and at/above the open.  Depth /
                        gap / bearish votes / run-up are available but open (None).
    layer 2  points     session window + formal match only (execution filters off: mdRate .25 /
                        rank 100 add ~1 bp per stock-day but drop 40% of them; SpreadPairElapsed
                        removes 86% of fh ticks with no label difference).
    layer 3  day        ridge day selector, bottom 35% excluded; price cap 3000 hard.
    layer 4  model      negFill's 14-feature A/M is computed as a diagnostic only
                        (use_model_gate=False): it lifts 2025 (65.7 -> 68.1 bp) but cuts 2026
                        (64.5 -> 48.9); fh trades as a static book.

Depth ratio and gap are recomputed from store columns (``FillLots_atHigh`` in
tickFeature equals the profile's rebuild bit-for-bit; ``gap_ticks = ToHigh *
RefPrice / TickSize``), so they need no envelope column.
"""

from __future__ import annotations

import polars as pl

from ..framework.events import DAY_KEY_COLS, EventDaySources
from ..framework.profiles.filllots_at_high import build_filllots_at_high_candidate_frame
from ..framework.profiles.tw_equity import NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1
from ..framework.registry import GateSpec, Params, RollingSelection, SignalConfig
from ._short_common import (
    PRICE_AMOUNT_GATE_PARAMS,
    short_day_selector,
    short_labeled_pl,
    short_model_spec,
    short_point_filter_spec,
    short_price_amount_gate,
)


ST_NAME = "fh_neg"
VERSION = "3"
ENVELOPE_COLUMNS = ("pair_changed", "large_fill", "visit_index", "high_time")
ENVELOPE_TO_REF_MIN = -0.015   # never short below -1.5% vs reference ...
ENVELOPE_TO_REF_MAX = 0.07     # ... or at/above +7% (near the limit, 1.08 stop zone)


def _store_col(name: str) -> str:
    return f"{ST_NAME}__{name}"


# --------------------------------------------------------------------------- #
# layer 1 — envelope
# --------------------------------------------------------------------------- #


def select_fh_neg_envelope(day: EventDaySources) -> pl.DataFrame:
    """Keys + raw-stream state for every negative-high-epoch spread-pair change."""

    frame = build_filllots_at_high_candidate_frame(day)
    pairs = (
        day.scan_tick_feature()
        .select(*DAY_KEY_COLS, "SpreadPairID")
        .sort(list(DAY_KEY_COLS))
        .with_columns(
            (pl.col("SpreadPairID") != pl.col("SpreadPairID").shift(1).over("QuoteCode"))
            .fill_null(False)
            .alias("pair_changed")
        )
        .drop("SpreadPairID")
        .collect()
    )
    high_time = (
        day.scan_tick_data()
        .select("QuoteCode", "RecordHigh", "TransTime")
        .group_by(["QuoteCode", "RecordHigh"])
        .agg(pl.col("TransTime").min().alias("high_time"))
        .collect()
    )
    out = frame.join(pairs, on=list(DAY_KEY_COLS), how="left", validate="1:1").join(
        high_time, on=["QuoteCode", "RecordHigh"], how="left", validate="m:1"
    )
    to_ref = (pl.col("BidPrice1") - pl.col("RefPrice")) / pl.col("RefPrice")
    keep = (
        (pl.col("FillLots_atHigh") < 0)
        & (pl.col("pair_changed").fill_null(False) | pl.col("_large_fill").fill_null(False))
        & (pl.col("RefPrice") > 0)
        & (pl.col("BidPrice1") > 0)
        & (to_ref > ENVELOPE_TO_REF_MIN)
        & (to_ref < ENVELOPE_TO_REF_MAX)
    )
    return (
        out.filter(keep.fill_null(False))
        .select(
            *DAY_KEY_COLS,
            pl.col("pair_changed").fill_null(False),
            pl.col("_large_fill").fill_null(False).alias("large_fill"),
            pl.col("_price_pair_visit").cast(pl.UInt32).alias("visit_index"),
            pl.col("high_time"),
        )
        .sort(list(DAY_KEY_COLS))
    )


# --------------------------------------------------------------------------- #
# derived expressions on store rows
# --------------------------------------------------------------------------- #


def depth_ratio_expr() -> pl.Expr:
    depth = pl.col("avg_bidLots1") + pl.col("avg_askLots1")
    return pl.when(depth.is_finite() & (depth > 0.0)).then(pl.col("FillLots_atHigh") / depth).otherwise(None)


def gap_ticks_expr() -> pl.Expr:
    return (
        pl.when(
            pl.col("ToHigh").is_finite()
            & pl.col("RefPrice").is_finite()
            & pl.col("TickSize").is_finite()
            & (pl.col("RefPrice") > 0.0)
            & (pl.col("TickSize") > 0.0)
        )
        .then(pl.col("ToHigh") * pl.col("RefPrice") / pl.col("TickSize"))
        .otherwise(None)
    )


def bear_votes_expr(weak_bid_share: float, momentum_bp: float) -> pl.Expr:
    return (
        (pl.col("B1_A1B1") <= weak_bid_share).cast(pl.UInt8)
        + (pl.col("MicroMomentum_30") <= -momentum_bp).cast(pl.UInt8)
        + (pl.col("LOB_NetPressure_30") < 0.0).cast(pl.UInt8)
    )


def minutes_since_high_expr() -> pl.Expr:
    return (pl.col("TransTime") - pl.col(_store_col("high_time"))).dt.total_seconds() / 60.0


# --------------------------------------------------------------------------- #
# layer 1 — condition (rolled daily)
# --------------------------------------------------------------------------- #


def fh_neg_condition(frame: pl.DataFrame, params: Params) -> pl.Series:
    """Shared price/amount edge + condition shape; ``None`` switches a term off."""

    expr = pl.repeat(True, pl.len(), dtype=pl.Boolean)
    if params["min_since_high_max"] is not None:
        expr = expr & (minutes_since_high_expr() <= params["min_since_high_max"])
    if params["depth_max"] is not None:
        expr = expr & (depth_ratio_expr() <= params["depth_max"])
    if params["gap_min"] is not None:
        expr = expr & (gap_ticks_expr() >= params["gap_min"])
    if params["bear_min"] is not None:
        votes = bear_votes_expr(params["weak_bid_share"], params["momentum_bp"])
        expr = expr & (votes >= params["bear_min"])
    if params["max_visit"] is not None:
        expr = expr & (pl.col(_store_col("visit_index")) <= params["max_visit"])
    if params["require_pair_changed"]:
        expr = expr & pl.col(_store_col("pair_changed"))
    if params["high_vs_open_min"] is not None:
        expr = expr & ((pl.col("RecordHigh") / pl.col("Open") - 1.0) >= params["high_vs_open_min"])
    if params["run_up_min"] is not None:
        # (High - Open) / RefPrice: how far the stock was pushed above its open, in ref terms
        expr = expr & (((pl.col("RecordHigh") - pl.col("Open")) / pl.col("RefPrice")) >= params["run_up_min"])
    if params["to_open_min"] is not None:
        # (Bid - Open) / Open >= x: x = 0 means "at or above the open"
        expr = expr & (pl.col("Open") > 0) & (
            ((pl.col("BidPrice1") - pl.col("Open")) / pl.col("Open")) >= params["to_open_min"]
        )
    if params["high_above_ref"]:
        expr = expr & (pl.col("RecordHigh") > pl.col("RefPrice"))
    shape = frame.select(expr.fill_null(False).alias("gate"))["gate"]
    return shape & short_price_amount_gate(frame, params)


CONDITION = GateSpec(
    gate_id="fh_neg_shape",
    version="7",   # v7: shared price/amount edge reads tickFeature's ToRef
    # Default = the levers that carry information (2026-09-30 level sweeps):
    # <= 5 minutes since the high and bid at/above the open. The ToRef band is
    # part of the envelope (-1.5%, 7%); depth / gap / bearish votes / run-up
    # showed no equal-day effect and stay open so the later layers keep the
    # widest sample. The grid rolls the two informative levers daily.
    params={
        **PRICE_AMOUNT_GATE_PARAMS,
        "to_ref_min": ENVELOPE_TO_REF_MIN,
        "to_ref_max": ENVELOPE_TO_REF_MAX,
        "min_since_high_max": 5.0,
        "depth_max": None,
        "gap_min": None,
        "bear_min": None,
        "weak_bid_share": 0.35,
        "momentum_bp": 0.10,
        "max_visit": None,
        "require_pair_changed": False,
        "high_vs_open_min": None,
        "high_above_ref": False,
        "run_up_min": None,
        "to_open_min": 0.0,
    },
    apply=fh_neg_condition,
    # v6 grid: minutes since the high x ToOpen floor = 4 x 5 = 20 sets.
    grid={
        "min_since_high_max": (5.0, 7.5, 10.0, 15.0),
        "to_open_min": (None, 0.01, 0.0, -0.01, -0.02),
    },
    stage="condition",
)


# layer 2 — session / formal match only; execution filters stay off for fh.
POINT_FILTER = short_point_filter_spec("fh_neg_points", "1")

# layer 3 — 2026-09-30, 421 days: ridge / depth-2 HistGB / depth-3 RF rank next-day
# stock-days about equally (daily IC 0.086-0.089, quintiles monotone); ridge is the default.
DAY_SELECTOR = short_day_selector("fh_neg_day")

# layer 4 — diagnostics only (static book).
MODEL = short_model_spec(
    version="1",
    use_model_gate=False,
    description="Static book: negFill-style A/M models kept as diagnostics only.",
)


FH_NEG = SignalConfig(
    st_name=ST_NAME,
    version=VERSION,
    stage="research",
    description=(
        "Short inside a negative intraday-high epoch when a new spread pair begins "
        "or a large fill prints; shape (minutes since high, ToOpen) is a daily-rolled condition."
    ),
    envelope=select_fh_neg_envelope,
    envelope_required_sources=("tick_data", "tick_feature", "pre_market"),
    envelope_columns=ENVELOPE_COLUMNS,
    universe=NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1,
    gate=CONDITION,
    selection=RollingSelection(window_days=60, objective="stockday_mean_label",
                               label_col="label_residual_bp", min_rows=300, min_stock_days=30),
    selection_pool=short_labeled_pl,
    point_filter=POINT_FILTER,
    day_selector=DAY_SELECTOR,
    model=MODEL,
)


__all__ = [
    "CONDITION",
    "DAY_SELECTOR",
    "ENVELOPE_COLUMNS",
    "FH_NEG",
    "MODEL",
    "POINT_FILTER",
    "bear_votes_expr",
    "depth_ratio_expr",
    "fh_neg_condition",
    "gap_ticks_expr",
    "minutes_since_high_expr",
    "select_fh_neg_envelope",
]
