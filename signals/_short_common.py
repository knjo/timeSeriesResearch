"""Shared building blocks for taker-short signals (negFill, fh_neg, ...).

Every number here is the deployed negFill contract from
``src/strategy/negFill/modelUpdate.py``; signals that share the short label,
session window, feature block and price/amount condition import from here so
there is exactly one implementation to keep in parity.  Each block is tagged
with the layer that consumes it.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
import polars as pl

from ..framework.registry import (
    CellThresholdSpec,
    DaySelectorSpec,
    GateSpec,
    ModelSpec,
    Params,
    RiskModelSpec,
)


TARGET_LABEL = "BidPrice1"
STOP_RATIO = 1.08
STOP_TICKS = 2
RAW_LABEL = "TakerSell_CloseBP"
RESIDUAL_LABEL = "residual_TakerSell_CloseBP"
TIMING_TARGET = "TakerSell_CloseBP_net"
SELECTION_TARGET = "TakerSell_CloseBP_netM"
SESSION_OPEN = "09:00:30"
SESSION_CLOSE = "12:30:00"
LABEL_ABS_MAX = 10000.0
HARD_PRICE_CAP = 3000.0

SHORT_FEATURE_COLUMNS = (
    "ToLow", "ToHigh", "Low_High", "ToRef", "ToOpen", "B1_A1B1", "B1_B1B5", "A1_A1A5",
    "B45_AB45", "B1_B12", "A1_A12", "RemainSeconds", "MD_ElaspeTime_30_re", "MD_L1Rate_30",
)

PRICE_AMOUNT_GATE_PARAMS = {"to_ref_min": -0.015, "to_ref_max": 0.05, "rank_max": 150, "to_ref_source": "mid"}

# ToRef conventions.  "mid" = tickFeature's delivered ``ToRef`` ((Mid - Ref)/Ref).  "bid1" = the
# deployed modelUpdate convention ((Bid1 - Ref)/Ref), kept for negfill so its thresholds match
# production; it is materialized under its own name (``ToRef_bid1`` / ``ToOpen_bid1``) and never
# overwrites the source column.
TO_REF_SOURCES = ("mid", "bid1")
TO_REF_BID1 = "ToRef_bid1"
TO_OPEN_BID1 = "ToOpen_bid1"


# --------------------------------------------------------------------------- #
# layer 1 — condition (polars) and the pool it is scored on
# --------------------------------------------------------------------------- #


def to_ref_expr(source: str = "mid") -> pl.Expr:
    """``mid``: tickFeature's ``ToRef`` as delivered; ``bid1``: the deployed (Bid1 − Ref)/Ref convention."""

    if source not in TO_REF_SOURCES:
        raise ValueError(f"unknown to_ref_source {source!r}; expected one of {TO_REF_SOURCES}")
    if source == "bid1":
        return ((pl.col("BidPrice1") - pl.col("RefPrice")) / pl.col("RefPrice")).round(6)
    return pl.col("ToRef")


def short_price_amount_gate(frame: pl.DataFrame, params: Params) -> pl.Series:
    """PriceCondition & AmountCondition from the negFill notebook / modelUpdate.

    ``params["to_ref_source"]`` picks the ToRef convention: ``mid`` (tickFeature
    as delivered) or ``bid1`` (modelUpdate's recompute; median gap 0.17%).
    """

    to_ref = to_ref_expr(str(params.get("to_ref_source", "mid")))
    expr = (
        (pl.col("BidPrice1") > 0)
        & (to_ref > params["to_ref_min"])
        & (to_ref < params["to_ref_max"])
        & (
            (pl.col("day_amount_rank") <= params["rank_max"])
            | (pl.col("AmountRank_canDayTrade") <= params["rank_max"])
        )
    )
    return frame.select(expr.fill_null(False).alias("gate"))["gate"]


def short_taker_label_pl(frame: pl.DataFrame) -> pl.DataFrame:
    """Taker short at Bid1 to close, 1.08×Ref stop (−2 ticks); TXF-beta residual over vol → z."""

    bid = pl.col(TARGET_LABEL)
    raw = (bid - pl.col("Close")) / bid * 10000
    stopped = (bid - pl.col("RefPrice") * STOP_RATIO - pl.col("TickSize") * STOP_TICKS) / bid * 10000
    raw = pl.when(pl.col("FutureHigh") > pl.col("RefPrice") * STOP_RATIO).then(stopped).otherwise(raw)
    residual = (raw + pl.col("txf_beta_60d") * pl.col("txf_to_1330_bp")) / (
        pl.col("txf_residual_vol_to_1330") * 1e4
    )
    return frame.with_columns(
        raw.alias(RAW_LABEL),
        residual.alias(RESIDUAL_LABEL),
        residual.alias("label_residual_bp"),
    )


def short_labeled_pl(frame: pl.DataFrame) -> pl.DataFrame:
    """Candidate rows -> labeled rows with a valid label (layer-1 scoring pool, layer-3 target)."""

    if frame.is_empty():
        return frame
    labeled = short_taker_label_pl(frame)
    valid = (
        (pl.col(TARGET_LABEL) > 0)
        & (pl.col(RAW_LABEL).abs() < LABEL_ABS_MAX)
        & pl.col("label_residual_bp").is_finite()
    )
    return labeled.filter(valid.fill_null(False))


# --------------------------------------------------------------------------- #
# layer 2 — fixed tick-level removals (polars)
# --------------------------------------------------------------------------- #


POINT_FILTER_PARAMS = {
    "session_open": SESSION_OPEN,     # keep ticks strictly after
    "session_close": SESSION_CLOSE,   # keep ticks strictly before
    "md_l1rate_min": None,            # OTC | MD_L1Rate_30 > x  (None = off)
    "elapsed_min": None,              # SpreadPairElapsed > x    (None = off)
    "rank_max": None,                 # amount rank <= x         (None = off)
}


def _parse_time(value: object) -> _dt.time:
    if isinstance(value, _dt.time):
        return value
    return _dt.time.fromisoformat(str(value))


def short_point_filter(frame: pl.DataFrame, params: Params) -> pl.Series:
    """Session window, formal matches, a live bid, plus optional signal-specific tick filters."""

    time_of_day = pl.col("TransTime").dt.time()
    expr = (
        (pl.col(TARGET_LABEL) > 0)
        & (pl.col("TrialMatch") == 0)
        & (time_of_day > pl.lit(_parse_time(params["session_open"])))
        & (time_of_day < pl.lit(_parse_time(params["session_close"])))
    )
    if params["md_l1rate_min"] is not None:
        expr = expr & ((pl.col("OTC") == 1) | (pl.col("MD_L1Rate_30") > params["md_l1rate_min"]))
    if params["elapsed_min"] is not None:
        expr = expr & (pl.col("SpreadPairElapsed") > params["elapsed_min"])
    if params["rank_max"] is not None:
        expr = expr & (
            (pl.col("AmountRank_canDayTrade") <= params["rank_max"])
            | (pl.col("day_amount_rank") <= params["rank_max"])
        )
    return frame.select(expr.fill_null(False).alias("gate"))["gate"]


def _legacy_day_rules(frame: pl.DataFrame, p: Params) -> pl.Series:
    """The seven deployed day-level rules (modelUpdate ``suspended_list``); baseline only, never configured as hard rules."""

    r1 = ((pl.col("d_intraday_return") > p["r1_return_low"]) & (pl.col("d_intraday_return") <= p["r1_return_high"])
          & (pl.col("dt_major_count") <= p["r1_dt_major_max"])).fill_null(False)
    cap = (pl.col("MarketCap") >= p["market_cap_max"]).fill_null(False)
    top1 = pl.col("top1_net_buy_amount") / pl.col("trading_turnover_bf")
    brench = ((pl.col("sell_branch_breadth_rank") >= p["sell_breadth_rank_min"]) | (pl.col("net_buy_gini") >= p["net_buy_gini_min"])
              | (top1 > p["top1_buy_share_min"])).fill_null(False)
    hard = (pl.col("RefPrice") <= p["ref_price_max"]) & ((pl.col("hft_strick_makerSpreadBP") > p["maker_spread_bp_min"]) | pl.col("hft_strick_makerSpreadBP").is_null())
    return frame.select((~r1 & ~cap & ~brench & hard.fill_null(False)).alias("gate"))["gate"]


LEGACY_DAY_RULES = GateSpec(
    gate_id="legacy_day_rules",
    version="1",
    params={"r1_return_low": 0.04, "r1_return_high": 0.10, "r1_dt_major_max": 0, "market_cap_max": 8e11, "sell_breadth_rank_min": 0.98,
            "net_buy_gini_min": 0.92, "top1_buy_share_min": 0.10, "ref_price_max": HARD_PRICE_CAP, "maker_spread_bp_min": -70.0},
    apply=_legacy_day_rules,
    stage="hard_rule",
)



def short_point_filter_with_day_rules(frame: pl.DataFrame, params: Params) -> pl.Series:
    """Tick filters plus the seven deployed day-level rules, applied row by row (the D-1 columns sit on every row)."""

    return short_point_filter(frame, params) & _legacy_day_rules(frame, params)


def short_point_filter_spec(gate_id: str, version: str = "1", *, day_rules: bool = False, **overrides) -> GateSpec:
    """``day_rules=True`` folds the seven deployed day-level rules into the fixed layer-2 removal."""

    unknown = sorted(set(overrides) - set(POINT_FILTER_PARAMS))
    if unknown:
        raise ValueError(f"unknown point filter parameter(s) {unknown}")
    params = {**POINT_FILTER_PARAMS, **overrides}
    if day_rules:
        params.update(LEGACY_DAY_RULES.params)
    return GateSpec(
        gate_id=gate_id,
        version=version,
        params=params,
        apply=short_point_filter_with_day_rules if day_rules else short_point_filter,
        stage="point_filter",
    )


# --------------------------------------------------------------------------- #
# layer 3 — day selector: targets, weight, hard rule, risk models
# --------------------------------------------------------------------------- #


def _price_cap(frame: pl.DataFrame, params: Params) -> pl.Series:
    return frame.select((pl.col("RefPrice") <= params["ref_price_max"]).fill_null(False).alias("gate"))["gate"]


PRICE_CAP = GateSpec(
    gate_id="price_cap",
    version="1",
    params={"ref_price_max": HARD_PRICE_CAP},
    apply=_price_cap,
    stage="hard_rule",
)


def short_day_labels(rows: pl.DataFrame) -> pl.DataFrame:
    """Candidate rows -> one row per stock-day with the ``ds_*`` risk targets (realized on day D).

    ``ds_*`` names are deliberately distinct from the strategy labels
    (``TakerSell_CloseBP`` / ``residual_TakerSell_CloseBP`` / ``label_residual_bp``):
    they describe the stock's day, not the short's PnL.  ``ds_pnl_bp`` is the
    stop-adjusted raw bp of the candidate points, kept for evaluation only.
    """

    labeled = short_taker_label_pl(rows)
    day_high = pl.max_horizontal(pl.col("RecordHigh"), pl.col("FutureHigh")).max()
    day_low = pl.min_horizontal(pl.col("RecordLow"), pl.col("FutureLow")).min()
    return (
        labeled.sort(["QuoteCode", "TransTime"])
        .group_by("QuoteCode", maintain_order=True)
        .agg(
            pl.col("Date").first(),
            day_high.alias("_high"), day_low.alias("_low"),
            pl.col("Open").first().alias("_open"), pl.col("RefPrice").first().alias("_ref"),
            pl.col(RAW_LABEL).filter(pl.col(RAW_LABEL).abs() < LABEL_ABS_MAX).mean().alias("ds_pnl_bp"),
        )
        .with_columns(
            ((pl.col("_high") - pl.col("_open")) / pl.col("_ref")).alias("ds_squeeze"),          # how far the day was pushed above its open
            (pl.col("_high") > pl.col("_ref") * STOP_RATIO).cast(pl.Float64).alias("ds_stop_hit"),  # the 1.08 stop would have fired
            pl.when(pl.col("_high") > pl.col("_low"))
            .then((pl.col("_high") - pl.col("_open")) / (pl.col("_high") - pl.col("_low")))
            .otherwise(None).alias("ds_open_pos"),                                               # 0 = opened at the high, 1 = opened at the low
        )
        .select("Date", "QuoteCode", "ds_squeeze", "ds_stop_hit", "ds_open_pos", "ds_pnl_bp")
    )


PARTICIPATION = 0.003       # share of D-1 turnover the book can take
POSITION_CAP_NTD = 5e6      # 500萬 per-stock position cap (BackTest ``Position < 500``)


def short_day_weight(frame: pl.DataFrame) -> pl.Series:
    """Sample weight = clip(D-1 turnover × 0.3% / 500萬, 0, 1): the share of the cap the book would fill."""

    return (pl.col("trading_turnover") * PARTICIPATION / POSITION_CAP_NTD).clip(0.0, 1.0).fill_null(0.0).alias("weight").pipe(lambda e: frame.select(e)["weight"])


# Shallow gradient-boosted trees (depth 3): the patterns the hand rules encode are tail values with
# interactions (e.g. "up 4-10% yesterday AND no major buyer"), which a linear model cannot express.
TREE_HYPER = {"max_depth": 3, "max_iter": 200, "learning_rate": 0.05, "min_samples_leaf": 50, "l2": 1.0}

SHORT_RISK_MODELS = (
    RiskModelSpec(model_id="stop", target="ds_stop_hit", task="classification", family="hist_gb", threshold=0.10,
                  direction="exclude_above", hyper=TREE_HYPER, description="P(High > Ref×1.08) tomorrow"),
    RiskModelSpec(model_id="squeeze", target="ds_squeeze", task="regression", family="hist_gb", threshold=0.03,
                  direction="exclude_above", hyper=TREE_HYPER, description="E[(High − Open)/Ref] tomorrow"),
    RiskModelSpec(model_id="open_pos", target="ds_open_pos", task="regression", family="hist_gb", threshold=0.70,
                  direction="exclude_above", hyper=TREE_HYPER, description="E[(High − Open)/(High − Low)] tomorrow"),
)


def short_day_selector(
    selector_id: str,
    *,
    models: tuple[RiskModelSpec, ...] = SHORT_RISK_MODELS,
    hard_rules: GateSpec | None = PRICE_CAP,
    baseline_rules: GateSpec | None = LEGACY_DAY_RULES,
    window_days: int = 60,
) -> DaySelectorSpec:
    return DaySelectorSpec(
        selector_id=selector_id,
        version="2",
        labels=short_day_labels,
        weight=short_day_weight,
        models=models,
        hard_rules=hard_rules,
        baseline_rules=baseline_rules,
        window_days=window_days,
        description=f"{len(models)} risk models on D-1 stock-day features, turnover-weighted, absolute thresholds; price cap {HARD_PRICE_CAP:.0f} hard.",
    )


# --------------------------------------------------------------------------- #
# layer 4 — model steps (pandas; exact port of modelUpdate.py)
# --------------------------------------------------------------------------- #


def short_label_pd(df: pd.DataFrame) -> pd.DataFrame:
    # ToRef / ToOpen are tickFeature columns and are NOT recomputed here; see short_label_bid1_pd for the deployed convention.
    out = df[df[TARGET_LABEL] * df["BidPrice1"] > 0].copy()
    out[RAW_LABEL] = (out[TARGET_LABEL] - out["Close"]) / out[TARGET_LABEL] * 10000
    stop_mask = out["FutureHigh"] > out["RefPrice"] * STOP_RATIO
    out.loc[stop_mask, RAW_LABEL] = (
        (out.loc[stop_mask, TARGET_LABEL] - out.loc[stop_mask, "RefPrice"] * STOP_RATIO
         - out.loc[stop_mask, "TickSize"] * STOP_TICKS)
        / out.loc[stop_mask, TARGET_LABEL] * 10000
    )
    out[RESIDUAL_LABEL] = (
        out[RAW_LABEL] + out["txf_beta_60d"] * out["txf_to_1330_bp"]
    ) / (out["txf_residual_vol_to_1330"] * 1e4)
    return out


def short_label_bid1_pd(df: pd.DataFrame) -> pd.DataFrame:
    """Label plus the deployed Bid1-based ToRef / ToOpen under their own names (source columns untouched)."""

    out = short_label_pd(df)
    out[TO_REF_BID1] = ((out["BidPrice1"] - out.RefPrice) / out.RefPrice).round(6)
    out[TO_OPEN_BID1] = ((out["BidPrice1"] - out.Open) / out.Open).round(6)
    return out


def short_train_filter_pd(df: pd.DataFrame) -> pd.DataFrame:
    """Label-validity guard (needs the label, so it cannot be a point filter)."""

    return (
        df[(df[RAW_LABEL].abs() < LABEL_ABS_MAX)]
        .sort_values(["Date", "QuoteCode", "TransTime"])
        .reset_index(drop=True)
    )


def short_features_pd(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    out["midEdge_300sBP"] = out["midEdge_300sBP"] * -1
    out[TIMING_TARGET] = (
        out[RESIDUAL_LABEL] - out.groupby(["Date", "QuoteCode"])[RESIDUAL_LABEL].transform("mean")
    )
    out[SELECTION_TARGET] = (
        out[RESIDUAL_LABEL] - out.groupby(["Date"])[RESIDUAL_LABEL].transform("mean")
    )
    out["B2_Last"] = out.BidLots2 / out.avg_bidLots1
    out["A2_Last"] = out.AskLots2 / out.avg_askLots1
    out["B1_B12"] = out.BidLots1 / (out.BidLots1 + out.BidLots2)
    out["A1_A12"] = out.AskLots1 / (out.AskLots1 + out.AskLots2)
    out["B45_AB45"] = (out.BidLots4 + out.BidLots5) / (
        out.AskLots4 + out.AskLots5 + out.BidLots4 + out.BidLots5
    )
    out["Total_Last"] = out.TotalFillLots / out.last_fillLots
    out["LOB_BidVelocity_30_re"] = out.LOB_BidVelocity_30.abs() / out.avg_bidLots1
    out["LOB_AskVelocity_30_re"] = out.LOB_AskVelocity_30.abs() / out.avg_askLots1
    out["FillLots_atLow_re"] = out.FillLots_atLow / (out.avg_bidLots1 + out.avg_askLots1)
    out["FillLots_atHigh_re"] = out.FillLots_atHigh / (out.avg_bidLots1 + out.avg_askLots1)
    out["L1_SellBiggestLots_30_re"] = out.L1_SellBiggestLots_30.abs() / out.big_sell_lots
    out["L1_BuyBiggestLots_30_re"] = out.L1_BuyBiggestLots_30.abs() / out.big_buy_lots
    out["MD_ElaspeTime_30_re"] = (out.MD_ElaspeTime_30 + 1).apply(np.log)
    out["MD_L1Rate_30_re"] = out.MD_L1Rate_30
    out["IA_BuyImpact_30_re"] = (out.IA_BuyImpact_30.abs() + 1).apply(np.log)
    out["IA_SellImpact_30_re"] = (out.IA_SellImpact_30.abs() + 1).apply(np.log)
    out["LOB_NetPressure_30_re"] = (out.LOB_NetPressure_30.abs() + 1).apply(np.log)
    out["QL_Asymmetry_re"] = (out.QL_Asymmetry.abs() + 1).apply(np.log)
    out["OTC"] = out["OTC"] * 1
    return out


def short_model_spec(*, to_ref_source: str = "mid", **overrides) -> ModelSpec:
    """negFill's 14-feature Ridge A(60d)/M(20d) with 16-cell prior-day thresholds.

    ``to_ref_source="bid1"`` swaps the ToRef / ToOpen features and the threshold
    cell column to the deployed Bid1 convention (``ToRef_bid1`` / ``ToOpen_bid1``).
    """

    if to_ref_source not in TO_REF_SOURCES:
        raise ValueError(f"unknown to_ref_source {to_ref_source!r}")
    bid1 = to_ref_source == "bid1"
    feature_columns = tuple(
        {"ToRef": TO_REF_BID1, "ToOpen": TO_OPEN_BID1}.get(c, c) if bid1 else c for c in SHORT_FEATURE_COLUMNS
    )
    base = dict(
        label=short_label_bid1_pd if bid1 else short_label_pd,
        train_filter=short_train_filter_pd,
        features=short_features_pd,
        feature_columns=feature_columns,
        threshold=CellThresholdSpec(ref_col=TO_REF_BID1) if bid1 else CellThresholdSpec(),
        timing_target=TIMING_TARGET,
        selection_target=SELECTION_TARGET,
        raw_label_col=RAW_LABEL,
        timing_lookback_days=60,
        selection_lookback_days=20,
        alpha=10.0,
        max_position_per_stock=500,
        feature_screening=True,
    )
    base.update(overrides)
    return ModelSpec(**base)


__all__ = [
    "HARD_PRICE_CAP",
    "LABEL_ABS_MAX",
    "LEGACY_DAY_RULES",
    "SHORT_RISK_MODELS",
    "POINT_FILTER_PARAMS",
    "PRICE_AMOUNT_GATE_PARAMS",
    "PRICE_CAP",
    "RAW_LABEL",
    "RESIDUAL_LABEL",
    "SELECTION_TARGET",
    "SESSION_CLOSE",
    "SESSION_OPEN",
    "SHORT_FEATURE_COLUMNS",
    "STOP_RATIO",
    "STOP_TICKS",
    "TARGET_LABEL",
    "TO_OPEN_BID1",
    "TO_REF_BID1",
    "TO_REF_SOURCES",
    "TIMING_TARGET",
    "short_day_labels",
    "short_day_selector",
    "short_day_weight",
    "short_features_pd",
    "short_label_bid1_pd",
    "short_label_pd",
    "short_labeled_pl",
    "short_model_spec",
    "short_point_filter",
    "short_point_filter_spec",
    "short_point_filter_with_day_rules",
    "short_price_amount_gate",
    "short_taker_label_pl",
    "short_train_filter_pd",
    "to_ref_expr",
]
