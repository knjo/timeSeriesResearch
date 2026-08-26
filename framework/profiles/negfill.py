"""Pinned negFill method-validation profiles; not generic defaults."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..dataset import MergeOptions
from ..events import (
    DAY_KEY_COLS,
    EventDaySources,
    EventIndexSpec,
    EventIndexValidationError,
)
from ..features import FeatureManifest, FeatureSpec
from ..labels import (
    Direction,
    DirectionalLabelSpec,
    EntrySpec,
    ExitSpec,
    HedgeSpec,
    LiquidityRole,
    TriggerOp,
    VolSpec,
)
from .tw_equity import (
    NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1 as _NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1,
)


_NEG_FILL_REQUIRED_COLUMNS = (
    "QuoteCode",
    "ChannelSeq",
    "Spread",
    "BidPreMove",
    "FillLots_atLow",
)


def select_negfill_keys(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    source: str = "tickFeature",
) -> pl.LazyFrame:
    """Select canonical negFill tick keys from a tickFeature frame."""

    lazy = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    schema_cols = set(lazy.collect_schema().names())
    missing = [
        column for column in _NEG_FILL_REQUIRED_COLUMNS if column not in schema_cols
    ]
    if missing:
        raise EventIndexValidationError(
            f"{source}: missing required negFill column(s): {missing}"
        )

    return (
        lazy.sort(list(DAY_KEY_COLS))
        .with_columns(
            pl.col("Spread").diff().over("QuoteCode").alias("_spread_diff")
        )
        .filter(
            (pl.col("_spread_diff").abs() > 0.001)
            & (pl.col("BidPreMove").abs() > 0.001)
            & (pl.col("FillLots_atLow") < 0)
        )
        .select(list(DAY_KEY_COLS))
    )


def select_negfill_events(day: EventDaySources) -> pl.LazyFrame:
    """Select negFill event ticks from one framework day source bundle."""

    return select_negfill_keys(
        day.scan_tick_feature(),
        source=str(day.path("tick_feature")),
    )


def negfill_event_index_spec(
    index_path: Path | str,
    *,
    allow_unmanifested_existing: bool = False,
) -> EventIndexSpec:
    """Return the pinned negFill event-index definition."""

    return EventIndexSpec(
        event_id="negfill",
        selector_version="negfill_spread_bid_move_v1",
        selector=select_negfill_events,
        index_path=Path(index_path),
        required_sources=("tick_feature",),
        calendar_source="tick_feature",
        allow_unmanifested_existing=allow_unmanifested_existing,
    )


def negfill_legacy_merge_options_v1() -> MergeOptions:
    """Historical merge behavior, named and isolated for reproducibility."""

    return MergeOptions(
        universe=_NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1,
        require_pre_market=False,
        attach_tick_bar=True,
        require_tick_bar=False,
        tick_bar_alignment="legacy_stock_tick_time",
        attach_txf=True,
        require_txf=False,
        strict_cardinality=True,
        feature_suffix="_right",
        schema_policy="union",
        preserve_tick_data_column_order=True,
    )


def negfill_causal_merge_options_v2() -> MergeOptions:
    """Reference universe with causal completed-slot tickBar alignment."""

    return MergeOptions(
        universe=_NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1,
        require_pre_market=True,
        tick_bar_alignment="causal_completed_slot",
    )


_EXPLICIT_FEATURES = (
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

_DERIVED_FEATURES = (
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


NEGFILL_V1 = FeatureManifest(
    name="negfill",
    version="1",
    features=tuple(
        FeatureSpec(
            name=name,
            role="tick",
            source=(
                "event_transform" if name in _DERIVED_FEATURES else "tick_feature"
            ),
            availability="event",
            transform="precomputed",
        )
        for name in _EXPLICIT_FEATURES + _DERIVED_FEATURES
    )
    + (
        FeatureSpec(
            name="txf_residual_vol_to_1330",
            role="tick",
            source="pre_market_beta_plus_time_clock",
            availability="event",
            transform="precomputed",
        ),
        FeatureSpec(
            name="txf_residual_vol_0900_1330_60d",
            role="tick",
            source="pre_market_beta",
            availability="pre_market",
            transform="identity",
        ),
    ),
)

if len(NEGFILL_V1.features) != 30:  # pragma: no cover - import-time invariant
    raise RuntimeError("NEGFILL_V1 must contain exactly 30 ordered features")


_DEFAULT_HEDGE = HedgeSpec()
_DEFAULT_VOL = VolSpec()

SHORT_TAKER_STOP108_V1 = DirectionalLabelSpec(
    profile_id="negfill.short_taker_stop108.v1",
    direction=Direction.SHORT,
    entry=EntrySpec("BidPrice1", LiquidityRole.TAKER),
    exit=ExitSpec(
        fallback_price_col="Close",
        trigger_price_col="FutureHigh",
        trigger_op=TriggerOp.GT,
        barrier_reference_col="RefPrice",
        barrier_multiplier=1.08,
        tick_size_col="TickSize",
        adverse_slippage_ticks=2.0,
        horizon="first_1.08_barrier_or_13:30",
    ),
    hedge=_DEFAULT_HEDGE,
    normalization=_DEFAULT_VOL,
    allow_exit_hedge_horizon_mismatch=True,
)


__all__ = [
    "NEGFILL_V1",
    "SHORT_TAKER_STOP108_V1",
    "negfill_causal_merge_options_v2",
    "negfill_event_index_spec",
    "negfill_legacy_merge_options_v1",
    "select_negfill_events",
    "select_negfill_keys",
]
