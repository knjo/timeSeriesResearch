"""Reference benchmark label profiles."""

from __future__ import annotations

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


LONG_TAKER_TP109_V1 = DirectionalLabelSpec(
    profile_id="benchmark.long_taker_tp109.v1",
    direction=Direction.LONG,
    entry=EntrySpec("AskPrice1", LiquidityRole.TAKER),
    exit=ExitSpec(
        fallback_price_col="Close",
        trigger_price_col="FutureHigh",
        trigger_op=TriggerOp.GT,
        barrier_reference_col="RefPrice",
        barrier_multiplier=1.09,
        horizon="first_1.09_barrier_or_13:30",
    ),
    hedge=HedgeSpec(),
    normalization=VolSpec(),
    allow_exit_hedge_horizon_mismatch=True,
)


__all__ = ["LONG_TAKER_TP109_V1"]
