"""Shared Taiwan-equity universe profiles."""

from __future__ import annotations

import polars as pl

from ..dataset import DatasetContractError, UniverseSpec


def _nextday_allow_day_trade(frame: pl.DataFrame) -> pl.DataFrame:
    candidates = [
        column
        for column in (
            "nextday_allow_day_trade_mark",
            "nextday_allow_day_trade_mark_pm",
        )
        if column in frame.columns
    ]
    if not candidates:
        raise DatasetContractError(
            "nextday_allow_day_trade_v1 requires "
            "nextday_allow_day_trade_mark from preMarket"
        )
    if len(candidates) == 2:
        disagreement = frame.filter(
            pl.col(candidates[0]).is_not_null()
            & pl.col(candidates[1]).is_not_null()
            & (pl.col(candidates[0]) != pl.col(candidates[1]))
        )
        if not disagreement.is_empty():
            raise DatasetContractError(
                "ambiguous next-day day-trade eligibility columns disagree"
            )
    return frame.filter(
        pl.any_horizontal([pl.col(column) == "X" for column in candidates])
    )


NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1 = UniverseSpec(
    "tw_equity.nextday_allow_day_trade", "1", _nextday_allow_day_trade
)


__all__ = ["NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1"]
