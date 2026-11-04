"""Read-only tick sources and causal trailing VWAP/minute-last SMA."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from src.features.market_beta import residual_vol_time_scale

US = 1_000_000
MINUTE = 60 * US


def timestamp(date: str, clock: str) -> int:
    return int(np.datetime64(datetime.strptime(date + clock, "%Y%m%d%H:%M:%S"), "us").astype(np.int64))


def trailing_centers(t, price, lots, boundaries):
    """Use [t-30m,t), including no trade at the update timestamp itself."""
    right = np.searchsorted(t, boundaries, side="left")
    left = np.searchsorted(t, boundaries - 30 * MINUTE, side="left")
    volume = np.r_[0.0, np.cumsum(lots)]
    amount = np.r_[0.0, np.cumsum(price * lots)]
    den = volume[right] - volume[left]
    vwap = np.full(len(boundaries), np.nan)
    np.divide(amount[right] - amount[left], den, out=vwap, where=den > 0)
    sma = np.full(len(boundaries), np.nan)
    # Equal weight to the last formal trade of each completed, nonempty minute.
    for j, end in enumerate(boundaries):
        edges = np.arange(end - 30 * MINUTE, end + MINUTE, MINUTE)
        ix = np.searchsorted(t, edges, side="left")
        nonempty = ix[1:] > ix[:-1]
        if np.any(nonempty):
            sma[j] = np.mean(price[ix[1:][nonempty] - 1])
    return vwap, sma


@dataclass
class StockTicks:
    t: np.ndarray
    price: np.ndarray
    lots: np.ndarray
    ask: np.ndarray
    ask_lots: np.ndarray
    close_time: int
    boundaries: np.ndarray
    vwap: np.ndarray
    sma: np.ndarray
    continuous_end: int | None = None

    def events(self, cascade_times=(), average="vwap"):
        valid = (self.price > 0) & (self.lots > 0)
        fill_t = self.t[valid]
        # Raw ticks at a shared timestamp precede scheduled resets. New entries
        # at that timestamp are excluded by strict post-entry search.
        extra = np.asarray(cascade_times, dtype=np.int64)
        t = np.r_[fill_t, self.boundaries, extra]
        kind = np.r_[np.zeros(len(fill_t), dtype=np.int8),
                     np.ones(len(self.boundaries), dtype=np.int8), np.full(len(extra), 2, dtype=np.int8)]
        order = np.lexsort((np.arange(len(t)), kind, t))
        t, kind = t[order], kind[order]
        source = np.searchsorted(self.t, t, side="right") - 1
        # A source before the first tick has no observable book.
        good = source >= 0
        ask = np.where(good, self.ask[np.maximum(source, 0)], 0.0)
        depth = np.where(good, self.ask_lots[np.maximum(source, 0)], 0)
        ask = np.where(depth > 0, ask, 0.0)
        quote_time = np.where(good, self.t[np.maximum(source, 0)], t)
        trade_ix = np.searchsorted(fill_t, t, side="right") - 1
        price = np.where(trade_ix >= 0, self.price[valid][np.maximum(trade_ix, 0)], 0.0)
        # Preserve each real trade's exact FillPrice and exact event-time ask,
        # rather than a later packet sharing its timestamp.
        source_event = np.r_[np.flatnonzero(valid), np.full(len(self.boundaries) + len(extra), -1)][order]
        is_fill = kind == 0
        source[is_fill] = source_event[is_fill]
        price[is_fill] = self.price[source[is_fill]]
        ask[is_fill] = np.where(self.ask_lots[source[is_fill]] > 0, self.ask[source[is_fill]], 0.0)
        quote_time[is_fill] = self.t[source[is_fill]]
        exec_price = ask.copy()
        exec_time = t.copy()
        executable_end = self.close_time if self.continuous_end is None else self.continuous_end
        valid_ask_ix = np.flatnonzero((self.ask > 0) & (self.ask_lots > 0) & (self.t < executable_end))
        ask[t >= executable_end] = 0.0
        exec_price[t >= executable_end] = np.nan
        exec_time[t >= executable_end] = -1
        missing = ask <= 0
        next_ix = np.searchsorted(valid_ask_ix, source[missing], side="right")
        exists = next_ix < len(valid_ask_ix)
        missing_rows = np.flatnonzero(missing)
        exec_time[missing_rows] = -1
        exec_price[missing_rows] = np.nan
        if np.any(exists):
            rows = missing_rows[exists]
            source_ix = valid_ask_ix[next_ix[exists]]
            exec_price[rows] = self.ask[source_ix]
            exec_time[rows] = self.t[source_ix]
        centers = np.r_[np.full(len(fill_t), np.nan), self.vwap if average == "vwap" else self.sma,
                        np.full(len(extra), np.nan)][order]
        scales = residual_vol_time_scale(np.clip((self.close_time - t) / (270 * MINUTE), 0, 1))
        return (t, kind, price, ask, quote_time, exec_price, exec_time, centers, scales)


def load_day(path: Path, date: str, codes: list[str]) -> dict[str, StockTicks]:
    close_t = timestamp(date, "13:30:00")
    open_t = timestamp(date, "09:00:00")
    frame = (pl.scan_parquet(path)
             .filter(pl.col("QuoteCode").is_in(codes)
                     & (pl.col("TransTime").cast(pl.Int64) >= open_t)
                     & (pl.col("TransTime").cast(pl.Int64) <= close_t))
             .select("QuoteCode", "TransTime", "ChannelSeq",
                     *[pl.when(pl.col("TrialMatch") == 0).then(pl.col(c)).otherwise(0).alias(c)
                       for c in ["FillPrice", "FillLots", "AskPrice1", "AskLots1"]])
             .collect().sort("QuoteCode", "TransTime", "ChannelSeq"))
    output = {}
    continuous_end = timestamp(date, "13:25:00")
    boundaries = minute_boundaries(date)
    for (code,), g in frame.partition_by("QuoteCode", as_dict=True, maintain_order=True).items():
        t = g["TransTime"].cast(pl.Int64).to_numpy()
        price = g["FillPrice"].to_numpy().astype(float)
        lots = g["FillLots"].to_numpy().astype(float)
        valid = (price > 0) & (lots > 0)
        if not np.any(valid):
            raise ValueError(f"No trades: {date}/{code}")
        vwap, sma = trailing_centers(t[valid], price[valid], lots[valid], boundaries)
        output[code] = StockTicks(t, price, lots, g["AskPrice1"].to_numpy().astype(float),
                                 g["AskLots1"].to_numpy(), close_t, boundaries, vwap, sma, continuous_end)
    if set(codes) != set(output):
        raise ValueError(f"Missing symbols {date}: {set(codes)-set(output)}")
    return output


def minute_boundaries(date: str):
    """Closing-auction quotes cannot fill a cover before the closing match."""
    return np.arange(timestamp(date, "09:30:00"), timestamp(date, "13:25:00"), MINUTE)
