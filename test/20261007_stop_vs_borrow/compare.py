"""Paired short-exit replay on an already accepted, one-lot-per-row ledger.

Uses the notebook's tick-touch engine and keeps all entry decisions fixed.
Polars owns the comparison; pandas is accepted at the notebook boundary.
"""
from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import numpy as np
import polars as pl

from src.pipeline_storage import load_pipeline_storage
from src.research.tool import _day_fills, _first_touch

ROOT = Path(__file__).resolve().parents[5]
VOL = "txf_residual_vol_to_1330"
POLICIES = ("tp_sl", "tp_no_sl_borrow")


def prepare_signals(signals, std_span: float = 2.0) -> pl.DataFrame:
    if not np.isfinite(std_span) or std_span <= 0:
        raise ValueError("std_span must be finite and positive")
    frame = signals.clone() if isinstance(signals, pl.DataFrame) else pl.from_pandas(signals)
    required = ["Date", "QuoteCode", "TransTime", "BidPrice1", "RefPrice", "Close", "TickSize", VOL]
    missing = set(required) - set(frame.columns)
    if missing or frame.is_empty():
        raise ValueError(f"Empty ledger or missing columns: {sorted(missing)}")
    optional = [c for c in ("ChannelSeq", "nextDayOpen", "TakerSell_CloseBP") if c in frame.columns]
    frame = frame.select(required + optional).with_columns(
        pl.col("Date").cast(pl.String).str.replace_all("-", "").str.slice(0, 8),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("TransTime").cast(pl.Datetime("us")),
        pl.col("BidPrice1", "RefPrice", "Close", "TickSize", VOL).cast(pl.Float64),
    ).with_row_index("comparison_row")
    frame["Date"].str.to_date("%Y%m%d", strict=True)
    if "nextDayOpen" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("nextDayOpen"))
    frame = frame.with_columns(pl.col("nextDayOpen").cast(pl.Float64).fill_nan(None))
    invalid = frame.filter(pl.any_horizontal([
        (~pl.col(c).is_finite() | (pl.col(c) <= 0)).fill_null(True)
        for c in ("BidPrice1", "RefPrice", "Close", "TickSize")
    ]) | (~pl.col(VOL).is_finite() | (pl.col(VOL) < 0)).fill_null(True)
        | pl.any_horizontal(pl.col("Date", "QuoteCode", "TransTime").is_null()))
    if invalid.height:
        raise ValueError(f"Invalid prices, volatility or keys in {invalid.height} rows")
    if "ChannelSeq" in frame.columns:
        if frame.select("Date", "QuoteCode", "ChannelSeq").unique().height != frame.height:
            raise ValueError("Duplicate event keys in accepted ledger")
    return frame.with_columns(
        (((pl.col("BidPrice1") * (1 - pl.col(VOL) * std_span) / pl.col("TickSize")).floor()
          * pl.col("TickSize")) + 0.001).alias("comparison_tp_price"),
        ((pl.min_horizontal(pl.col("BidPrice1") * (1 + pl.col(VOL) * std_span),
                            pl.col("RefPrice") * 1.08) / pl.col("TickSize")).ceil()
         * pl.col("TickSize")).alias("comparison_sl_price"),
    )


def touch_times(frame: pl.DataFrame, tick_dir: Path, *, verbose: bool = True) -> tuple[pl.DataFrame, list[dict]]:
    """Same strict FillPrice comparisons and post-entry timing as first_touch_times.

    Reuse its primitives to audit stock/day coverage without reading ticks twice.
    A missing tick file or product is an error, never an implicit no-touch.
    """
    parts, sources = [], []
    groups = frame.partition_by("Date", as_dict=True, maintain_order=True)
    for number, ((date,), day) in enumerate(groups.items(), 1):
        path = tick_dir / f"{date}_StockTick.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        stat = path.stat()
        sources.append({"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        fills = _day_fills(tick_dir, date, day["QuoteCode"].unique().to_list())
        for (code,), stock in day.partition_by("QuoteCode", as_dict=True).items():
            ticks = fills.filter(pl.col("QuoteCode") == code)
            if ticks.is_empty() or ticks["TransTime"].max() <= stock["TransTime"].max():
                raise ValueError(f"Missing post-entry tick coverage: {date} {code}")
            times = ticks["TransTime"].to_numpy().astype("datetime64[us]")
            prices = ticks["FillPrice"].to_numpy().astype(float)
            entry = stock["TransTime"].to_numpy().astype("datetime64[us]")
            low = _first_touch(times, prices, entry, stock["comparison_tp_price"].to_numpy(), side="low", strict=True)
            high = _first_touch(times, prices, entry, stock["comparison_sl_price"].to_numpy(), side="high", strict=True)
            parts.append(pl.DataFrame({"comparison_row": stock["comparison_row"], "low_time": low, "high_time": high}))
        if verbose and (number % 25 == 0 or number == len(groups)):
            print(f"Exit replay: {number}/{len(groups)} days", flush=True)
    return frame.join(pl.concat(parts), on="comparison_row", validate="1:1"), sources


def mark_policies(frame: pl.DataFrame) -> pl.DataFrame:
    tp_first = (pl.col("low_time").is_not_null()
                & (pl.col("high_time").is_null() | (pl.col("low_time") < pl.col("high_time")))).fill_null(False)
    return frame.with_columns(
        pl.when(tp_first).then(pl.lit("tp"))
        .when(pl.col("high_time").is_not_null()).then(pl.lit("sl"))
        .otherwise(pl.lit("close")).alias("tp_sl_reason"),
        pl.when(pl.col("low_time").is_not_null()).then(pl.lit("tp"))
        .when(pl.col("Close") > pl.col("RefPrice") * 1.09).then(pl.lit("borrow"))
        .otherwise(pl.lit("close")).alias("tp_no_sl_borrow_reason"),
    )


def resolve_next_open(frame: pl.DataFrame, market_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Keep the collected nextDayOpen; audit/fill from the next session's open_price.

    Only still-open, high-close rows need an overnight label. No forward-fill or
    next-available-price substitution is permitted if that session has no open.
    """
    market_dates = sorted(p.name[:8] for p in market_dir.glob("*_marketData.parquet"))
    borrow = frame.filter(pl.col("tp_no_sl_borrow_reason") == "borrow")
    rows = []
    for (date,), day in borrow.partition_by("Date", as_dict=True).items():
        i = bisect_right(market_dates, date)
        if date not in market_dates or i >= len(market_dates):
            raise ValueError(f"No next-session market file for borrow date {date}")
        next_date = market_dates[i]
        path = market_dir / f"{next_date}_marketData.parquet"
        opens = (pl.scan_parquet(path).select(pl.col("quote_code").cast(pl.String).alias("QuoteCode"),
                                             pl.col("open_price").cast(pl.Float64).alias("market_open"))
                 .filter(pl.col("QuoteCode").is_in(day["QuoteCode"].unique().to_list())).collect())
        if opens["QuoteCode"].n_unique() != opens.height:
            raise ValueError(f"Duplicate market open keys: {path}")
        for (code,), stock in day.partition_by("QuoteCode", as_dict=True).items():
            valid = stock.filter(pl.col("nextDayOpen").is_finite() & (pl.col("nextDayOpen") > 0))["nextDayOpen"]
            source_open = opens.filter(pl.col("QuoteCode") == code)["market_open"]
            if source_open.is_empty() or source_open[0] is None or not np.isfinite(source_open[0]) or source_open[0] <= 0:
                raise ValueError(f"Unpriced borrow exit: {date} {code}, next session {next_date}")
            open_price = float(source_open[0])
            if len(valid) and not np.allclose(valid.to_numpy(), open_price, rtol=0, atol=1e-8):
                raise ValueError(f"nextDayOpen disagrees with next-session market open: {date} {code}")
            rows.append({"Date": date, "QuoteCode": code, "comparison_next_date": next_date,
                         "comparison_next_open": float(valid[0]) if len(valid) else open_price,
                         "filled_missing_rows": stock.height - len(valid), "market_source": str(path)})
    audit = pl.DataFrame(rows, schema={"Date": pl.String, "QuoteCode": pl.String,
        "comparison_next_date": pl.String, "comparison_next_open": pl.Float64,
        "filled_missing_rows": pl.Int64, "market_source": pl.String})
    return frame.join(audit.select(pl.exclude("filled_missing_rows", "market_source")),
                      on=["Date", "QuoteCode"], how="left", validate="m:1"), audit


def price_policies(frame: pl.DataFrame, fee_bp: float = 19.3) -> pl.DataFrame:
    if not np.isfinite(fee_bp) or fee_bp < 0:
        raise ValueError("fee_bp must be finite and nonnegative")
    needed = frame.filter((pl.col("tp_no_sl_borrow_reason") == "borrow") &
        (~pl.col("comparison_next_open").is_finite() | (pl.col("comparison_next_open") <= 0)).fill_null(True))
    if needed.height:
        raise ValueError(f"Unpriced borrow exits: {needed.height}")
    out = frame.with_columns(
        pl.when(pl.col("tp_sl_reason") == "tp").then(pl.col("comparison_tp_price"))
        .when(pl.col("tp_sl_reason") == "sl").then(pl.col("comparison_sl_price") + pl.col("TickSize"))
        .otherwise(pl.col("Close")).alias("tp_sl_exit"),
        pl.when(pl.col("tp_no_sl_borrow_reason") == "tp").then(pl.col("comparison_tp_price"))
        .when(pl.col("tp_no_sl_borrow_reason") == "borrow").then(pl.col("comparison_next_open") * 1.03)
        .otherwise(pl.col("Close")).alias("tp_no_sl_borrow_exit"),
        (pl.col("BidPrice1") * 1000 * fee_bp / 10000).alias("trading_fee_twd"),
        pl.when(pl.col("tp_no_sl_borrow_reason") == "borrow")
        .then(pl.col("comparison_next_open") * 0.03 * 1000).otherwise(0.0).alias("borrow_fee_twd"),
        pl.when(pl.col("tp_no_sl_borrow_reason") == "borrow")
        .then((pl.col("Close") - pl.col("comparison_next_open")) * 1000).otherwise(0.0).alias("overnight_gap_pnl_twd"),
    )
    out = out.with_columns(*[
        ((pl.col("BidPrice1") - pl.col(f"{p}_exit")) / pl.col("BidPrice1") * 10000).alias(f"{p}_bp")
        for p in POLICIES
    ]).with_columns(*[
        ((pl.col("BidPrice1") - pl.col(f"{p}_exit")) * 1000 - pl.col("trading_fee_twd")).alias(f"{p}_pnl_twd")
        for p in POLICIES
    ])
    return out.with_columns(
        (pl.col("tp_no_sl_borrow_pnl_twd") - pl.col("tp_sl_pnl_twd")).alias("delta_pnl_twd"),
    ).sort("comparison_row")


def summarize(ledger: pl.DataFrame, calendar: list[str]) -> dict:
    money = [f"{p}_pnl_twd" for p in POLICIES] + ["delta_pnl_twd", "borrow_fee_twd", "overnight_gap_pnl_twd"]
    daily = (pl.DataFrame({"Date": sorted(set(calendar) | set(ledger["Date"].to_list()))})
             .join(ledger.group_by("Date").agg(pl.col(money).sum(), pl.len().alias("signals")), on="Date", how="left")
             .fill_null(0).sort("Date"))
    rows = []
    for period in ["all"] + sorted(set(d[:4] for d in calendar)):
        days = daily if period == "all" else daily.filter(pl.col("Date").str.starts_with(period))
        trades = ledger if period == "all" else ledger.filter(pl.col("Date").str.starts_with(period))
        for p in POLICIES:
            pnl = days[f"{p}_pnl_twd"].to_numpy()
            equity = np.cumsum(pnl)
            peak = np.maximum.accumulate(np.r_[0.0, equity])[1:]
            vol = np.std(pnl, ddof=1) if len(pnl) > 1 else 0.0
            reasons = trades[f"{p}_reason"]
            notional = trades["BidPrice1"].sum() * 1000
            rows.append({"period": period, "policy": p, "days": len(days), "signals": len(trades),
                "net_pnl_wan": float(pnl.sum() / 10000),
                "net_bp": float(pnl.sum() / notional * 10000) if notional > 0 else None,
                "sharpe": float(np.mean(pnl) / vol * np.sqrt(252)) if vol > 0 else None,
                "mdd_wan": float(np.min(equity - peak) / 10000),
                "worst_day_wan": float(np.min(pnl) / 10000),
                "tp_signals": int((reasons == "tp").sum()), "sl_signals": int((reasons == "sl").sum()),
                "borrow_signals": int((reasons == "borrow").sum()),
                "borrow_fee_wan": float(trades["borrow_fee_twd"].sum() / 10000) if p == POLICIES[1] else 0.0})
    monthly = daily.with_columns(pl.col("Date").str.slice(0, 6).alias("month")).group_by("month").agg(
        (pl.col(money).sum() / 10000).name.suffix("_wan"), pl.col("signals").sum()).sort("month")
    stock_days = ledger.group_by("Date", "QuoteCode").agg(pl.col(money).sum(), pl.len().alias("signals"),
        (pl.col("tp_no_sl_borrow_reason") == "borrow").sum().alias("borrow_signals")).sort("delta_pnl_twd")
    transitions = ledger.group_by("tp_sl_reason", "tp_no_sl_borrow_reason").agg(
        pl.len().alias("signals"), (pl.col(money).sum() / 10000).name.suffix("_wan")).sort("tp_sl_reason", "tp_no_sl_borrow_reason")
    # Fixed two-policy comparison: paired circular five-day block bootstrap.
    delta = daily["delta_pnl_twd"].to_numpy() / 10000
    rng = np.random.default_rng(20261007)
    starts = rng.integers(0, len(delta), size=(2000, (len(delta) + 4) // 5))
    indices = ((starts[..., None] + np.arange(5)) % len(delta)).reshape(2000, -1)[:, :len(delta)]
    interval = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
    return {"ledger": ledger, "summary": pl.DataFrame(rows), "daily": daily, "monthly": monthly,
            "stock_days": stock_days, "transitions": transitions,
            "uncertainty": {"mean_daily_delta_wan": float(delta.mean()),
                "mean_daily_delta_95pct_wan": interval.tolist(), "block_days": 5, "resamples": 2000}}


def compare_exit_policies(signals, *, fee_bp: float = 19.3, std_span: float = 2.0,
                          tick_dir: Path | None = None, market_dir: Path | None = None,
                          verbose: bool = True) -> dict:
    storage = load_pipeline_storage(project_root=ROOT)
    tick_dir = Path(tick_dir) if tick_dir else storage.tick_dir
    market_dir = Path(market_dir) if market_dir else storage.market_dir
    frame = prepare_signals(signals, std_span)
    frame, sources = touch_times(frame, tick_dir, verbose=verbose)
    frame = mark_policies(frame)
    frame, open_audit = resolve_next_open(frame, market_dir)
    ledger = price_policies(frame, fee_bp)
    start, end = ledger["Date"].min(), ledger["Date"].max()
    calendar = sorted(p.name[:8] for p in market_dir.glob("*_marketData.parquet") if start <= p.name[:8] <= end)
    result = summarize(ledger, calendar)
    result.update({"open_audit": open_audit, "tick_sources": sources,
                   "parameters": {"std_span": std_span, "fee_bp": fee_bp, "borrow_fee_rate": 0.03,
                                  "borrow_close_ratio": 1.09, "pnl_date": "entry_date"}})
    return result


def plot_comparison(result: dict):
    import matplotlib.pyplot as plt

    daily = result["daily"].to_pandas()
    dates = daily["Date"].astype("datetime64[ns]")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for p, label in zip(POLICIES, ("TP + SL", "TP + no SL + next-open borrow")):
        pnl = daily[f"{p}_pnl_twd"].to_numpy() / 10000
        equity = pnl.cumsum()
        axes[0].plot(dates, equity, label=label)
        axes[1].plot(dates, equity - np.maximum.accumulate(np.r_[0.0, equity])[1:], label=label)
    axes[2].plot(dates, daily["delta_pnl_twd"].cumsum() / 10000, color="tab:purple", label="No-SL minus SL")
    for ax, title in zip(axes, ("Cumulative net PnL", "Drawdown", "Cumulative paired difference")):
        ax.set_title(title + " (entry-date attribution)")
        ax.set_ylabel("TWD 10,000")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    return fig
