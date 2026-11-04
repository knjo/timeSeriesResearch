"""posLimit x daily cap grid on the rebuilt negFill book, risk control = notebook cell 34 verbatim except posLimit / cap.

variant "notebook": stop_dates removed, cap applied every other day (what the notebook cell does today).
variant "graded":   production gate semantics -- daily cap = cap x scale_by_date (stop 0 / reduce 0.1 / tsmc 0.4).
Metrics follow BackTest (taker): PnL = BidPrice1 x (BP - 19.3) / 10, Position = BidPrice1 x 1000, daily sums on traded days,
Sharpe = mean/std x sqrt(252), unit bp = PnL / sum(daily Position), MDD on the period's own cumulative PnL.
Usage: uv run python grid.py <book_dir> <out_dir>
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kevin/Project/HFT"); sys.path.append(str(ROOT))
from src.research.tool import add_stop_loss_prod, first_touch_times
from src.strategy.negFill.stopTradingGate import load_stop_trading_days

BOOK, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
POS_LIMITS = [500, 450, 400, 350, 300, 250]
CAPS = list(range(4000, 7001, 250))
PERIODS = {"2023-2025": ("20230101", "20251231"), "2026": ("20260101", "20991231")}
FEE = 19.3
KEY = ["Date", "QuoteCode", "ChannelSeq"]

book = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(BOOK / "book_chunk*.parquet")))], ignore_index=True)
book['Date'] = pd.to_datetime(book['Date'])
book = book.sort_values(['Date', 'QuoteCode', 'TransTime']).reset_index(drop=True)
print(f"book rows {len(book):,} dates {book.Date.nunique()} ({book.Date.min().date()}..{book.Date.max().date()})", flush=True)

gate = load_stop_trading_days()
stopTradingDay = pd.to_datetime(sorted(gate["stop_dates"]), format="%Y%m%d")
scale_by_date = {str(k): float(v) for k, v in gate["scale_by_date"].items()}

m = book
r1_mask = (((m['d_intraday_return'] > 0.04) & (m['d_intraday_return'] <= 0.1) & (m['dt_major_count'] <= 0)) | (m.MarketCap >= 8 * (10 ** 11)))
brench_mask = ((m['sell_branch_breadth_rank'] >= 0.98) | (m['net_buy_gini'] >= 0.92) | (m['top1_net_buy_amount'] / m['trading_turnover_bf'] > 0.1))
m['PosLots'] = m.groupby(['Date', 'QuoteCode']).BidPrice1.transform('cumcount') + 1
m['Position'] = m.groupby(['Date', 'QuoteCode']).BidPrice1.transform('cumsum') / 10
m['accLots'] = m['PosLots']
base = (~m.Date.isin(stopTradingDay) & (m.accLots < m.avg_askLots1 + m.avg_bidLots1) & (m.RefPrice < 3000) & ~r1_mask & ~brench_mask
        & (m.RemainSeconds >= 3600 + 60 * 25) & (((m.RemainSeconds - 14500).abs() > 300) | (m.Date != pd.to_datetime('2026-10-21')))
        & ((m.hft_strick_makerSpreadBP > -70) | m.hft_strick_makerSpreadBP.isna()) & (m.close_return_20d > -0.2))

touch = None
rows = []
for posLimit in POS_LIMITS:
    swr = m[base & (m.Position < posLimit)].reset_index(drop=True)
    swr['Date'] = swr.Date.dt.strftime('%Y%m%d')
    swr['stopLossProd'] = add_stop_loss_prod(swr, cache_path=OUT / "stop_times_cache.parquet")
    vol = swr['txf_residual_vol_to_1330']
    swr['tpPrice'] = np.floor(swr['BidPrice1'] * (1 - vol * 2) / swr.TickSize) * swr.TickSize + 0.001
    swr['slPrice'] = np.ceil(np.minimum(swr['BidPrice1'] * (1 + vol * 2), swr.RefPrice * 1.08) / swr.TickSize) * swr.TickSize
    if touch is None:   # posLimit 500 = superset; first-touch depends only on the row
        t = first_touch_times(swr, low_col='tpPrice', high_col='slPrice')
        touch = swr[KEY].assign(low_time=t.low_time.values, high_time=t.high_time.values)
    tt = swr[KEY].merge(touch, on=KEY, how='left', validate='one_to_one')
    tp_first = tt.low_time.notna() & (tt.high_time.isna() | (tt.low_time < tt.high_time))
    sl_first = tt.high_time.notna() & ~tp_first
    swr.loc[tp_first.values, 'TakerSell_CloseBP'] = (swr['BidPrice1'] - swr.tpPrice) / swr['BidPrice1'] * 10000
    swr.loc[sl_first.values, 'TakerSell_CloseBP'] = (swr['BidPrice1'] - swr.slPrice - swr.TickSize * 1) / swr['BidPrice1'] * 10000
    swr = swr.sort_values(['Date', 'TransTime']).reset_index(drop=True)
    cum = swr.groupby('Date').BidPrice1.cumsum() / 10
    slp_block = (swr.stopLossProd >= 3) & (swr.RemainSeconds < 12300)
    day_scale = swr.Date.map(scale_by_date).fillna(1.0)
    pnl_row = swr.BidPrice1 * (swr.TakerSell_CloseBP - FEE) / 10
    pos_row = swr.BidPrice1 * 1000
    if posLimit == 500:
        swr.to_parquet(OUT / "risk_rows_pos500_nocap.parquet", index=False)
    for cap in CAPS:
        for variant, keep in (("notebook", (cum < cap) & ~slp_block), ("graded", (cum < cap * day_scale) & ~slp_block)):
            daily = pd.DataFrame({"Date": swr.Date[keep], "PnL": pnl_row[keep], "Position": pos_row[keep]}).groupby("Date").sum()
            for period, (a, b) in PERIODS.items():
                d = daily[(daily.index >= a) & (daily.index <= b)]
                p = d.PnL.to_numpy(); c = np.cumsum(p)
                rows.append({"variant": variant, "period": period, "posLimit_wan": posLimit, "cap_wan": cap, "trade_days": len(d),
                             "sharpe": p.mean() / p.std(ddof=1) * np.sqrt(252) if len(p) > 1 else np.nan,
                             "pnl_wan": p.sum() / 1e4, "unit_bp": p.sum() / d.Position.sum() * 1e4,
                             "mdd_wan": (c - np.maximum.accumulate(c)).min() / 1e4, "avg_position_wan": d.Position.mean() / 1e4,
                             "usage_pct": d.Position.mean() / 1e4 / cap * 100})
    print(f"posLimit {posLimit}: rows {len(swr):,} | tp {int(tp_first.sum()):,} sl {int(sl_first.sum()):,}", flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT / "grid.csv", index=False)
pd.set_option("display.width", 250)
for variant in ("notebook", "graded"):
    for period in PERIODS:
        r = res[(res.variant == variant) & (res.period == period)]
        print(f"\n===== {variant} | {period} =====")
        for metric, fmt in (("sharpe", "{:.2f}"), ("pnl_wan", "{:,.0f}"), ("unit_bp", "{:.1f}"), ("mdd_wan", "{:,.0f}"), ("usage_pct", "{:.0f}")):
            print(f"-- {metric}")
            print(r.pivot(index="posLimit_wan", columns="cap_wan", values=metric).sort_index().map(lambda v: fmt.format(v)).to_string())
