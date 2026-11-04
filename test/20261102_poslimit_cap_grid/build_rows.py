"""Per-row risk book (no daily cap) for a posLimit superset: grid.py's cell-34 filters + 2σ TP/SL, saved for cap/posLimit sweeps.
stopLossProd is NOT final here (it depends on posLimit) -- recompute per posLimit downstream.
Usage: uv run python build_rows.py <book_dir> <out_parquet> <posLimit_wan> [keep_stop]
keep_stop: keep the gate's stop_dates (for re-tuning the stop threshold); default excludes them like cell 34.
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kevin/Project/HFT"); sys.path.append(str(ROOT))
from src.research.tool import first_touch_times
from src.strategy.negFill.stopTradingGate import load_stop_trading_days

BOOK, OUT, POS = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]); KEEP_STOP = sys.argv[4:] == ["keep_stop"]
book = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(BOOK / "book_chunk*.parquet")))], ignore_index=True)
book['Date'] = pd.to_datetime(book['Date'])
m = book.sort_values(['Date', 'QuoteCode', 'TransTime']).reset_index(drop=True)
stopTradingDay = pd.to_datetime([] if KEEP_STOP else sorted(load_stop_trading_days()["stop_dates"]), format="%Y%m%d")
r1_mask = (((m['d_intraday_return'] > 0.04) & (m['d_intraday_return'] <= 0.1) & (m['dt_major_count'] <= 0)) | (m.MarketCap >= 8 * (10 ** 11)))
brench_mask = ((m['sell_branch_breadth_rank'] >= 0.98) | (m['net_buy_gini'] >= 0.92) | (m['top1_net_buy_amount'] / m['trading_turnover_bf'] > 0.1))
m['PosLots'] = m.groupby(['Date', 'QuoteCode']).BidPrice1.transform('cumcount') + 1
m['Position'] = m.groupby(['Date', 'QuoteCode']).BidPrice1.transform('cumsum') / 10
m['accLots'] = m['PosLots']
base = (~m.Date.isin(stopTradingDay) & (m.accLots < m.avg_askLots1 + m.avg_bidLots1) & (m.RefPrice < 3000) & ~r1_mask & ~brench_mask
        & (m.RemainSeconds >= 3600 + 60 * 25) & (((m.RemainSeconds - 14500).abs() > 300) | (m.Date != pd.to_datetime('2026-10-21')))
        & ((m.hft_strick_makerSpreadBP > -70) | m.hft_strick_makerSpreadBP.isna()) & (m.close_return_20d > -0.2))
swr = m[base & (m.Position < POS)].reset_index(drop=True)
del book, m
swr['Date'] = swr.Date.dt.strftime('%Y%m%d')
vol = swr['txf_residual_vol_to_1330']
swr['tpPrice'] = np.floor(swr['BidPrice1'] * (1 - vol * 2) / swr.TickSize) * swr.TickSize + 0.001
swr['slPrice'] = np.ceil(np.minimum(swr['BidPrice1'] * (1 + vol * 2), swr.RefPrice * 1.08) / swr.TickSize) * swr.TickSize
t = first_touch_times(swr, low_col='tpPrice', high_col='slPrice')
tp_first = t.low_time.notna() & (t.high_time.isna() | (t.low_time < t.high_time))
sl_first = t.high_time.notna() & ~tp_first
swr.loc[tp_first.values, 'TakerSell_CloseBP'] = (swr['BidPrice1'] - swr.tpPrice) / swr['BidPrice1'] * 10000
swr.loc[sl_first.values, 'TakerSell_CloseBP'] = (swr['BidPrice1'] - swr.slPrice - swr.TickSize * 1) / swr['BidPrice1'] * 10000
swr = swr.sort_values(['Date', 'TransTime']).reset_index(drop=True)
swr[["Date", "QuoteCode", "ChannelSeq", "RefPrice", "TransTime", "BidPrice1", "Position", "RemainSeconds", "TakerSell_CloseBP"]].to_parquet(OUT, index=False)
print(f"rows {len(swr):,} | tp {int(tp_first.sum()):,} sl {int(sl_first.sum()):,} -> {OUT}", flush=True)
