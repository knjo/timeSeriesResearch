"""Key risk/return metrics per posLimit x cap, 2023-04-21..2026-10-30 combined, daily cap = cap x gate scale.
gate: graded = production scale_by_date (comp tiers + 2-day cooldown + TSMC gap); comp = comp tiers only (<-1.5 stop, <-0.7 x0.1); stop = stop days only.
PnL stats on every book date (stop / no-trade days count as 0 PnL); volume on traded days only. Usage: uv run python risk_metrics.py <scratch> [graded|comp|stop]
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append("/home/kevin/Project/HFT")
from src.strategy.negFill.stopTradingGate import load_stop_trading_days, scale_for_comp
from src.research.tool import add_stop_loss_prod
S = Path(sys.argv[1]); MODE = sys.argv[2] if len(sys.argv) > 2 else "graded"; FEE = 19.3
POS = [250, 300, 350, 400, 450, 500]; CAPS = [4000, 5000, 6000, 7000, 8000, 10000]
rr = pd.read_parquet(S / "negfill_grid/risk_rows_pos500_nocap.parquet", columns=["Date", "QuoteCode", "RefPrice", "TransTime", "BidPrice1", "Position", "RemainSeconds", "TakerSell_CloseBP"])
gate = load_stop_trading_days()
sc = {"graded": {str(k): float(v) for k, v in gate["scale_by_date"].items()},
      "comp": {str(r["next_trade_day"]): scale_for_comp(r["comp"]) for r in gate["payload"]["detail"] if scale_for_comp(r["comp"]) < 1},   # base tiers, no cooldown (comp_scale includes it)
      "stop": {}}[MODE]
grid = pd.read_csv(S / "negfill_grid/grid.csv")
cal = sorted(set().union(*[set(pd.read_parquet(f, columns=["Date"]).Date.dt.strftime("%Y%m%d")) for f in glob.glob(str(S / "negfill_book/book_chunk*.parquet"))]))
print("book dates", len(cal), cal[0], cal[-1], "| stop days in range", sum(d in set(map(str, gate["stop_dates"])) for d in cal), flush=True)
rows = []
for pos in POS:
    sw = rr[rr.Position < pos].reset_index(drop=True)
    sw["stopLossProd"] = add_stop_loss_prod(sw, cache_path=S / "negfill_grid/stop_times_cache.parquet")
    sw = sw.sort_values(["Date", "TransTime"]).reset_index(drop=True)
    sw = sw[~((sw.stopLossProd >= 3) & (sw.RemainSeconds < 12300))].reset_index(drop=True)
    cum = sw.groupby("Date").BidPrice1.cumsum() / 10
    day_scale = sw.Date.map(sc).fillna(1.0)
    pnl = sw.BidPrice1 * (sw.TakerSell_CloseBP - FEE) / 10 / 1e4; notional = sw.BidPrice1 / 10   # both 萬
    for cap in CAPS:
        keep = cum < cap * day_scale
        d = pd.DataFrame({"Date": sw.Date[keep], "pnl": pnl[keep], "pos": notional[keep]}).groupby("Date").sum()
        vol = d.pos.mean(); traded = len(d)
        d = d.reindex(cal, fill_value=0.0)   # stop / no-trade days = 0 PnL
        p = d.pnl.to_numpy(); c = np.cumsum(p)
        if MODE == "graded" and cap in (4000, 5000, 6000, 7000):   # parity with grid.csv
            ref = grid[(grid.variant == "graded") & (grid.posLimit_wan == pos) & (grid.cap_wan == cap)]
            assert abs(ref.pnl_wan.sum() - p.sum()) < 0.01, (pos, cap)
        rows.append({"posLimit_wan": pos, "cap_wan": cap, "all_days": len(p), "trade_days": traded, "mdd_wan": (c - np.maximum.accumulate(c)).min(), "mdd_peak": d.index[np.argmax(c[:np.argmin(c - np.maximum.accumulate(c)) + 1])], "mdd_trough": d.index[np.argmin(c - np.maximum.accumulate(c))],
                     "worst_day_wan": p.min(), "pnl_p10_wan": np.percentile(p, 10), "pnl_mean_wan": p.mean(), "pnl_p50_wan": np.percentile(p, 50),
                     "pnl_p90_wan": np.percentile(p, 90), "avg_volume_wan_traded_days": vol, "total_pnl_wan": p.sum(),
                     "sharpe": p.mean() / p.std(ddof=1) * np.sqrt(252), "loss_days_pct": (p < 0).mean() * 100})
    print("pos", pos, "done", flush=True)
res = pd.DataFrame(rows); res.to_csv(S / f"usage_band/risk_metrics_{MODE}.csv", index=False)
pd.set_option("display.width", 250)
print(res.round(1).to_string(index=False))
