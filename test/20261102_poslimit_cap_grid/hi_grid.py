"""posLimit 300-1000 x cap 5000-10000 under the comp-only gate (comp<-1.5 stop, -1.5..-0.7 cap x0.1; no cooldown, no TSMC).
PnL stats on every book date (no-trade days = 0); volume on traded days; usage = normal-day (comp scale 1) mean notional / cap.
Usage: uv run python hi_grid.py <scratch> <rows_parquet> <out_csv>
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append("/home/kevin/Project/HFT")
from src.strategy.negFill.stopTradingGate import load_stop_trading_days, scale_for_comp
from src.research.tool import add_stop_loss_prod

S, ROWS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); FEE = 19.3
POS = list(range(300, 1001, 100)); CAPS = list(range(5000, 10001, 500))
rr = pd.read_parquet(ROWS)
gate = load_stop_trading_days()
sc = {str(r["next_trade_day"]): scale_for_comp(r["comp"]) for r in gate["payload"]["detail"] if scale_for_comp(r["comp"]) < 1}   # base tiers, no cooldown (comp_scale includes it)
cal = sorted(set().union(*[set(pd.read_parquet(f, columns=["Date"]).Date.dt.strftime("%Y%m%d")) for f in glob.glob(str(S / "negfill_book/book_chunk*.parquet"))]))
cal_scale = pd.Series([sc.get(d, 1.0) for d in cal], index=cal)
ref = pd.read_csv(S / "usage_band/risk_metrics_comp.csv")
rows = []
for pos in POS:
    sw = rr[rr.Position < pos].reset_index(drop=True)
    sw["stopLossProd"] = add_stop_loss_prod(sw, cache_path=S / "negfill_grid/stop_times_cache.parquet")   # depends on posLimit
    sw = sw.sort_values(["Date", "TransTime"]).reset_index(drop=True)
    sw = sw[~((sw.stopLossProd >= 3) & (sw.RemainSeconds < 12300))].reset_index(drop=True)
    cum = sw.groupby("Date").BidPrice1.cumsum() / 10
    day_scale = sw.Date.map(sc).fillna(1.0)
    pnl = sw.BidPrice1 * (sw.TakerSell_CloseBP - FEE) / 10 / 1e4; notional = sw.BidPrice1 / 10   # 萬
    for cap in CAPS:
        keep = cum < cap * day_scale
        d = pd.DataFrame({"Date": sw.Date[keep], "pnl": pnl[keep], "pos": notional[keep]}).groupby("Date").sum()
        vol = d.pos.mean()
        d = d.reindex(cal, fill_value=0.0)
        p = d.pnl.to_numpy(); c = np.cumsum(p); dd = c - np.maximum.accumulate(c)
        normal = d.pos[cal_scale == 1.0].to_numpy()
        r = {"posLimit_wan": pos, "cap_wan": cap, "usage_normal_pct": normal.mean() / cap * 100, "hit_cap_pct": (normal >= cap * 0.97).mean() * 100,
             "total_pnl_wan": p.sum(), "mdd_wan": dd.min(), "mdd_trough": cal[int(np.argmin(dd))], "worst_day_wan": p.min(),
             "pnl_p10_wan": np.percentile(p, 10), "pnl_mean_wan": p.mean(), "pnl_p90_wan": np.percentile(p, 90),
             "avg_volume_wan_traded_days": vol, "sharpe": p.mean() / p.std(ddof=1) * np.sqrt(252)}
        x = ref[(ref.posLimit_wan == pos) & (ref.cap_wan == cap)]
        if len(x):   # parity with risk_metrics.py comp run (posLimit 500 superset rows)
            assert abs(x.total_pnl_wan.iloc[0] - r["total_pnl_wan"]) < 0.01 and abs(x.mdd_wan.iloc[0] - r["mdd_wan"]) < 0.01, (pos, cap)
        rows.append(r)
    print("pos", pos, "done", flush=True)
res = pd.DataFrame(rows); res.to_csv(OUT, index=False)
pd.set_option("display.width", 250)
for k in ["usage_normal_pct", "total_pnl_wan", "mdd_wan", "worst_day_wan", "pnl_p10_wan", "pnl_mean_wan", "pnl_p90_wan", "sharpe"]:
    print("--", k); print(res.pivot(index="posLimit_wan", columns="cap_wan", values=k).round(1).to_string())
