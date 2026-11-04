"""negFill 400 萬 × 7,000 萬、comp 分級＋出關冷卻（正式 comp_scale）× 台積電跳空 >1% 一律 0.5（使用者 11/03 指定，避免 0.1/0.4 過度擬合）× cap：逐日損益 + 主管報告指標。
重放語意同 ../gate_tune.py（cap × scale）；另算「完全不控管」對照。輸入為 inputs/ 的 11/03 快照。
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append("/home/kevin/Project/HFT")
from src.research.tool import add_stop_loss_prod

OUT = Path(__file__).resolve().parent
INP = OUT / "inputs"   # 11/03 快照：build_rows.py 400 keep_stop 成交列、停損時間 cache、book 交易日曆、gate_daily.csv
FEE, CAP, DPY = 19.3, 7000, 242.0

sw = pd.read_parquet(INP / "rows_pos400_keepstop.parquet")
sw["stopLossProd"] = add_stop_loss_prod(sw, cache_path=INP / "stop_times_cache.parquet")
sw = sw.sort_values(["Date", "TransTime"]).reset_index(drop=True)
sw = sw[~((sw.stopLossProd >= 3) & (sw.RemainSeconds < 12300))].reset_index(drop=True)
sw["cum"] = sw.groupby("Date").BidPrice1.cumsum() / 10
sw["pnl"] = sw.BidPrice1 * (sw.TakerSell_CloseBP - FEE) / 10 / 1e4
sw["notional"] = sw.BidPrice1 / 10
cal = pd.read_csv(INP / "calendar.csv", dtype=str).Date.tolist()
gd = pd.read_csv(INP / "gate_daily_20261103.csv", dtype={"asof_date": str, "next_trade_day": str}, float_precision="round_trip").set_index("next_trade_day")
cs, gap = gd.comp_scale.reindex(cal), gd.tsmc_gap.reindex(cal); assert cs.notna().all() and gap.notna().all()
TSMC_SCALE = 0.5
scale = np.minimum(cs.to_numpy(), np.where(gap.to_numpy() > 0.01, TSMC_SCALE, 1.0))
print("scale counts", pd.Series(scale).value_counts().sort_index().to_dict())

by = {d: g for d, g in sw.groupby("Date")}
rec = []
for d, sc in zip(cal, scale):
    g = by.get(d)
    if g is None:
        rec.append((0.0, 0.0, 0, 0.0, 0.0)); continue
    k = g.cum.to_numpy() < CAP * sc; kf = g.cum.to_numpy() < CAP
    rec.append((g.pnl.to_numpy()[k].sum(), g.notional.to_numpy()[k].sum(), g.QuoteCode[k].nunique(), g.pnl.to_numpy()[kf].sum(), g.notional.to_numpy()[kf].sum()))
d = pd.DataFrame(rec, index=pd.to_datetime(cal), columns=["pnl", "notional", "n_names", "pnl_nogate", "notional_nogate"])
d["scale"] = scale; d["stop"] = d.scale == 0; d["reduce"] = (d.scale > 0) & (d.scale < 1); d["reduce_comp"] = d.scale == 0.1; d["reduce_tsmc"] = d.scale == 0.5
d.to_parquet(OUT / "daily_400x7000_tsmc05.parquet")


def dd_info(p, idx):
    c = np.cumsum(p); dd = c - np.maximum.accumulate(np.maximum(c, 0)); tr = int(np.argmin(dd)); pk = int(np.argmax(c[:tr + 1])) if c[:tr + 1].max() > 0 else 0
    r = np.where(c[tr:] >= c[pk])[0]
    return dd.min(), str(idx[pk].date()), str(idx[tr].date()), (str(idx[tr + r[0]].date()) if len(r) else None)


p = d.pnl.to_numpy(); traded = d[d.notional > 0]; normal = traded[traded.scale == 1]
mdd, pk, tr, rc = dd_info(p, d.index)
w = d[d.index >= "2026-07-01"]; rmdd, rpk, rtr, rrc = dd_info(w.pnl.to_numpy(), w.index)
ng = d.pnl_nogate.to_numpy(); cng = np.cumsum(ng)
summary = {
    "start": cal[0], "end": cal[-1], "calendar_days": len(cal), "stop_days": int(d.stop.sum()), "reduce_days": int(d.reduce.sum()), "traded_days": len(traded),
    "days_per_year": DPY, "total_pnl": p.sum(), "annual_pnl": p.mean() * DPY,
    "mdd": mdd, "mdd_peak": pk, "mdd_trough": tr, "mdd_recover": rc,
    "recent_mdd": rmdd, "recent_peak": rpk, "recent_trough": rtr, "recent_recover": rrc,
    "worst_day": p.min(), "worst_day_date": str(d.index[int(np.argmin(p))].date()), "best_day": p.max(),
    "p10": float(np.percentile(traded.pnl, 10)), "p50": float(np.percentile(traded.pnl, 50)), "p90": float(np.percentile(traded.pnl, 90)),
    "win_rate": float((traded.pnl > 0).mean()), "avg_notional": traded.notional.mean(), "avg_notional_normal": normal.notional.mean(),
    "avg_names": traded.n_names.mean(), "sharpe": p.mean() / p.std(ddof=1) * np.sqrt(252),
    "top10_share": float(traded.pnl[traded.pnl >= np.percentile(traded.pnl, 90)].sum() / traded.pnl.sum()),
    "nogate_total": ng.sum(), "nogate_mdd": (cng - np.maximum.accumulate(cng)).min(), "nogate_worst": ng.min(),
    "nogate_sharpe": ng.mean() / ng.std(ddof=1) * np.sqrt(252),
    "restricted_full_pnl": float(d.pnl_nogate[d.scale < 1].sum()), "restricted_actual_pnl": float(d.pnl[d.scale < 1].sum()),
}
yr = []
for y, gg in d.groupby(d.index.year):
    pp = gg.pnl.to_numpy(); cc = np.cumsum(pp); t = gg[gg.notional > 0]
    yr.append({"year": int(y), "start": str(gg.index[0].date()), "end": str(gg.index[-1].date()), "days": len(gg), "traded_days": len(t),
               "stop_days": int(gg.stop.sum()), "reduce_days": int(gg.reduce.sum()),
               "pnl": pp.sum(), "annualized": pp.mean() * DPY, "mdd": (cc - np.maximum.accumulate(np.maximum(cc, 0))).min(), "worst_day": pp.min(),
               "p10": float(np.percentile(t.pnl, 10)), "p50": float(np.percentile(t.pnl, 50)), "p90": float(np.percentile(t.pnl, 90)),
               "win_rate": float((t.pnl > 0).mean()), "avg_notional": t.notional.mean()})
summary["years"] = yr
json.dump(summary, open(OUT / "summary_tsmc05.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in summary.items() if k != "years"}, ensure_ascii=False, indent=1, default=float))
print(pd.DataFrame(yr).round(1).to_string(index=False))
