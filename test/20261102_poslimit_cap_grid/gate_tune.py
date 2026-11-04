"""posLimit 400 x cap 7000: re-tune the negFill stop-trading gate, comp and TSMC (2330 gap) layers separately, then combined.
Daily cap = 7000 x scale (cap semantics). Real per-row replay: rows keep the gate's stop days (build_rows.py ... keep_stop) so a looser
stop threshold is replayable; each day's PnL at a cap is the prefix sum of rows with cumulative notional < cap.
PnL stats on every book date (no-trade days = 0 PnL); volume on traded days.
Usage: uv run python gate_tune.py <scratch> <rows_parquet> <out_dir>
"""
import sys, glob, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append("/home/kevin/Project/HFT")
from src.strategy.negFill.stopTradingGate import load_stop_trading_days, scale_for_comp, apply_cooldown, LOCAL_DIR
from src.research.tool import add_stop_loss_prod

S, ROWS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
FEE, POS, CAP = 19.3, 400, 7000
SCALES = np.round(np.arange(0, 1.0001, 0.05), 2); SIDX = {float(s): i for i, s in enumerate(SCALES)}

sw = pd.read_parquet(ROWS)
sw["stopLossProd"] = add_stop_loss_prod(sw, cache_path=S / "negfill_grid/stop_times_cache.parquet")   # depends on posLimit
sw = sw.sort_values(["Date", "TransTime"]).reset_index(drop=True)
sw = sw[~((sw.stopLossProd >= 3) & (sw.RemainSeconds < 12300))].reset_index(drop=True)
sw["cum"] = sw.groupby("Date").BidPrice1.cumsum() / 10
sw["pnl"] = sw.BidPrice1 * (sw.TakerSell_CloseBP - FEE) / 10 / 1e4; sw["notional"] = sw.BidPrice1 / 10   # 萬

cal = sorted(set().union(*[set(pd.read_parquet(f, columns=["Date"]).Date.dt.strftime("%Y%m%d")) for f in glob.glob(str(S / "negfill_book/book_chunk*.parquet"))]))
P = np.zeros((len(cal), len(SCALES))); V = np.zeros_like(P); ci = {d: i for i, d in enumerate(cal)}
for d, g in sw.groupby("Date"):
    cum, cp, cv = g.cum.to_numpy(), np.cumsum(g.pnl.to_numpy()), np.cumsum(g.notional.to_numpy())
    n = np.searchsorted(cum, CAP * SCALES, side="left")
    P[ci[d]] = np.where(n > 0, cp[np.maximum(n - 1, 0)], 0.0); V[ci[d]] = np.where(n > 0, cv[np.maximum(n - 1, 0)], 0.0)
Y26 = np.array([d >= "20260101" for d in cal])

gd = pd.read_csv(LOCAL_DIR / "gate_daily.csv", dtype={"asof_date": str, "next_trade_day": str}, float_precision="round_trip").set_index("next_trade_day")   # gap 0.020000000000000018 must stay > 0.02 like production
comp = gd.comp.reindex(cal).to_numpy(); gap = gd.tsmc_gap.reindex(cal).to_numpy()
print(f"book dates {len(cal)} {cal[0]}..{cal[-1]} | comp missing {np.isnan(comp).sum()} | gap missing {np.isnan(gap).sum()}", flush=True)
# apply_cooldown walks the gate's own calendar (asof days); book dates are a subset, so cooldown runs on the full gate series
gate_days = gd.index.tolist(); comp_all = gd.comp.to_numpy()


def metrics(scale, tag=None):
    idx = np.array([SIDX[round(float(s), 2)] for s in scale])
    p = P[np.arange(len(cal)), idx]; v = V[np.arange(len(cal)), idx]
    out = {}
    for k, m in (("", np.ones(len(cal), bool)), ("_23_25", ~Y26), ("_26", Y26)):
        x = p[m]; c = np.cumsum(x); dd = c - np.maximum.accumulate(c)
        out |= {f"total_pnl_wan{k}": x.sum(), f"mdd_wan{k}": dd.min(), f"sharpe{k}": x.mean() / x.std(ddof=1) * np.sqrt(252)}
    c = np.cumsum(p); dd = c - np.maximum.accumulate(c)
    out |= {"mdd_trough": cal[int(np.argmin(dd))], "worst_day_wan": p.min(), "pnl_p10_wan": np.percentile(p, 10), "pnl_mean_wan": p.mean(),
            "pnl_p90_wan": np.percentile(p, 90), "avg_volume_wan_traded_days": v[v > 0].mean(),
            "restricted_days": int((scale < 1).sum()), "stop_days": int((scale == 0).sum())}
    return out


def comp_scale(stop_thr, reduce_thr, reduce_scale, cooldown):
    base = [scale_for_comp(c, stop_thr, reduce_thr, reduce_scale) for c in comp_all]
    s = pd.Series(apply_cooldown(base, reduce_scale, cooldown)[0], index=gate_days)
    return s.reindex(cal).fillna(1.0).to_numpy()


def tsmc_scale(tiers):   # tiers: ((gap_gt, scale), ...) high to low
    out = np.ones(len(cal))
    for i, g in enumerate(gap):
        for thr, s in tiers:
            if not np.isnan(g) and g > thr:
                out[i] = s; break
    return out


# parity: production scale_by_date and the earlier comp run (comp_scale incl. 2-day cooldown) at 400 x 7000
gate = load_stop_trading_days()
prod = np.array([float(gate["scale_by_date"].get(d, gate["scale_by_date"].get(int(d), 1.0))) for d in cal])
ref_g = pd.read_csv(S / "usage_band/risk_metrics_graded.csv").query("posLimit_wan == 400 and cap_wan == 7000").iloc[0]
ref_c = pd.read_csv(S / "usage_band/risk_metrics_comp_withcooldown.csv").query("posLimit_wan == 400 and cap_wan == 7000").iloc[0]
mg, mc = metrics(prod), metrics(comp_scale(-1.5, -0.7, 0.1, 2))
assert abs(mg["total_pnl_wan"] - ref_g.total_pnl_wan) < 0.01 and abs(mg["mdd_wan"] - ref_g.mdd_wan) < 0.01, (mg["total_pnl_wan"], ref_g.total_pnl_wan)
assert abs(mc["total_pnl_wan"] - ref_c.total_pnl_wan) < 0.01 and abs(mc["mdd_wan"] - ref_c.mdd_wan) < 0.01, (mc["total_pnl_wan"], ref_c.total_pnl_wan)
print("parity ok: production graded and comp+cooldown2 match risk_metrics", flush=True)

ref_rows = [{"name": "no gate", **metrics(np.ones(len(cal)))},
            {"name": "production (comp + cooldown 2 + TSMC)", **mg},
            {"name": "comp -1.5/-0.7 x0.1 + cooldown 2", **mc},
            {"name": "comp -1.5/-0.7 x0.1 (no cooldown)", **metrics(comp_scale(-1.5, -0.7, 0.1, 0))},
            {"name": "TSMC 1%:0.1 / 2%:0.4 only", **metrics(tsmc_scale(((0.02, 0.4), (0.01, 0.1))))}]
pd.DataFrame(ref_rows).to_csv(OUT / "reference.csv", index=False)

NONE = -np.inf
STOPS = [NONE, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0]
REDUCES = [NONE, -1.5, -1.25, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
RSCALES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
rows = []
for st, rd, rs, cd in itertools.product(STOPS, REDUCES, RSCALES, [0, 1, 2]):
    if rd != NONE and rd <= st:
        continue
    if rd == NONE and rs != 0.1:   # reduce scale is irrelevant without a reduce tier
        continue
    rows.append({"stop_thr": st, "reduce_thr": rd, "reduce_scale": rs, "reduce_cap_wan": CAP * rs, "cooldown_days": cd, **metrics(comp_scale(st, rd, rs, cd))})
comp_grid = pd.DataFrame(rows); comp_grid.to_csv(OUT / "comp_grid.csv", index=False)
print("comp grid", len(comp_grid), flush=True)

TS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
rows = []
for thr, s in itertools.product([0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03], TS):
    rows.append({"kind": "one", "t1": thr, "s1": s, "t2": np.nan, "s2": np.nan, **metrics(tsmc_scale(((thr, s),)))})
for t1, t2, s1, s2 in itertools.product([0.0075, 0.01, 0.0125, 0.015], [0.02, 0.025, 0.03], TS, TS):
    rows.append({"kind": "two", "t1": t1, "s1": s1, "t2": t2, "s2": s2, **metrics(tsmc_scale(((t2, s2), (t1, s1))))})
tsmc_grid = pd.DataFrame(rows); tsmc_grid.to_csv(OUT / "tsmc_grid.csv", index=False)
print("tsmc grid", len(tsmc_grid), flush=True)

# scale lookup for the combined step
np.save(OUT / "P.npy", P); np.save(OUT / "V.npy", V)
pd.DataFrame({"date": cal, "comp": comp, "tsmc_gap": gap, "prod_scale": prod}).to_csv(OUT / "calendar.csv", index=False)
