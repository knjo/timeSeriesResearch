"""Extend group_poscap.py above posLimit 400 on the restricted days (G1/G2/G3) only; normal days stay 400 x 7000, comp-stop days cap 0.
Rows for posLimit 400-1000 come from the posLimit-1000 superset (build_rows.py without keep_stop): it lacks only the comp-stop days,
which are cap 0 anyway. Parity: posLimit 400 from the superset must equal group_poscap's cube on every non-stop day.
Usage: uv run python group_poscap_hi.py <scratch> <rows_pos1000_nocap.parquet> <gate_tune_out_dir>
"""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append("/home/kevin/Project/HFT")
from src.research.tool import add_stop_loss_prod

S, ROWS, D = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); FEE = 19.3
LO = [50, 100, 150, 200, 250, 300, 350, 400]; HI = [400, 500, 600, 700, 800, 1000]
CAPS = [0, 250, 350, 500, 700, 850, 1000, 1250, 1500, 1750, 2100, 2800, 3500, 5000, 7000]; CI = {c: i for i, c in enumerate(CAPS)}
cal = pd.read_csv(D / "calendar.csv", dtype={"date": str}, float_precision="round_trip")
dates = cal.date.tolist(); n = len(dates); di = {d: i for i, d in enumerate(dates)}
YEAR = cal.date.str[:4].to_numpy(); Y26 = YEAR == "2026"
comp, gap = cal.comp.to_numpy(), np.nan_to_num(cal.tsmc_gap.to_numpy(), nan=-1)
STOP = comp < -1.5
GROUPS = {"G1 comp −1.5～−0.7": (comp >= -1.5) & (comp < -0.7), "G2 台積電 1–2%": (gap > 0.01) & (gap <= 0.02), "G3 台積電 >2%": gap > 0.02}
BASE = {"G1 comp −1.5～−0.7": (400, 1000), "G2 台積電 1–2%": (400, 700), "G3 台積電 >2%": (400, 2800)}

PLO = np.load(D / "poscap_PV.npz")["P"]
cache = D / "poscap_hi_P.npz"
if cache.exists():
    PHI = np.load(cache)["P"]
else:
    rr = pd.read_parquet(ROWS)
    PHI = np.zeros((len(HI), n, len(CAPS)))
    for j, p in enumerate(HI):
        sw = rr[rr.Position < p].reset_index(drop=True)
        sw["stopLossProd"] = add_stop_loss_prod(sw, cache_path=S / "negfill_grid/stop_times_cache.parquet")
        sw = sw.sort_values(["Date", "TransTime"]).reset_index(drop=True)
        sw = sw[~((sw.stopLossProd >= 3) & (sw.RemainSeconds < 12300))].reset_index(drop=True)
        cum = (sw.groupby("Date").BidPrice1.cumsum() / 10).to_numpy()
        pnl = (sw.BidPrice1 * (sw.TakerSell_CloseBP - FEE) / 10 / 1e4).to_numpy()
        for d, ix in sw.groupby("Date").indices.items():
            cp = np.cumsum(pnl[ix]); k = np.searchsorted(cum[ix], CAPS, side="left")
            PHI[j, di[d]] = np.where(k > 0, cp[np.maximum(k - 1, 0)], 0.0)
        print(f"pos {p}: rows {len(sw):,}", flush=True)
    np.savez(cache, P=PHI)
ok = ~STOP
assert np.allclose(PHI[0][ok], PLO[LO.index(400)][ok]), "posLimit 400 superset != keep_stop rows on non-stop days"
print("parity ok: posLimit 400 from the 1000 superset == keep_stop cube on all non-stop days", flush=True)

POS = LO + HI[1:]


def cube(p):
    return PLO[LO.index(p)] if p in LO else PHI[HI.index(p)]


def daily(cfg):
    pos = np.full(n, 400); cap = np.full(n, 7000); seen = np.zeros(n, bool)   # group values replace the normal-day 400 x 7000 (they may exceed 400)
    for g, (p, c) in cfg.items():
        m = GROUPS[g]
        pos[m] = np.where(seen[m], np.minimum(pos[m], p), p); cap[m] = np.where(seen[m], np.minimum(cap[m], c), c); seen |= m
    cap[STOP] = 0
    out = np.empty(n)
    for p in np.unique(pos):
        m = pos == p; out[m] = cube(p)[np.where(m)[0], [CI[c] for c in cap[m]]]
    return out


def stats(x):
    c = np.cumsum(x); dd = c - np.maximum.accumulate(c)
    return {"total_pnl_wan": x.sum(), "mdd_wan": dd.min(), "sharpe": x.mean() / x.std(ddof=1) * np.sqrt(252)}


def evaluate(cfg, g=None):
    x = daily(cfg); r = stats(x) | {"worst_day_wan": x.min()}
    r |= {f"sharpe_{k}": stats(x[m])["sharpe"] for k, m in (("23_25", ~Y26), ("26", Y26))}
    if g is not None:
        m = GROUPS[g] & ~STOP
        r |= {"grp_pnl_wan": x[m].sum(), "grp_pnl_wan_23_25": x[m & ~Y26].sum(), "grp_pnl_wan_26": x[m & Y26].sum(),
              "grp_mean_wan": x[m].mean(), "grp_t": x[m].mean() / x[m].std(ddof=1) * np.sqrt(m.sum()) if x[m].std() > 0 else np.nan}
    return r


assert abs(evaluate(BASE)["total_pnl_wan"] - 14049.53) < 0.01
rows = [{"group": g, "posLimit_wan": p, "cap_wan": c, **evaluate(BASE | {g: (p, c)}, g)} for g in GROUPS for p in POS for c in CAPS]
grid = pd.DataFrame(rows); grid.to_csv(D / "group_poscap_grid_ext.csv", index=False)
uni = pd.DataFrame([{"posLimit_wan": p, "cap_wan": c, **evaluate({g: (p, c) for g in GROUPS})} for p in POS for c in CAPS])
uni.to_csv(D / "group_poscap_uniform_ext.csv", index=False)
pd.set_option("display.width", 250)
SEL = [500, 700, 850, 1000, 1250, 1500, 2100, 2800, 3500, 7000]
for g in GROUPS:
    x = grid[grid.group == g]
    for v in ["sharpe", "grp_pnl_wan", "grp_pnl_wan_23_25", "grp_pnl_wan_26"]:
        print(f"-- {g} | {v}"); print(x.pivot(index="posLimit_wan", columns="cap_wan", values=v)[SEL].round(2).to_string())
for v in ["sharpe", "mdd_wan", "total_pnl_wan", "sharpe_23_25", "sharpe_26"]:
    print(f"-- uniform | {v}"); print(uni.pivot(index="posLimit_wan", columns="cap_wan", values=v)[SEL].round(2).to_string())
