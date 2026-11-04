"""posLimit 400 x cap 7000, no cooldown: per-group per-name limit (posLimit) x daily cap on the restricted days.
Groups (asof-day state -> next trade day): G1 comp in [-1.5, -0.7); G2 2330 gap in (1%, 2%]; G3 gap > 2%; comp < -1.5 stays cap 0.
A day in several groups takes the min posLimit and the min cap. Normal days: posLimit 400, cap 7000.
Real per-row replay: rows for each posLimit = Position < p subset of the 400 rows (keep_stop), stopLossProd recomputed per p,
daily PnL at each cap = prefix sum of rows with cumulative notional < cap. Every book date counts (no-trade days = 0).
Usage: uv run python group_poscap.py <scratch> <rows_pos400_keepstop.parquet> <gate_tune_out_dir>
"""
import sys, glob, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append("/home/kevin/Project/HFT")
from src.research.tool import add_stop_loss_prod

S, ROWS, D = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); FEE = 19.3
POS = [50, 100, 150, 200, 250, 300, 350, 400]
CAPS = [0, 250, 350, 500, 700, 850, 1000, 1250, 1500, 1750, 2100, 2800, 3500, 5000, 7000]
PI, CI = {p: i for i, p in enumerate(POS)}, {c: i for i, c in enumerate(CAPS)}
cal = pd.read_csv(D / "calendar.csv", dtype={"date": str}, float_precision="round_trip")
dates = cal.date.tolist(); n = len(dates); di = {d: i for i, d in enumerate(dates)}
YEAR = cal.date.str[:4].to_numpy(); Y26 = YEAR == "2026"

cache = D / "poscap_PV.npz"
if cache.exists():
    z = np.load(cache); P, V = z["P"], z["V"]
else:
    rr = pd.read_parquet(ROWS)
    P = np.zeros((len(POS), n, len(CAPS))); V = np.zeros_like(P)
    for p in POS:
        sw = rr[rr.Position < p].reset_index(drop=True)
        sw["stopLossProd"] = add_stop_loss_prod(sw, cache_path=S / "negfill_grid/stop_times_cache.parquet")   # depends on posLimit
        sw = sw.sort_values(["Date", "TransTime"]).reset_index(drop=True)
        sw = sw[~((sw.stopLossProd >= 3) & (sw.RemainSeconds < 12300))].reset_index(drop=True)
        cum = (sw.groupby("Date").BidPrice1.cumsum() / 10).to_numpy()
        pnl = (sw.BidPrice1 * (sw.TakerSell_CloseBP - FEE) / 10 / 1e4).to_numpy(); notional = (sw.BidPrice1 / 10).to_numpy()   # 萬
        for d, ix in sw.groupby("Date").indices.items():
            cp, cv = np.cumsum(pnl[ix]), np.cumsum(notional[ix])
            k = np.searchsorted(cum[ix], CAPS, side="left")
            P[PI[p], di[d]] = np.where(k > 0, cp[np.maximum(k - 1, 0)], 0.0); V[PI[p], di[d]] = np.where(k > 0, cv[np.maximum(k - 1, 0)], 0.0)
        print(f"pos {p}: rows {len(sw):,}", flush=True)
    np.savez(cache, P=P, V=V)
# parity: posLimit 400 everywhere with the gate_tune scale grid (same rows, same stopLossProd)
P400 = np.load(D / "P.npy")   # columns = 7000 x scale steps of 0.05
for cap, k in ((7000, 20), (700, 2), (1050, 3), (2800, 8)):
    if cap in CI:
        assert np.allclose(P[PI[400], :, CI[cap]], P400[:, k]), cap
print("parity ok vs gate_tune P.npy", flush=True)

comp, gap = cal.comp.to_numpy(), np.nan_to_num(cal.tsmc_gap.to_numpy(), nan=-1)
STOP = comp < -1.5
GROUPS = {"G1 comp −1.5～−0.7": (comp >= -1.5) & (comp < -0.7), "G2 台積電 1–2%": (gap > 0.01) & (gap <= 0.02), "G3 台積電 >2%": gap > 0.02}
BASE = {"G1 comp −1.5～−0.7": (400, 1000), "G2 台積電 1–2%": (400, 700), "G3 台積電 >2%": (400, 2800)}   # current-ish caps at 400 x 7000
for g, m in GROUPS.items():
    print(f"{g}: {int(m.sum())} days, {int((m & STOP).sum())} also comp-stop", flush=True)


def daily(cfg):
    pos = np.full(n, 400); cap = np.full(n, 7000)
    for g, (p, c) in cfg.items():
        m = GROUPS[g]; pos[m] = np.minimum(pos[m], p); cap[m] = np.minimum(cap[m], c)
    cap[STOP] = 0
    idx = np.arange(n)
    return P[[PI[p] for p in pos], idx, [CI[c] for c in cap]], V[[PI[p] for p in pos], idx, [CI[c] for c in cap]]


def stats(x):
    c = np.cumsum(x); dd = c - np.maximum.accumulate(c)
    return {"total_pnl_wan": x.sum(), "mdd_wan": dd.min(), "sharpe": x.mean() / x.std(ddof=1) * np.sqrt(252)}


def evaluate(cfg, g=None):
    p, v = daily(cfg); r = stats(p) | {"worst_day_wan": p.min(), "pnl_p10_wan": np.percentile(p, 10), "pnl_mean_wan": p.mean(), "pnl_p90_wan": np.percentile(p, 90)}
    r |= {f"{k}_23_25": val for k, val in stats(p[~Y26]).items() if k != "total_pnl_wan"} | {f"{k}_26": val for k, val in stats(p[Y26]).items() if k != "total_pnl_wan"}
    if g is not None:   # stats on this group's days (comp-stop days excluded: they are always 0)
        m = GROUPS[g] & ~STOP; x = p[m]; vv = v[m]
        r |= {"grp_days": int(m.sum()), "grp_pnl_wan": x.sum(), "grp_mean_wan": x.mean(), "grp_std_wan": x.std(ddof=1), "grp_t": x.mean() / x.std(ddof=1) * np.sqrt(m.sum()),
              "grp_worst_wan": x.min(), "grp_avg_notional_wan": vv.mean(), "grp_bp": x.sum() / vv.sum() * 1e4 if vv.sum() > 0 else np.nan,
              "grp_pnl_wan_23_25": p[m & ~Y26].sum(), "grp_pnl_wan_26": p[m & Y26].sum()}
    return r


rows = []
for g in GROUPS:
    for p, c in itertools.product(POS, CAPS):
        cfg = dict(BASE); cfg[g] = (p, c)
        rows.append({"group": g, "posLimit_wan": p, "cap_wan": c, **evaluate(cfg, g)})
grid = pd.DataFrame(rows); grid.to_csv(D / "group_poscap_grid.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
base = evaluate(BASE); print("baseline (400 on all groups; caps 1000/700/2800):", {k: round(v, 2) for k, v in base.items()})
for g in GROUPS:
    x = grid[grid.group == g]
    for v in ["sharpe", "mdd_wan", "total_pnl_wan", "grp_mean_wan", "grp_std_wan"]:
        print(f"-- {g} | {v} (rows posLimit, cols cap)"); print(x.pivot(index="posLimit_wan", columns="cap_wan", values=v).round(2).to_string())

# joint posLimit sweep with each group's cap from a short list (coordinate check that the per-group optimum is not an interaction artefact)
jrows = []
for p1, p2, p3 in itertools.product(POS, POS, POS):
    for c1, c2, c3 in itertools.product([700, 1000, 1500], [500, 700, 1000], [1000, 2100, 2800]):
        cfg = {"G1 comp −1.5～−0.7": (p1, c1), "G2 台積電 1–2%": (p2, c2), "G3 台積電 >2%": (p3, c3)}
        jrows.append({"pos_g1": p1, "cap_g1": c1, "pos_g2": p2, "cap_g2": c2, "pos_g3": p3, "cap_g3": c3, **evaluate(cfg)})
joint = pd.DataFrame(jrows); joint.to_csv(D / "group_poscap_joint.csv", index=False)
print("joint", len(joint)); print(joint.sort_values("sharpe", ascending=False).head(15).round(2).to_string(index=False))

# one shared (posLimit, cap) for all three groups
urows = [{"posLimit_wan": p, "cap_wan": c, **evaluate({g: (p, c) for g in GROUPS})} for p, c in itertools.product(POS, CAPS)]
uni = pd.DataFrame(urows); uni.to_csv(D / "group_poscap_uniform.csv", index=False)
for v in ["sharpe", "mdd_wan", "total_pnl_wan", "worst_day_wan", "pnl_p10_wan", "sharpe_23_25", "sharpe_26"]:
    print(f"-- uniform | {v} (rows posLimit, cols cap)"); print(uni.pivot(index="posLimit_wan", columns="cap_wan", values=v).round(2).to_string())
both = joint[(joint.mdd_wan >= base["mdd_wan"] - 1e-9) & (joint.sharpe >= base["sharpe"] - 1e-9)]
print(f"joint configs at least as good as baseline on both MDD and Sharpe: {len(both)}")
print(both.sort_values("sharpe", ascending=False).head(10).round(2).to_string(index=False))
