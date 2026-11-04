"""Dump daily PnL (萬) for key restricted-day (posLimit, cap) configs from group_poscap.py's cache, for bootstrap checks.
Usage: uv run python group_poscap_series.py <gate_tune_out_dir>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
D = Path(sys.argv[1])
POS = [50, 100, 150, 200, 250, 300, 350, 400]
CAPS = [0, 250, 350, 500, 700, 850, 1000, 1250, 1500, 1750, 2100, 2800, 3500, 5000, 7000]
PI, CI = {p: i for i, p in enumerate(POS)}, {c: i for i, c in enumerate(CAPS)}
P = np.load(D / "poscap_PV.npz")["P"]
cal = pd.read_csv(D / "calendar.csv", dtype={"date": str}, float_precision="round_trip"); n = len(cal)
comp, gap = cal.comp.to_numpy(), np.nan_to_num(cal.tsmc_gap.to_numpy(), nan=-1)
STOP = comp < -1.5
G = {"g1": (comp >= -1.5) & (comp < -0.7), "g2": (gap > 0.01) & (gap <= 0.02), "g3": gap > 0.02}
BASE = {"g1": (400, 1000), "g2": (400, 700), "g3": (400, 2800)}


def daily(cfg):
    pos = np.full(n, 400); cap = np.full(n, 7000)
    for g, (p, c) in cfg.items():
        m = G[g]; pos[m] = np.minimum(pos[m], p); cap[m] = np.minimum(cap[m], c)
    cap[STOP] = 0
    return P[[PI[p] for p in pos], np.arange(n), [CI[c] for c in cap]]


cfgs = {"baseline": BASE}
for p in POS:
    cfgs[f"uniform_{p}x1000"] = {g: (p, 1000) for g in G}
cfgs |= {"g1_100x1000": BASE | {"g1": (100, 1000)}, "g2_100x7000": BASE | {"g2": (100, 7000)}, "g2_cap0": BASE | {"g2": (400, 0)},
         "g3_cap0": BASE | {"g3": (400, 0)}, "g3_50x7000": BASE | {"g3": (50, 7000)}, "joint_best_sharpe": {"g1": (400, 1000), "g2": (400, 1000), "g3": (200, 2100)},
         "no_restricted_cap_7000": {g: (400, 7000) for g in G}}
out = pd.DataFrame({"date": cal.date, "comp": comp, "tsmc_gap": cal.tsmc_gap, "stop": STOP, **{f"in_{g}": m for g, m in G.items()}, **{k: daily(v) for k, v in cfgs.items()}})
out.to_parquet(D / "group_poscap_daily_series.parquet", index=False)
for k in cfgs:
    x = out[k].to_numpy(); c = np.cumsum(x)
    print(f"{k:24s} total {x.sum():9.1f} mdd {(c - np.maximum.accumulate(c)).min():8.1f} sharpe {x.mean() / x.std(ddof=1) * np.sqrt(252):.3f}")
