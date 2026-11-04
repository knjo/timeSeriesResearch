"""Paired stationary block bootstrap (L=10, B=5000; same generator as the verifier's bootstrap_pos_only.py) for restricted-day
per-name limits above 400. Daily series rebuilt with group_poscap_hi.py's replay. Usage: uv run python group_poscap_boot.py <gate_tune_out_dir>
"""
import sys, runpy
from pathlib import Path
import numpy as np, pandas as pd
D = Path(sys.argv[1])
here = Path(__file__).with_name("group_poscap_hi.py")
sys.argv = [str(here), "unused", "unused", str(D)]   # cubes are cached in D; only the replay helpers are needed
ns = runpy.run_path(str(here), run_name="poscap_hi")
daily, GROUPS, STOP, YEAR = ns["daily"], ns["GROUPS"], ns["STOP"], ns["YEAR"]
G1, G2, G3 = list(GROUPS)
BASE = {G1: (400, 1000), G2: (400, 700), G3: (400, 2800)}
CFG = {"baseline": BASE, **{f"uniform_{p}x1000": {g: (p, 1000) for g in GROUPS} for p in (400, 600, 800, 1000)},
       "uniform_1000x1250": {g: (1000, 1250) for g in GROUPS}, "g1_1000x1250": BASE | {G1: (1000, 1250)}, "g2_1000x1000": BASE | {G2: (1000, 1000)},
       "g1_1000x1250_g2_1000x1000": BASE | {G1: (1000, 1250), G2: (1000, 1000)}}
X = np.vstack([daily(v) for v in CFG.values()]); names = list(CFG); n = X.shape[1]
rng = np.random.default_rng(7); B = 5000; L = 10
start = rng.integers(0, n, (B, n)); nb = rng.random((B, n)) < 1 / L; nb[:, 0] = True
idx = np.empty((B, n), np.int64); cur = start[:, 0].copy()
for t in range(n):
    cur = np.where(nb[:, t], start[:, t], (cur + 1) % n); idx[:, t] = cur
Y = X[:, idx]
sh = Y.mean(-1) / Y.std(-1, ddof=1) * np.sqrt(252); c = np.cumsum(Y, -1); mdd = (c - np.maximum.accumulate(c, -1)).min(-1)
sh0 = X.mean(-1) / X.std(-1, ddof=1) * np.sqrt(252); c0 = np.cumsum(X, -1); mdd0 = (c0 - np.maximum.accumulate(c0, -1)).min(-1)
restr = np.zeros(n, bool)
for m in GROUPS.values():
    restr |= m
restr &= ~STOP
rows = []
for ref in ("baseline", "uniform_400x1000"):
    r = names.index(ref)
    for j, k in enumerate(names):
        if k == ref:
            continue
        ds, dm = sh[j] - sh[r], mdd[j] - mdd[r]; dd = (X[j] - X[r])[restr]
        yr = {f"dsharpe_{y}": (lambda a, b: a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1))(X[j][YEAR == y], X[r][YEAR == y]) * np.sqrt(252) for y in sorted(set(YEAR))}
        rows.append({"config": k, "vs": ref, "sharpe_diff": sh0[j] - sh0[r], "sharpe_ci_lo": np.percentile(ds, 2.5), "sharpe_ci_hi": np.percentile(ds, 97.5),
                     "p_better": np.mean(ds > 0), "mdd_diff_wan": mdd0[j] - mdd0[r], "mdd_ci_lo": np.percentile(dm, 2.5), "mdd_ci_hi": np.percentile(dm, 97.5),
                     "total_diff_wan": X[j].sum() - X[r].sum(), "restr_day_t": dd.mean() / dd.std(ddof=1) * np.sqrt(len(dd)), **yr})
out = pd.DataFrame(rows); out.to_csv(D / "group_poscap_boot_hi.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(out.round(3).to_string(index=False))
for j, k in enumerate(names):
    x = X[j][restr]; print(f"{k:28s} restricted days: mean {x.mean():6.2f} std {x.std(ddof=1):6.2f} worst {x.min():7.1f} | sharpe {sh0[j]:.3f} mdd {mdd0[j]:.1f} total {X[j].sum():.0f}")
