"""Fine G3 (2330 gap > 2%) cap sweep at posLimit 400. G3 days read from gate_tune P.npy (cap 7000 x scale step 0.05 -> 350 萬 steps),
G1/G2 from poscap_PV.npz (same posLimit-400 rows; parity asserted). comp-stop days cap 0. Overlap = min cap."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
D = Path(sys.argv[1])
cal = pd.read_csv(D / "calendar.csv", dtype={"date": str}, float_precision="round_trip")
n = len(cal); Y = cal.date.str[:4].to_numpy(); Y26 = Y == "2026"
comp, gap = cal.comp.to_numpy(), np.nan_to_num(cal.tsmc_gap.to_numpy(), nan=-1)
STOP = comp < -1.5; G1 = (comp >= -1.5) & (comp < -0.7); G2 = (gap > 0.01) & (gap <= 0.02); G3 = gap > 0.02
P05 = np.load(D / "P.npy")                      # [day, k] cap = 350*k
PLO = np.load(D / "poscap_PV.npz")["P"][7]      # posLimit 400, [day, CAPS]
CAPS = [0, 250, 350, 500, 700, 850, 1000, 1250, 1500, 1750, 2100, 2800, 3500, 5000, 7000]; CI = {c: i for i, c in enumerate(CAPS)}
for c, k in ((700, 2), (2800, 8), (3500, 10), (7000, 20)):
    assert np.allclose(PLO[:, CI[c]], P05[:, k])
def run(c1, c2, k3):
    x = PLO[:, CI[7000]].copy()
    cap_other = np.full(n, 7000); cap_other[G1] = c1; cap_other[G2] = np.minimum(cap_other[G2], c2)
    for i in range(n):
        if STOP[i]: x[i] = 0; continue
        if G3[i]:
            c3 = 350 * k3
            if cap_other[i] < c3: x[i] = PLO[i, CI[cap_other[i]]]
            else: x[i] = P05[i, k3]
        else: x[i] = PLO[i, CI[cap_other[i]]]
    return x
def st(x):
    c = np.cumsum(x); dd = c - np.maximum.accumulate(c); i = dd.argmin(); j = c[:i + 1].argmax()
    return x.sum(), dd.min(), x.mean() / x.std(ddof=1) * np.sqrt(252), cal.date[j], cal.date[i]
rows = []
for c1, c2 in ((1000, 700), (1000, 1000)):
    for k in range(0, 21):
        x = run(c1, c2, k); tot, mdd, sh, pk, tr = st(x)
        m = G3 & ~STOP
        r = {"G1_G2_cap_wan": f"{c1}/{c2}", "G3_cap_wan": 350 * k, "total_pnl_wan": tot, "mdd_wan": mdd, "sharpe": sh,
             "sharpe_23_25": st(x[~Y26])[2], "sharpe_26": st(x[Y26])[2], "worst_day_wan": x.min(), "mdd_peak": pk, "mdd_trough": tr,
             "g3_pnl_wan": x[m].sum(), "g3_t": x[m].mean() / x[m].std(ddof=1) * np.sqrt(m.sum()) if x[m].std() > 0 else np.nan,
             "g3_worst_wan": x[m].min(), "g3_win_days_pct": (x[m] > 0).mean() * 100, "g3_pnl_wan_20250808": x[cal.date.to_numpy() == "20250808"].sum()}
        for y in ("2023", "2024", "2025", "2026"):
            r[f"g3_pnl_wan_{y}"] = x[m & (Y == y)].sum()
        rows.append(r)
out = pd.DataFrame(rows); out.to_csv(Path(sys.argv[2]) / "g3_fine.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(f"G3 days {int((G3 & ~STOP).sum())} (excl stop)")
print(out.round(2).to_string(index=False))
