"""G3 days: daily PnL difference between G3 cap 3500 and 1050 (the 1050->3500 tranche), t-stat, by year, and which G3 days sit in the MDD window."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
D = Path(sys.argv[1])
cal = pd.read_csv(D / "calendar.csv", dtype={"date": str}, float_precision="round_trip")
comp, gap = cal.comp.to_numpy(), np.nan_to_num(cal.tsmc_gap.to_numpy(), nan=-1)
m = (gap > 0.02) & ~(comp < -1.5) & ~((comp >= -1.5) & (comp < -0.7))   # G3 days where G3 cap binds (G1 overlap would cap at 1000)
P = np.load(D / "P.npy"); Y = cal.date.str[:4].to_numpy()
for lo, hi in ((3, 10), (3, 11), (0, 10), (8, 10)):
    d = (P[:, hi] - P[:, lo])[m]
    print(f"cap {350*lo}->{350*hi}: days {m.sum()}, sum {d.sum():.1f} 萬, mean {d.mean():.2f}, t {d.mean()/d.std(ddof=1)*np.sqrt(len(d)):.2f}, win% {(d>0).mean()*100:.0f}, worst {d.min():.1f}, best {d.max():.1f}")
    print("  by year:", {y: round(float((P[:, hi] - P[:, lo])[m & (Y == y)].sum()), 1) for y in ("2023", "2024", "2025", "2026")},
          "days:", {y: int((m & (Y == y)).sum()) for y in ("2023", "2024", "2025", "2026")})
w = (cal.date >= "20250723") & (cal.date <= "20250818")
print("G3 days in 2025-07-23..08-18:", cal.date[w & (gap > 0.02)].tolist())
d = (P[:, 10] - P[:, 3]); top = np.argsort(-np.abs(np.where(m, d, 0)))[:8]
print(pd.DataFrame({"date": cal.date[top], "gap_pct": gap[top] * 100, "pnl_cap1050_wan": P[top, 3], "pnl_cap3500_wan": P[top, 10], "diff_wan": d[top]}).round(1).to_string(index=False))
