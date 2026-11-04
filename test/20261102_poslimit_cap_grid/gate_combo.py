"""posLimit 400 x cap 7000: comp x TSMC combined gate, bucket tables, per-year stats and a 2023-25 -> 2026 walk-forward pick.
Reads gate_tune.py outputs (P.npy = daily PnL per scale step of the 7000 cap, calendar.csv).
Usage: uv run python gate_combo.py <gate_tune_out_dir>
"""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append("/home/kevin/Project/HFT")
from src.strategy.negFill.stopTradingGate import scale_for_comp, apply_cooldown, LOCAL_DIR

D = Path(sys.argv[1]); CAP = 7000
P = np.load(D / "P.npy"); cal = pd.read_csv(D / "calendar.csv", dtype={"date": str}, float_precision="round_trip")
dates = cal.date.tolist(); gap = cal.tsmc_gap.to_numpy(); n = len(dates)
SCALES = np.round(np.arange(0, 1.0001, 0.05), 2); SIDX = {float(s): i for i, s in enumerate(SCALES)}
YEAR = cal.date.str[:4].to_numpy(); Y26 = YEAR == "2026"
gd = pd.read_csv(LOCAL_DIR / "gate_daily.csv", dtype={"asof_date": str, "next_trade_day": str}, float_precision="round_trip").set_index("next_trade_day")


def daily(scale):
    return P[np.arange(n), [SIDX[round(float(s), 2)] for s in scale]]


def stats(p, m=None):
    x = p if m is None else p[m]; c = np.cumsum(x); dd = c - np.maximum.accumulate(c)
    return {"total_pnl_wan": x.sum(), "mdd_wan": dd.min(), "sharpe": x.mean() / x.std(ddof=1) * np.sqrt(252), "worst_day_wan": x.min()}


def metrics(scale):
    p = daily(scale); out = stats(p)
    for k, m in (("_23_25", ~Y26), ("_26", Y26)):
        out |= {f"{a}{k}": b for a, b in stats(p, m).items() if a != "worst_day_wan"}
    out |= {"pnl_p10_wan": np.percentile(p, 10), "pnl_mean_wan": p.mean(), "pnl_p90_wan": np.percentile(p, 90), "restricted_days": int((scale < 1).sum())}
    return out


def comp_scale(stop_thr, reduce_thr, reduce_scale, cooldown):
    base = [scale_for_comp(c, stop_thr, reduce_thr, reduce_scale) for c in gd.comp]
    return pd.Series(apply_cooldown(base, reduce_scale, cooldown)[0], index=gd.index).reindex(dates).fillna(1.0).to_numpy()


def tsmc_scale(tiers):
    out = np.ones(n)
    for thr, s in sorted(tiers, reverse=True)[::-1]:   # low threshold first, higher tiers overwrite
        out[np.nan_to_num(gap, nan=-1) > thr] = s
    return out


# bucket tables: full-cap daily PnL by comp / by yesterday's 2330 gap (萬)
full = P[:, -1]
for name, col, bins in (("comp", cal.comp, [-9, -1.5, -1.0, -0.7, -0.5, -0.3, 0, 0.5, 9]),
                        ("tsmc_gap", cal.tsmc_gap * 100, [-9, -1, 0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 9])):
    b = pd.cut(col, bins)
    t = pd.DataFrame({"bucket": b, "pnl": full, "year": YEAR}).groupby("bucket", observed=True)
    out = t.pnl.agg(days="size", mean_wan="mean", std_wan="std", total_wan="sum")
    out["t_stat"] = out.mean_wan / (out.std_wan / np.sqrt(out.days))
    out = out.join(pd.DataFrame({"bucket": b, "pnl": full, "year": YEAR}).pivot_table(index="bucket", columns="year", values="pnl", aggfunc="mean", observed=True).add_prefix("mean_wan_"))
    out.reset_index().assign(bucket=lambda x: x.bucket.astype(str)).to_csv(D / f"bucket_{name}.csv", index=False)
    print(f"-- full-cap daily PnL by {name}\n{out.round(2).to_string()}", flush=True)

COMPS = {f"comp {r:g}/x{s:g}/cd{cd}": (-1.5, r, s, cd) for r, s, cd in itertools.product([-1.0, -0.9, -0.8, -0.7, -0.6, -0.5], [0.1, 0.15, 0.2], [0, 2])}
COMPS["comp off"] = None
TSMCS = {"tsmc off": ()}
for t, s in itertools.product([0.0075, 0.01, 0.015, 0.02], [0.0, 0.1, 0.2, 0.4]):
    TSMCS[f"gap>{t*100:g}% x{s:g}"] = ((t, s),)
for s1, s2 in itertools.product([0.0, 0.1, 0.2], [0.0, 0.1, 0.2, 0.4]):
    TSMCS[f"1-2% x{s1:g} / >2% x{s2:g}"] = ((0.01, s1), (0.02, s2))
cs = {k: (np.ones(n) if v is None else comp_scale(*v)) for k, v in COMPS.items()}
ts = {k: tsmc_scale(v) for k, v in TSMCS.items()}
assert np.array_equal(np.minimum(cs["comp -0.7/x0.1/cd2"], ts["1-2% x0.1 / >2% x0.4"]), cal.prod_scale.to_numpy()), "production scale_by_date not reproduced"
rows = [{"comp": a, "tsmc": b, **metrics(np.minimum(cs[a], ts[b]))} for a in COMPS for b in TSMCS]
combo = pd.DataFrame(rows); combo.to_csv(D / "combo_grid.csv", index=False)
print("combo", len(combo), flush=True)

# walk-forward: pick on 2023-25 Sharpe, read 2026
pick = combo.sort_values("sharpe_23_25", ascending=False).iloc[0]
print(f"walk-forward pick on 2023-25: {pick.comp} + {pick.tsmc} | 23-25 sharpe {pick.sharpe_23_25:.2f} mdd {pick.mdd_wan_23_25:.0f} | 2026 sharpe {pick.sharpe_26:.2f} mdd {pick.mdd_wan_26:.0f}")

# per-year table for named candidates
CANDS = {"no gate": ("comp off", "tsmc off"),
         "production (comp+cd2+TSMC 1%x0.1/2%x0.4)": ("comp -0.7/x0.1/cd2", "1-2% x0.1 / >2% x0.4"),
         "comp only (current, no cd)": ("comp -0.7/x0.1/cd0", "tsmc off"),
         "TSMC only gap>1% x0.1": ("comp off", "gap>1% x0.1"),
         "comp + TSMC gap>1% x0.1 (no cd)": ("comp -0.7/x0.1/cd0", "gap>1% x0.1"),
         "comp + TSMC 1%x0.1/2%x0.4 (no cd)": ("comp -0.7/x0.1/cd0", "1-2% x0.1 / >2% x0.4"),
         "comp x0.2 + TSMC gap>1% x0.2 (no cd)": ("comp -0.7/x0.2/cd0", "gap>1% x0.2"),
         "comp + TSMC gap>0.75% x0.1 (no cd)": ("comp -0.7/x0.1/cd0", "gap>0.75% x0.1")}
yr = []
for name, (a, b) in CANDS.items():
    p = daily(np.minimum(cs[a], ts[b])); r = {"name": name, **metrics(np.minimum(cs[a], ts[b]))}
    for y in sorted(set(YEAR)):
        r |= {f"sharpe_{y}": stats(p, YEAR == y)["sharpe"], f"pnl_wan_{y}": stats(p, YEAR == y)["total_pnl_wan"], f"mdd_wan_{y}": stats(p, YEAR == y)["mdd_wan"]}
    yr.append(r)
yr = pd.DataFrame(yr); yr.to_csv(D / "candidates.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(yr.round(2).to_string())
