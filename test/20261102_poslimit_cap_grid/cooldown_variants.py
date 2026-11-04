"""posLimit 400 x cap 7000: is the post-stop cooldown worth keeping, and in which form?
  run   = production apply_cooldown: after a restricted run that contained a stop day ends, the next N base==1 days are reduce_scale
  stop  = proposed: the N trading days after any stop day are capped at reduce_scale; reduce-band days count toward N
  none  = no cooldown
Also buckets full-cap daily PnL by trading days since the last stop day, split by the comp state of that day.
Usage: uv run python cooldown_variants.py <gate_tune_out_dir>
"""
import sys
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
STOP, REDUCE = -1.5, -0.7


def daily(scale):
    return P[np.arange(n), [SIDX[round(float(s), 2)] for s in scale]]


def stats(p, m=None):
    x = p if m is None else p[m]; c = np.cumsum(x); dd = c - np.maximum.accumulate(c)
    return {"total_pnl_wan": x.sum(), "mdd_wan": dd.min(), "sharpe": x.mean() / x.std(ddof=1) * np.sqrt(252)}


def post_stop_cooldown(base, reduce_scale, days):
    out, left = [], 0
    for b in base:
        if b <= 0:
            out.append(0.0); left = days; continue
        if left > 0:
            out.append(min(float(b), reduce_scale)); left -= 1
        else:
            out.append(float(b))
    return out


def comp_scale(kind, reduce_scale, days):
    base = [scale_for_comp(c, STOP, REDUCE, reduce_scale) for c in gd.comp]
    s = {"none": base, "run": apply_cooldown(base, reduce_scale, days)[0], "stop": post_stop_cooldown(base, reduce_scale, days)}[kind]
    return pd.Series(s, index=gd.index).reindex(dates).fillna(1.0).to_numpy()


def tsmc_scale(tiers):
    out = np.ones(n)
    for thr, s in sorted(tiers):
        out[np.nan_to_num(gap, nan=-1) > thr] = s
    return out


TS = tsmc_scale(((0.01, 0.1), (0.02, 0.4)))

# 1. days since the last stop day (gate calendar), by comp state of that day
base_all = pd.Series([scale_for_comp(c, STOP, REDUCE, 0.1) for c in gd.comp], index=gd.index)
since = []; k = None
for b in base_all:
    if b <= 0:
        k = 0
    elif k is not None:
        k += 1
    since.append(k)
since = pd.Series(since, index=gd.index, dtype="float").reindex(dates).to_numpy()
state = pd.Series(["stop" if b <= 0 else ("reduce" if b < 1 else "trade") for b in base_all], index=gd.index).reindex(dates).fillna("trade").to_numpy()
full = P[:, -1]; at_r = P[:, SIDX[0.1]]
b = pd.DataFrame({"since": since, "state": state, "full": full, "at_0.1": at_r, "comp": cal.comp, "year": YEAR})
b["since_b"] = pd.cut(b.since, [-1, 0, 1, 2, 3, 5, 10, 9999], labels=["0 (stop)", "1", "2", "3", "4-5", "6-10", ">10"])
bucket = b.groupby(["since_b", "state"], observed=True).agg(days=("full", "size"), mean_full_wan=("full", "mean"), std_full_wan=("full", "std"),
                                                             total_full_wan=("full", "sum"), total_at_0p1_wan=("at_0.1", "sum"))
bucket["t_stat"] = bucket.mean_full_wan / (bucket.std_full_wan / np.sqrt(bucket.days))
bucket["gain_if_full_vs_0p1_wan"] = bucket.total_full_wan - bucket.total_at_0p1_wan
yr = b[b.since.between(1, 2)].pivot_table(index=["since_b", "state"], columns="year", values="full", aggfunc="mean", observed=True).add_prefix("mean_full_wan_")
bucket = bucket.join(yr)
bucket.reset_index().to_csv(D / "cooldown_bucket.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("-- full-cap daily PnL by trading days since last stop day x comp state of the day"); print(bucket.round(2).to_string())
# individual post-stop trade-state days within 2 days (what the cooldown actually touches)
touched = b[(b.since.between(1, 2)) & (b.state == "trade")]
print(f"\ncooldown-touched days (since 1-2, comp recovered): {len(touched)} | full-cap sum {touched.full.sum():.0f} | at x0.1 sum {touched['at_0.1'].sum():.0f} | by year:")
print(touched.groupby("year").agg(days=("full", "size"), full_wan=("full", "sum"), at_0p1_wan=("at_0.1", "sum"), mean_full_wan=("full", "mean")).round(1).to_string())

# 2. variants
rows = []
for rs in (0.1, 0.15):
    for kind, days in [("none", 0), ("run", 1), ("run", 2), ("run", 3), ("stop", 1), ("stop", 2), ("stop", 3), ("stop", 5)]:
        cs = comp_scale(kind, rs, days)
        for with_tsmc, sc in (("comp only", cs), ("comp + TSMC", np.minimum(cs, TS))):
            p = daily(sc); r = {"reduce_scale": rs, "cooldown": kind, "days": days, "layer": with_tsmc, **stats(p)}
            r |= {f"{k}_23_25": v for k, v in stats(p, ~Y26).items() if k != "total_pnl_wan"} | {f"{k}_26": v for k, v in stats(p, Y26).items() if k != "total_pnl_wan"}
            r |= {f"sharpe_{y}": stats(p, YEAR == y)["sharpe"] for y in sorted(set(YEAR))}
            r |= {"pnl_p10_wan": np.percentile(p, 10), "restricted_days": int((sc < 1).sum()), "cooldown_days_used": int(((cs < 1) & (comp_scale("none", rs, 0) == 1)).sum())}
            rows.append(r)
out = pd.DataFrame(rows); out.to_csv(D / "cooldown_variants.csv", index=False)
cols = ["reduce_scale", "cooldown", "days", "cooldown_days_used", "total_pnl_wan", "mdd_wan", "sharpe", "pnl_p10_wan", "sharpe_23_25", "mdd_wan_23_25", "sharpe_26", "mdd_wan_26", "sharpe_2023", "sharpe_2024", "sharpe_2025", "sharpe_2026"]
for layer in ("comp only", "comp + TSMC"):
    print(f"\n-- {layer}"); print(out[out.layer == layer][cols].round(2).to_string(index=False))

# 3. which days differ between run-2 and stop-2 (reduce_scale 0.15)
r2, s2 = comp_scale("run", 0.15, 2), comp_scale("stop", 0.15, 2)
d = np.where(r2 != s2)[0]
diff = pd.DataFrame({"date": np.array(dates)[d], "comp": cal.comp.to_numpy()[d].round(2), "since_stop": since[d], "run2": r2[d], "stop2": s2[d], "pnl_full_wan": full[d].round(1), "pnl_x0.15_wan": P[d, SIDX[0.15]].round(1)})
diff.to_csv(D / "cooldown_diff_days.csv", index=False)
print(f"\n-- days where run-2 and stop-2 differ: {len(diff)} (run2 only restricted: {(diff.run2 < diff.stop2).sum()}, stop2 only: {(diff.stop2 < diff.run2).sum()})")
print(diff.to_string(index=False))
