"""Paired daily inference, cost sensitivity and complete trial accounting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from spec import candidates


def metrics(a):
    a = np.asarray(a, dtype=float)
    wealth = np.r_[0., np.cumsum(a)]
    mdd = np.min(wealth-np.maximum.accumulate(wealth))
    return {"pnl_wan": a.sum()/1e4, "daily_mean_wan": a.mean()/1e4,
            "sharpe": a.mean()/a.std(ddof=1)*np.sqrt(252) if len(a)>1 and a.std(ddof=1)>1e-10 else None,
            "mdd_wan": mdd/1e4, "worst_day_wan": a.min()/1e4,
            "cvar05_wan": np.sort(a)[:max(1, int(np.ceil(len(a)*.05)))].mean()/1e4}


def block_inference(diffs, families, *, draws=2000, block=5, seed=20260922):
    """Paired circular-block bootstrap and centered family max-t single step."""
    n, p = diffs.shape
    rng = np.random.default_rng(seed)
    b = min(block, n)
    starts = rng.integers(0, n, size=(draws, int(np.ceil(n/b))))
    ix = ((starts[..., None]+np.arange(b)) % n).reshape(draws, -1)[:, :n]
    boots = np.empty((draws, p))
    for j in range(p):
        boots[:, j] = diffs[:, j][ix].mean(axis=1)
    means = diffs.mean(axis=0)
    se = boots.std(axis=0, ddof=1)
    scaled = (boots-means) / np.maximum(se, 1e-12)
    actual = means / np.maximum(se, 1e-12)
    lo, hi = np.quantile(boots, [.025, .975], axis=0)
    output = []
    for j in range(p):
        family_ix = np.array([k for k, f in enumerate(families) if f == families[j]])
        nullmax = scaled[:, family_ix].max(axis=1)
        output.append({"delta_day_wan": means[j]/1e4, "delta_ci_low_wan": lo[j]/1e4,
                       "delta_ci_high_wan": hi[j]/1e4,
                       "family_adjusted_p": (1+np.sum(nullmax >= actual[j]))/(draws+1)})
    centered = diffs-diffs.mean(axis=0)
    norm = np.linalg.norm(centered, axis=0)
    active = centered[:, norm>1e-9]
    if active.shape[1]:
        corr = np.corrcoef(active.T) if active.shape[1]>1 else np.ones((1,1))
        eig = np.linalg.eigvalsh(corr)
        effective = float(eig.sum()**2/np.sum(eig**2))
    else:
        effective = 0.
    return output, effective


def table(frame, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"]*len(cols)) + "|\n"
    rows = []
    for row in frame.select(cols).iter_rows():
        rows.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return head+"\n".join(rows)


def summarize(output: Path):
    specs = candidates()
    rows = pl.scan_parquet(output / "days" / "*.parquet")
    daily = rows.group_by("candidate", "Date").agg(
        pl.len().alias("n_trades"), pl.col("notional").sum(), pl.col("pnl_twd").sum(),
        pl.col("stress_cost_twd").sum(), pl.col("net_bp").mean().alias("trade_mean_net_bp"),
        pl.col("reset_exit").sum(), pl.col("reset_observed").sum(), pl.col("waited_for_ask").sum(),
        pl.col("unfilled_close_mark").sum(), (pl.col("reason") == 1).sum().alias("sl_count"),
        (pl.col("reason") == 2).sum().alias("tp_count"),
        ((pl.col("reason") == 2) & (pl.col("gross_bp") < 0)).sum().alias("tp_loss_count"),
        (pl.col("quote_age_seconds") > 60).sum().alias("stale_exit_count"),
    ).with_columns((pl.col("pnl_twd")/pl.col("notional")*1e4).alias("day_net_bp")).collect().sort("candidate", "Date")
    daily.write_csv(output / "daily.csv")
    names = [s.name for s in specs]
    arrays = {name: daily.filter(pl.col("candidate") == name).sort("Date") for name in names}
    dates = arrays[names[0]]["Date"].to_numpy()
    assert all(np.array_equal(arrays[name]["Date"].to_numpy(), dates) for name in names)
    diffs = np.column_stack([arrays[s.name]["pnl_twd"].to_numpy()-arrays[s.baseline]["pnl_twd"].to_numpy() for s in specs])
    # All moving-vol comparisons share a family, including the alternate average.
    families = ["volatility" if s.family == "sensitivity" else s.family for s in specs]
    inference, effective = block_inference(diffs, families)
    records = []
    for j, spec in enumerate(specs):
        d = arrays[spec.name]
        pnl = d["pnl_twd"].to_numpy()
        record = {"candidate": spec.name, "family": spec.family, "baseline": spec.baseline,
                  "n_days": d.height, "n_trades": int(d["n_trades"].sum()), **metrics(pnl), **inference[j],
                  "net_pos_bp": pnl.sum()/d["notional"].sum()*1e4,
                  "net_equal_day_bp": d["day_net_bp"].mean(),
                  "net_trade_bp": (d["trade_mean_net_bp"]*d["n_trades"]).sum()/d["n_trades"].sum(),
                  "delta_wan": diffs[:, j].sum()/1e4,
                  "positive_delta_days": int(np.sum(diffs[:,j] > 1e-8)),
                  "negative_delta_days": int(np.sum(diffs[:,j] < -1e-8)),
                  "affected_days": int(np.sum(np.abs(diffs[:,j]) > 1e-8)),
                  "stress_pnl_wan": (pnl-d["stress_cost_twd"].to_numpy()).sum()/1e4,
                  "stress_delta_wan": ((pnl-d["stress_cost_twd"].to_numpy())-(arrays[spec.baseline]["pnl_twd"].to_numpy()-arrays[spec.baseline]["stress_cost_twd"].to_numpy())).sum()/1e4,
                  **{c: int(d[c].sum()) for c in ["reset_exit", "reset_observed", "waited_for_ask", "unfilled_close_mark", "sl_count", "tp_count", "tp_loss_count", "stale_exit_count"]}}
        b = arrays[spec.baseline]
        valid = (d["unfilled_close_mark"].to_numpy() == 0) & (b["unfilled_close_mark"].to_numpy() == 0)
        record["ex_unfilled_days"] = int(valid.sum())
        record["ex_unfilled_delta_wan"] = diffs[valid,j].sum()/1e4
        records.append(record)
    summary = pl.DataFrame(records)
    summary.write_csv(output / "summary.csv")
    months = daily.with_columns(pl.col("Date").str.slice(0,6).alias("month")).group_by("candidate", "month").agg(pl.col("pnl_twd").sum(), pl.col("n_trades").sum()).sort("candidate", "month")
    months.write_csv(output / "monthly.csv")
    periods = []
    for spec in specs:
        d, b = arrays[spec.name], arrays[spec.baseline]
        for label, mask in [("2025", dates < "20260101"), ("2026", dates >= "20260101"),
                            ("pre_20260701", dates < "20260701"), ("from_20260701", dates >= "20260701")]:
            if not np.any(mask):
                continue
            a = d["pnl_twd"].to_numpy()[mask]
            base = b["pnl_twd"].to_numpy()[mask]
            periods.append({"candidate":spec.name,"period":label,"n_days":int(mask.sum()),
                            "delta_wan": (a-base).sum()/1e4, **metrics(a)})
    pl.DataFrame(periods).write_csv(output / "periods.csv")
    comparisons = []
    for spec in specs:
        # Same-day decomposition into entries before vs after cascade activation.
        if spec.k and spec.scope == 0:
            base = rows.filter(pl.col("candidate") == "ref08").select("trade_id", pl.col("pnl_twd").alias("base_pnl"))
            q = (rows.filter(pl.col("candidate") == spec.name).join(base, on="trade_id")
                 .with_columns((pl.col("pnl_twd")-pl.col("base_pnl")).alias("delta")))
            g = q.group_by("entered_before_activation").agg(
                pl.len().alias("n"), (pl.col("delta").abs()>1e-8).sum().alias("changed_n"),
                pl.col("delta").sum().alias("delta_twd"), pl.col("reset_exit").sum(),
            ).collect().with_columns(pl.lit(spec.name).alias("candidate"))
            comparisons.append(g)
    if comparisons:
        pl.concat(comparisons).write_csv(output / "cascade_decomposition.csv")
    audit = {
        "raw_trial_count": len(specs), "effective_trials_participation_ratio": effective,
        "bootstrap_draws": 2000, "block_length": 5, "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "selection_bias": "paired centered family max-t; exploratory family, no prospective winner claim",
        "input_label_mae_vs_ref08_formula": None,
    }
    signal = pl.read_parquet(output / "input_signals.parquet")
    exit_label = pl.when(pl.col("FutureHigh") > pl.col("RefPrice")*1.08).then(pl.col("RefPrice")*1.08+2*pl.col("TickSize")).otherwise(pl.col("Close"))
    audit["input_label_mae_vs_ref08_formula"] = signal.select((pl.col("TakerSell_CloseBP")-(pl.col("BidPrice1")-exit_label)/pl.col("BidPrice1")*1e4).abs().mean()).item()
    audit["input_label_net_pnl_wan"] = signal.select((pl.col("BidPrice1")*1000*(pl.col("TakerSell_CloseBP")-19.3)/1e4).sum()).item()/1e4
    (output / "audit.json").write_text(json.dumps(audit, indent=2))
    text = ["# Dynamic exit replay summary", "", f"{len(dates)} days, {signal.height} original entries, {len(specs)} frozen variants.",
            "All PnL is after 19.3bp. Delta is relative to the baseline column; TWD units are ten-thousands.", "",
            table(summary, ["candidate","baseline","pnl_wan","delta_wan","net_pos_bp","sharpe","mdd_wan","cvar05_wan","reset_exit","unfilled_close_mark"])]
    (output / "SUMMARY.md").write_text("\n".join(text))
    plot_summary(daily, output)
    print(summary.select("candidate","pnl_wan","delta_wan","sharpe","mdd_wan","reset_exit","unfilled_close_mark"))


def plot_summary(daily, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import datetime
    selections = ["ref08", "ref05", "ref06", "cascade_k3_to05", "cascade_k5_to05", "cascade_k3_to06", "cascade_k5_to06"]
    fig, axes = plt.subplots(2,1, figsize=(13,10))
    for ax, names in zip(axes, [selections, ["ref08", "vol_entry_2p5", "vol_clock_2p5", "vol_rolling_2p5", "vol_near_sl_2p5", "vol_near_tp_2p5", "vol_near_both_2p5"]]):
        for name in names:
            d = daily.filter(pl.col("candidate") == name).sort("Date")
            ax.plot([datetime.strptime(v,"%Y%m%d") for v in d["Date"]], np.cumsum(d["pnl_twd"].to_numpy())/1e4, label=name)
        ax.set_ylabel("Cumulative net PnL (TWD 10,000)")
        ax.legend(ncol=3, fontsize=8)
        ax.grid(alpha=.2)
    axes[0].set_title("Stop-count-triggered tightening")
    axes[1].set_title("Entry vs dynamic VWAP barriers, span = 2.5")
    fig.tight_layout()
    fig.savefig(output / "equity.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    summarize(p.parse_args().output)
