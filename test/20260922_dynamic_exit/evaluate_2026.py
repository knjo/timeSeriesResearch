"""Primary 2026 evaluation after the pre-20251224 OTC RefPrice source audit.

The same frozen replay is reused. No candidate, entry, or exit is refitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from spec import candidates
from summarize import block_inference, metrics, plot_summary, table


def evaluate(output):
    target = output / "primary_2026"
    target.mkdir(exist_ok=True)
    daily = pl.read_csv(output / "daily.csv", schema_overrides={"Date": pl.String}).filter(pl.col("Date") >= "20260101")
    daily.write_csv(target / "daily.csv")
    specs = candidates()
    arrays = {s.name: daily.filter(pl.col("candidate") == s.name).sort("Date") for s in specs}
    dates = arrays["ref08"]["Date"].to_numpy()
    diff = np.column_stack([arrays[s.name]["pnl_twd"].to_numpy()-arrays[s.baseline]["pnl_twd"].to_numpy() for s in specs])
    inference, effective = block_inference(diff, ["volatility" if s.family == "sensitivity" else s.family for s in specs])
    records = []
    monthly = daily.with_columns(pl.col("Date").str.slice(0,6).alias("month")).group_by("candidate", "month").agg(pl.col("pnl_twd").sum()).sort("candidate", "month")
    monthly.write_csv(target / "monthly.csv")
    month_arrays = {s.name: monthly.filter(pl.col("candidate") == s.name)["pnl_twd"].to_numpy() for s in specs}
    monthly_delta, paired_days = [], []
    for j, s in enumerate(specs):
        d, b = arrays[s.name], arrays[s.baseline]
        pnl = d["pnl_twd"].to_numpy()
        valid = (d["unfilled_close_mark"].to_numpy() == 0) & (b["unfilled_close_mark"].to_numpy() == 0)
        month_diff = month_arrays[s.name]-month_arrays[s.baseline]
        records.append({"candidate":s.name, "family":s.family, "baseline":s.baseline,
                        "n_days":d.height, "n_trades":int(d["n_trades"].sum()), **metrics(pnl), **inference[j],
                        "delta_wan":diff[:,j].sum()/1e4, "net_pos_bp":pnl.sum()/d["notional"].sum()*1e4,
                        "net_equal_day_bp":d["day_net_bp"].mean(),
                        "net_trade_bp":(d["trade_mean_net_bp"]*d["n_trades"]).sum()/d["n_trades"].sum(),
                        "positive_delta_days":int(np.sum(diff[:,j]>1e-8)), "negative_delta_days":int(np.sum(diff[:,j]<-1e-8)),
                        "affected_days":int(np.sum(np.abs(diff[:,j])>1e-8)),
                        "positive_months":int(np.sum(month_diff>1e-8)), "negative_months":int(np.sum(month_diff<-1e-8)),
                        "stress_pnl_wan":(pnl-d["stress_cost_twd"].to_numpy()).sum()/1e4,
                        "stress_delta_wan":(diff[:,j]-d["stress_cost_twd"].to_numpy()+b["stress_cost_twd"].to_numpy()).sum()/1e4,
                        "ex_unfilled_days":int(valid.sum()), "ex_unfilled_delta_wan":diff[valid,j].sum()/1e4,
                        **{c:int(d[c].sum()) for c in ["reset_exit", "reset_observed", "waited_for_ask", "unfilled_close_mark", "sl_count", "tp_count", "tp_loss_count", "stale_exit_count"]}})
        monthly_delta.append(monthly.filter(pl.col("candidate") == s.name).with_columns(pl.Series("delta_wan", month_diff/1e4)))
        paired_days.append(pl.DataFrame({"Date":dates,"candidate":s.name,"baseline":s.baseline,"delta_twd":diff[:,j]}))
    summary = pl.DataFrame(records)
    summary.write_csv(target / "summary.csv")
    pl.concat(monthly_delta).write_csv(target / "monthly_delta.csv")
    pl.concat(paired_days).write_csv(target / "paired_days.csv")
    # Collect only columns needed for full row-level integrity and decomposition.
    rows = (pl.scan_parquet(output / "days" / "*.parquet").filter(pl.col("Date") >= "20260101")
            .select("trade_id", "Date", "QuoteCode", "candidate", "family", "baseline", "TransTime", "exit_time",
                    "reason", "exit_price", "BidPrice1", "RefPrice", "txf_residual_vol_to_1330", "pnl_twd",
                    "reset_exit", "reset_observed", "unfilled_close_mark", "activation_us", "entered_before_activation",
                    "sl_at_exit", "tp_at_exit").collect())
    expected = rows.filter(pl.col("candidate") == "ref08").height
    support = rows.group_by("candidate").agg(pl.len().alias("n"),pl.col("trade_id").n_unique().alias("ids"))
    assert support.filter((pl.col("n") != expected) | (pl.col("ids") != expected)).is_empty()
    assert rows.filter(pl.col("exit_time") <= pl.col("TransTime")).is_empty()
    seconds = (pl.col("exit_time").dt.hour().cast(pl.Int64)*3600
               +pl.col("exit_time").dt.minute().cast(pl.Int64)*60+pl.col("exit_time").dt.second())
    assert rows.filter((seconds >= 13*3600+25*60) & (seconds < 13*3600+30*60)).is_empty()
    volrows = rows.filter(pl.col("family").is_in(["volatility","sensitivity"]))
    assert volrows.filter(pl.col("sl_at_exit") > pl.col("RefPrice")*1.08+1e-8).is_empty()
    integrity = {"same_entries_per_candidate":expected,"candidates":support.height,"all_exits_strictly_after_entry":True,
                 "no_vol_hard_sl_violations":True,"no_closing_auction_immediate_fills":True,"near_exit_later_than_entry_baseline":0,
                 "cascade_pre_activation_changes":0,"cascade_scope_decomposition_error_twd":0.0}
    anatomy = []
    for s in specs:
        r = rows.filter(pl.col("candidate") == s.name)
        b = rows.filter(pl.col("candidate") == s.baseline).select("trade_id",pl.col("pnl_twd").alias("base_pnl"),pl.col("exit_time").alias("base_exit"))
        r = r.join(b,on="trade_id").with_columns((pl.col("pnl_twd")-pl.col("base_pnl")).alias("delta"))
        if s.mode in (2,4,5,6,7):
            bad = r.filter(pl.col("exit_time")>pl.col("base_exit")).height
            integrity["near_exit_later_than_entry_baseline"] += bad
            assert not bad, s.name
        if s.k:
            bad = r.filter((pl.col("base_exit").cast(pl.Int64)<pl.col("activation_us")) & (pl.col("delta").abs()>1e-7)).height
            integrity["cascade_pre_activation_changes"] += bad
            assert not bad, s.name
        if s.k and s.scope == 0:
            anatomy.append(r.group_by("entered_before_activation").agg(
                pl.len().alias("n"),(pl.col("delta").abs()>1e-8).sum().alias("changed_n"),
                pl.col("delta").sum().alias("delta_twd"),pl.col("reset_exit").sum(),
            ).with_columns(pl.lit(s.name).alias("candidate")))
        if s.mode in (4,5,6,7):
            sl0 = np.minimum(r["BidPrice1"].to_numpy()*(1+s.span*r["txf_residual_vol_to_1330"].to_numpy()),r["RefPrice"].to_numpy()*1.08)
            tp0 = r["BidPrice1"].to_numpy()*(1-s.span*r["txf_residual_vol_to_1330"].to_numpy())
            assert np.all(r["sl_at_exit"].to_numpy()<=sl0+1e-8)
            assert np.all(r["tp_at_exit"].to_numpy()>=tp0-1e-8)
    pl.concat(anatomy).write_csv(target / "cascade_decomposition.csv")
    for k in (3,5):
        for pct in (5,6):
            name=f"cascade_k{k}_to{pct:02d}"
            ordered=lambda n: rows.filter(pl.col("candidate")==n).sort("trade_id")["pnl_twd"].to_numpy()
            err=np.max(np.abs((ordered(name)-ordered("ref08"))-(ordered(name+"_existing")+ordered(name+"_new")-2*ordered("ref08"))))
            integrity["cascade_scope_decomposition_error_twd"]=max(integrity["cascade_scope_decomposition_error_twd"],float(err))
            assert err<1e-6
    integrity["inverted_band_rows"] = volrows.filter(pl.col("sl_at_exit")>=0).filter(pl.col("sl_at_exit")<=pl.col("tp_at_exit")).height
    (target / "integrity.json").write_text(json.dumps(integrity,indent=2))
    selected=["ref08","cascade_k3_to05","cascade_k3_to06","cascade_k5_to05","cascade_k5_to06",
              "vol_entry_2p5","vol_rolling_2p5","vol_near_sl_2p5","vol_near_tp_2p5","vol_near_both_2p5"]
    # Concrete immediate-reset examples, with both PnL and the barrier actually in force.
    examples=rows.filter(pl.col("reset_exit") & pl.col("candidate").is_in(selected)).sort("Date","exit_time")
    examples.head(100).write_csv(target / "reset_examples.csv")
    (target / "evaluation_manifest.json").write_text(json.dumps({
        "source_fingerprint":json.loads((output/"manifest.json").read_text())["fingerprint"],
        "analysis_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluation_period":[str(dates.min()),str(dates.max())],
        "primary_period_reason":"2026 notebook horizon; known OTC RefPrice defect before 20251224, found before full replay outcomes",
        "raw_trial_count":len(specs),"effective_trials_participation_ratio":effective,
        "bootstrap_draws":2000,"block_length":5,"candidates":[asdict(s) for s in specs],
        "status":"RESEARCH_DIAGNOSTIC_ONLY",
    },indent=2))
    (target / "SUMMARY.md").write_text("# Primary 2026 paired exit comparison\n\n"+table(summary,["candidate","baseline","pnl_wan","delta_wan","net_pos_bp","sharpe","mdd_wan","cvar05_wan","positive_months","family_adjusted_p"]))
    plot_summary(daily,target)
    print(summary.filter(pl.col("candidate").is_in(selected)).select("candidate","pnl_wan","delta_wan","sharpe","mdd_wan","family_adjusted_p"))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    evaluate(parser.parse_args().output)
