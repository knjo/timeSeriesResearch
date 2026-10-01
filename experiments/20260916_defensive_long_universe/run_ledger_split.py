"""Split the kept six-candidate taker-long ledger by the defensive stock-day list.

Input ledger: ``20260827_netfill_long_cover/artifacts/static_long_cover_v2/accepted_signals.parquet``
(static execution gate, one lot per accepted event, per-stock caps; no model).  PnL
convention is the ledger's own: ``AskPrice1 * (label_raw_bp - fee) / 10`` TWD per lot,
position ``AskPrice1 * 1000``.

Groups per stock-day: ``defensive`` (in the top-20% theme list), ``weak`` (bottom-20%
list), ``both`` (in both lists via different themes), ``middle`` (only in
non-quintile themes) and ``none`` (no valid theme).  The calendar is
restricted to the defensive list coverage so every accepted trade has a defined group.

Outputs (``artifacts/v1/ledger_split/``): summary by candidate x group, yearly split,
monthly split, day-level defensive-minus-rest contrast with HAC(5), and weight tiers
inside the defensive group.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from study_stats import calendar_sharpe, hac_mean_test, max_drawdown  # noqa: E402

PROJECT_ROOT = HERE.parents[4]
LEDGER = (
    PROJECT_ROOT
    / "src/research/intradaySignal/experiments/20260827_netfill_long_cover/artifacts/static_long_cover_v2/accepted_signals.parquet"
)
DEFENSIVE = HERE / "artifacts" / "v1" / "defensive_stock_daily.parquet"
OUT = HERE / "artifacts" / "v1" / "ledger_split"
FEE_BP = 19.3
GROUP_ORDER = ["defensive", "weak", "both", "middle", "none", "all", "non_defensive"]


def load_ledger() -> pl.DataFrame:
    ledger = pl.read_parquet(LEDGER).with_columns(
        pl.col("Date").str.strptime(pl.Date, "%Y%m%d").alias("decision_date"),
        (pl.col("AskPrice1") * 1000.0).alias("position"),
        (pl.col("AskPrice1") * pl.col("label_raw_bp") / 10.0).alias("gross_pnl"),
        (pl.col("AskPrice1") * (pl.col("label_raw_bp") - FEE_BP) / 10.0).alias("net_pnl"),
    )
    defensive = pl.read_parquet(DEFENSIVE)
    top = defensive.filter(pl.col("side") == "top").select(
        "decision_date", pl.col("quote_code").alias("QuoteCode"), pl.col("defensive_weight").alias("def_w")
    )
    bottom = defensive.filter(pl.col("side") == "bottom").select(
        "decision_date", pl.col("quote_code").alias("QuoteCode"), pl.col("defensive_weight").alias("weak_w")
    )
    middle = defensive.filter(pl.col("side") == "middle").select(
        "decision_date", pl.col("quote_code").alias("QuoteCode"), pl.lit(True).alias("in_middle")
    )
    covered = defensive.select(pl.col("decision_date").unique())
    ledger = (
        ledger.join(covered, on="decision_date", how="inner")
        .join(top, on=["decision_date", "QuoteCode"], how="left")
        .join(bottom, on=["decision_date", "QuoteCode"], how="left")
        .join(middle, on=["decision_date", "QuoteCode"], how="left")
        .with_columns(
            pl.when(pl.col("def_w").is_not_null() & pl.col("weak_w").is_not_null())
            .then(pl.lit("both"))
            .when(pl.col("def_w").is_not_null())
            .then(pl.lit("defensive"))
            .when(pl.col("weak_w").is_not_null())
            .then(pl.lit("weak"))
            .when(pl.col("in_middle")).then(pl.lit("middle"))
            .otherwise(pl.lit("none"))
            .alias("group")
        )
    )
    return ledger


def book_metrics(frame: pl.DataFrame, calendar: pl.Series) -> dict[str, float]:
    daily = (
        frame.group_by("decision_date")
        .agg(pl.col("net_pnl").sum(), pl.col("gross_pnl").sum(), pl.col("position").sum())
        .join(calendar.to_frame("decision_date"), on="decision_date", how="right")
        .fill_null(0.0)
        .sort("decision_date")
    )
    net = daily["net_pnl"].to_numpy()
    pos = float(frame["position"].sum())
    monthly = daily.group_by(daily["decision_date"].dt.strftime("%Y-%m").alias("m")).agg(pl.col("net_pnl").sum())
    return {
        "trades": float(frame.height),
        "stock_days": float(frame.select(["decision_date", "QuoteCode"]).unique().height),
        "active_days": float(frame["decision_date"].n_unique()),
        "gross_bp": float(frame["gross_pnl"].sum()) / pos * 1e4 if pos else np.nan,
        "net_bp": float(frame["net_pnl"].sum()) / pos * 1e4 if pos else np.nan,
        "mean_z": float(frame["label_z"].drop_nans().drop_nulls().mean()) if frame.height else np.nan,
        "resid_bp": float(frame["label_residual_bp"].drop_nans().drop_nulls().mean()) if frame.height else np.nan,
        "net_pnl_wan": float(net.sum()) / 1e4,
        "sharpe": calendar_sharpe(net),
        "mdd_wan": max_drawdown(net) / 1e4,
        "pos_months": float((monthly["net_pnl"] > 0).sum()),
        "months": float(monthly.height),
        "mean_daily_notional_wan": float(daily["position"].mean()) / 1e4,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = load_ledger()
    calendar = ledger["decision_date"].unique().sort()
    print(f"ledger rows {ledger.height}, calendar {calendar.min()}..{calendar.max()} ({calendar.len()} days)")
    print(ledger.group_by("group").len().sort("group"))

    rows = []
    for cand in sorted(ledger["candidate_id"].unique()):
        sub = ledger.filter(pl.col("candidate_id") == cand)
        views = {
            "all": sub,
            "defensive": sub.filter(pl.col("group") == "defensive"),
            "weak": sub.filter(pl.col("group") == "weak"),
            "both": sub.filter(pl.col("group") == "both"),
            "middle": sub.filter(pl.col("group") == "middle"),
            "none": sub.filter(pl.col("group") == "none"),
            "non_defensive": sub.filter(pl.col("group") != "defensive"),
        }
        for period, pf in (("full", None), ("2025", 2025), ("2026", 2026)):
            for g, v in views.items():
                vv = v if pf is None else v.filter(pl.col("decision_date").dt.year() == pf)
                cal = calendar if pf is None else calendar.filter(calendar.dt.year() == pf)
                if vv.is_empty():
                    continue
                m = book_metrics(vv, cal)
                rows.append({"candidate": cand, "period": period, "group": g, **m})
    summary = pl.DataFrame(rows)
    summary.write_csv(OUT / "summary.csv")

    # day-level contrast: within each day, capw net bp of defensive minus the rest
    contrast_rows = []
    for cand in sorted(ledger["candidate_id"].unique()):
        sub = ledger.filter(pl.col("candidate_id") == cand)
        day = (
            sub.with_columns((pl.col("group") == "defensive").alias("is_def"))
            .group_by(["decision_date", "is_def"])
            .agg(pl.col("net_pnl").sum(), pl.col("position").sum(), pl.col("label_z").mean().alias("z"))
            .with_columns((pl.col("net_pnl") / pl.col("position") * 1e4).alias("bp"))
            .pivot(on="is_def", index="decision_date", values=["bp", "z"])
        )
        cols = day.columns
        bp_t = [c for c in cols if c.startswith("bp") and c.endswith("true")]
        bp_f = [c for c in cols if c.startswith("bp") and c.endswith("false")]
        z_t = [c for c in cols if c.startswith("z") and c.endswith("true")]
        z_f = [c for c in cols if c.startswith("z") and c.endswith("false")]
        if not bp_t or not bp_f:
            continue
        both = day.drop_nulls([bp_t[0], bp_f[0]])
        for period, pf in (("full", None), ("2025", 2025), ("2026", 2026)):
            b = both if pf is None else both.filter(pl.col("decision_date").dt.year() == pf)
            if b.height < 5:
                continue
            d_bp = hac_mean_test((b[bp_t[0]] - b[bp_f[0]]).to_numpy())
            d_z = hac_mean_test((b[z_t[0]] - b[z_f[0]]).to_numpy())
            contrast_rows.append(
                {
                    "candidate": cand,
                    "period": period,
                    "days_with_both": b.height,
                    "delta_net_bp": d_bp["mean"],
                    "delta_net_bp_t": d_bp["t"],
                    "delta_net_bp_p": d_bp["p"],
                    "delta_z": d_z["mean"],
                    "delta_z_t": d_z["t"],
                    "delta_z_p": d_z["p"],
                    "pos_day_share": float(((b[bp_t[0]] - b[bp_f[0]]) > 0).mean()),
                }
            )
    contrast = pl.DataFrame(contrast_rows)
    contrast.write_csv(OUT / "day_contrast.csv")

    # weight tiers inside the defensive group (terciles of def_w per candidate, full period)
    tier_rows = []
    for cand in sorted(ledger["candidate_id"].unique()):
        d = ledger.filter((pl.col("candidate_id") == cand) & (pl.col("group") == "defensive"))
        if d.height < 30:
            continue
        q = d["def_w"].quantile(1 / 3), d["def_w"].quantile(2 / 3)
        d = d.with_columns(
            pl.when(pl.col("def_w") <= q[0]).then(pl.lit("w_low"))
            .when(pl.col("def_w") <= q[1]).then(pl.lit("w_mid"))
            .otherwise(pl.lit("w_high")).alias("tier")
        )
        for tier in ("w_low", "w_mid", "w_high"):
            t = d.filter(pl.col("tier") == tier)
            m = book_metrics(t, calendar)
            tier_rows.append({"candidate": cand, "tier": tier, **m})
    pl.DataFrame(tier_rows).write_csv(OUT / "defensive_weight_tiers.csv")

    # monthly net bp by group for the promoted long candidate and the all-sample baseline
    monthly = (
        ledger.with_columns(pl.col("decision_date").dt.strftime("%Y-%m").alias("month"))
        .group_by(["candidate_id", "group", "month"])
        .agg(pl.col("net_pnl").sum(), pl.col("position").sum(), pl.len().alias("trades"))
        .with_columns((pl.col("net_pnl") / pl.col("position") * 1e4).alias("net_bp"))
        .sort(["candidate_id", "group", "month"])
    )
    monthly.write_csv(OUT / "monthly.csv")

    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(40)
    show = summary.select(
        "candidate", "period", "group", "trades", "stock_days", "gross_bp", "net_bp", "mean_z",
        "resid_bp", "net_pnl_wan", "sharpe", "mdd_wan", "pos_months", "months",
    ).with_columns(pl.col(pl.Float64).round(3))
    print(show.filter(pl.col("period") == "full").sort(["candidate", "group"]))
    print(show.filter(pl.col("period") != "full").filter(pl.col("group").is_in(["all", "defensive", "non_defensive", "weak"])).sort(["candidate", "period", "group"]))
    print(contrast.with_columns(pl.col(pl.Float64).round(4)))
    print(pl.DataFrame(tier_rows).select("candidate", "tier", "trades", "gross_bp", "net_bp", "mean_z", "sharpe").with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
