"""Shared-pool bias of the defensive stock-day list on the long taker eligible pool.

Reads ``artifacts/v1/pool_stockday/*.parquet`` (stock-day sums of the pinned long
label) and the defensive list, then reports for each stock-day group
(``defensive`` / ``weak`` / ``both`` / ``middle`` / ``none``):

- equal-day ``Delta = mean(label | group) - mean(label | full pool)`` for raw bp,
  residual bp and z, row-weighted inside the day exactly like the framework's
  shared-pool primary estimand, with HAC(5) inference over active days;
- the same contrast with every stock-day equally weighted (composition check);
- absolute equal-day levels of the group, so a positive ``Delta`` is not mistaken
  for a positive taker-long return;
- yearly / monthly sign stability, selection rate and stock coverage;
- pairwise day-level contrasts defensive-weak and weak-none (score effect versus
  plain theme-membership effect) and defensive-weight terciles.

Outputs go to ``artifacts/v1/pool_bias/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from study_stats import hac_mean_test  # noqa: E402

POOL_DIR = HERE / "artifacts" / "v1" / "pool_stockday"
DEFENSIVE = HERE / "artifacts" / "v1" / "defensive_stock_daily.parquet"
OUT = HERE / "artifacts" / "v1" / "pool_bias"
LABELS = ("label_raw_bp", "label_residual_bp", "label_z")
GROUPS = ("defensive", "weak", "both", "middle", "none")


def load_pool() -> pl.DataFrame:
    frames = [pl.read_parquet(p) for p in sorted(POOL_DIR.glob("*.parquet"))]
    pool = pl.concat(frames).with_columns(
        pl.col("Date").str.strptime(pl.Date, "%Y%m%d").alias("decision_date")
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
    return (
        pool.join(covered, on="decision_date", how="inner")
        .join(top, on=["decision_date", "QuoteCode"], how="left")
        .join(bottom, on=["decision_date", "QuoteCode"], how="left")
        .join(middle, on=["decision_date", "QuoteCode"], how="left")
        .with_columns(
            pl.when(pl.col("def_w").is_not_null() & pl.col("weak_w").is_not_null()).then(pl.lit("both"))
            .when(pl.col("def_w").is_not_null()).then(pl.lit("defensive"))
            .when(pl.col("weak_w").is_not_null()).then(pl.lit("weak"))
            .when(pl.col("in_middle")).then(pl.lit("middle"))
            .otherwise(pl.lit("none"))
            .alias("group")
        )
    )


def day_means(frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Row-weighted and stock-equal daily means per label for the given grouping."""

    exprs = [pl.col("n_rows").sum().alias("rows"), pl.col("QuoteCode").n_unique().alias("stocks")]
    for label in LABELS:
        exprs.append((pl.col(f"sum_{label}").sum() / pl.col(f"n_{label}").sum()).alias(f"rw_{label}"))
        exprs.append((pl.col(f"sum_{label}") / pl.col(f"n_{label}")).mean().alias(f"se_{label}"))
    return frame.group_by(by).agg(exprs)


def contrast_table(delta: pl.DataFrame, value_cols: list[str], key_cols: list[str]) -> pl.DataFrame:
    rows = []
    for keys, sub in delta.group_by(key_cols, maintain_order=True):
        base = dict(zip(key_cols, keys))
        for period, pf in (("full", None), ("2025", 2025), ("2026", 2026)):
            s = sub if pf is None else sub.filter(pl.col("decision_date").dt.year() == pf)
            if s.height < 5:
                continue
            row = {**base, "period": period, "days": s.height}
            for col in value_cols:
                v = s[col].to_numpy()
                h = hac_mean_test(v)
                months = s.group_by(s["decision_date"].dt.strftime("%Y-%m").alias("m")).agg(pl.col(col).mean())
                row[f"{col}"] = h["mean"]
                row[f"{col}_t"] = h["t"]
                row[f"{col}_p"] = h["p"]
                row[f"{col}_posday"] = float(np.mean(v > 0))
                row[f"{col}_posmonth"] = f"{int((months[col] > 0).sum())}/{months.height}"
            rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = load_pool()
    dates = pool["decision_date"].unique().sort()
    print(f"pool stock-days {pool.height}, dates {dates.min()}..{dates.max()} ({dates.len()})")
    print(pool.group_by("group").agg(pl.len().alias("stock_days"), pl.col("n_rows").sum().alias("rows")).sort("group"))

    full = day_means(pool, ["decision_date"]).rename({c: f"pool_{c}" for c in ["rows", "stocks"] + [f"rw_{l}" for l in LABELS] + [f"se_{l}" for l in LABELS]})
    grp = day_means(pool, ["decision_date", "group"])
    joined = grp.join(full, on="decision_date", how="inner")
    delta_cols = []
    for label in LABELS:
        joined = joined.with_columns(
            (pl.col(f"rw_{label}") - pl.col(f"pool_rw_{label}")).alias(f"d_rw_{label}"),
            (pl.col(f"se_{label}") - pl.col(f"pool_se_{label}")).alias(f"d_se_{label}"),
        )
        delta_cols += [f"d_rw_{label}", f"d_se_{label}"]
    joined = joined.with_columns((pl.col("rows") / pl.col("pool_rows")).alias("selection_rate"))
    joined.write_parquet(OUT / "daily_group_effects.parquet")

    # shared-pool contrast per group
    shared = contrast_table(joined.sort(["group", "decision_date"]), delta_cols, ["group"])
    shared.write_csv(OUT / "shared_pool_contrast.csv")
    # absolute levels per group (equal-day, row-weighted)
    levels = contrast_table(joined.sort(["group", "decision_date"]), [f"rw_{l}" for l in LABELS], ["group"])
    levels_pool = contrast_table(full.with_columns(pl.lit("pool").alias("group")).rename({f"pool_rw_{l}": f"rw_{l}" for l in LABELS}).sort("decision_date"), [f"rw_{l}" for l in LABELS], ["group"])
    levels = pl.concat([levels, levels_pool], how="diagonal_relaxed")
    levels.write_csv(OUT / "absolute_levels.csv")
    coverage = (
        joined.group_by("group")
        .agg(
            pl.col("decision_date").n_unique().alias("active_days"),
            pl.col("stocks").mean().alias("mean_stocks"),
            pl.col("selection_rate").mean().alias("mean_selection_rate"),
        )
        .sort("group")
    )
    coverage.write_csv(OUT / "coverage.csv")

    # pairwise: defensive - weak, weak - none, defensive - none (row-weighted means, same day)
    wide = joined.pivot(on="group", index="decision_date", values=[f"rw_{l}" for l in LABELS])
    pair_rows = []
    for a, b in (("defensive", "weak"), ("defensive", "middle"), ("weak", "middle"), ("middle", "none"), ("defensive", "none")):
        cols = {}
        for label in LABELS:
            ca, cb = f"rw_{label}_{a}", f"rw_{label}_{b}"
            if ca in wide.columns and cb in wide.columns:
                cols[f"{label}"] = (pl.col(ca) - pl.col(cb))
        w = wide.drop_nulls([f"rw_{LABELS[0]}_{a}", f"rw_{LABELS[0]}_{b}"]).with_columns(
            [e.alias(k) for k, e in cols.items()]
        ).with_columns(pl.lit(f"{a}-{b}").alias("pair"))
        pair_rows.append(contrast_table(w.sort("decision_date"), list(cols), ["pair"]))
    pairs = pl.concat(pair_rows, how="diagonal_relaxed")
    pairs.write_csv(OUT / "pairwise_contrast.csv")

    # defensive weight terciles (pooled cutpoints over defensive stock-days)
    d = pool.filter(pl.col("group") == "defensive")
    q1, q2 = d["def_w"].quantile(1 / 3), d["def_w"].quantile(2 / 3)
    d = d.with_columns(
        pl.when(pl.col("def_w") <= q1).then(pl.lit("w_low")).when(pl.col("def_w") <= q2).then(pl.lit("w_mid")).otherwise(pl.lit("w_high")).alias("tier")
    )
    tiers = day_means(d, ["decision_date", "tier"]).join(full, on="decision_date", how="inner")
    tiers = tiers.with_columns([(pl.col(f"rw_{l}") - pl.col(f"pool_rw_{l}")).alias(f"d_rw_{l}") for l in LABELS])
    tier_tab = contrast_table(tiers.sort(["tier", "decision_date"]), [f"d_rw_{l}" for l in LABELS] + [f"rw_{l}" for l in LABELS], ["tier"])
    tier_tab.write_csv(OUT / "defensive_weight_tiers.csv")

    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(260)
    r = lambda f: f.with_columns(pl.col(pl.Float64).round(3))  # noqa: E731
    print("\n== coverage ==")
    print(coverage)
    print("\n== shared-pool contrast (group minus full pool, row-weighted within day) ==")
    print(r(shared.select("group", "period", "days", "d_rw_label_raw_bp", "d_rw_label_raw_bp_t", "d_rw_label_raw_bp_posmonth", "d_rw_label_residual_bp", "d_rw_label_residual_bp_t", "d_rw_label_z", "d_rw_label_z_t", "d_rw_label_z_posday", "d_rw_label_z_posmonth")))
    print("\n== shared-pool contrast, stock-equal weighting ==")
    print(r(shared.select("group", "period", "days", "d_se_label_raw_bp", "d_se_label_raw_bp_t", "d_se_label_residual_bp", "d_se_label_residual_bp_t", "d_se_label_z", "d_se_label_z_t")))
    print("\n== absolute equal-day levels ==")
    print(r(levels.select("group", "period", "days", "rw_label_raw_bp", "rw_label_raw_bp_t", "rw_label_residual_bp", "rw_label_residual_bp_t", "rw_label_z", "rw_label_z_t")))
    print("\n== pairwise ==")
    print(r(pairs.select("pair", "period", "days", "label_raw_bp", "label_raw_bp_t", "label_residual_bp", "label_residual_bp_t", "label_z", "label_z_t", "label_z_posmonth")))
    print("\n== defensive weight terciles (delta vs pool, and absolute) ==")
    print(r(tier_tab.select("tier", "period", "days", "d_rw_label_raw_bp", "d_rw_label_raw_bp_t", "d_rw_label_z", "d_rw_label_z_t", "rw_label_raw_bp", "rw_label_z")))


if __name__ == "__main__":
    main()
