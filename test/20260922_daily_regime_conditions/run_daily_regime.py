"""Daily regime conditions: signalwithRisk_df 的每日損益 vs D 盤前 (或開盤 30 秒內) 可知的日級條件.

日損益 (照 notebook cell 38): pnl_wan = sum(BidPrice1 * (TakerSell_CloseBP - fee) / 1e4 / 10),
position_wan = sum(BidPrice1 / 10), ret_bp = pnl / position * 1e4.

條件 (每個交易日 D 一個值, 全部在 D 決策時點已知; top-100 = preMarket(D).day_amount_rank < 100 = D-1 成交金額前 100):
  top_open_low_share       top-100 中 D 開盤 < 參考價 的比例               (D 開盤, marketData(D))
  top_open_ret_mean        top-100 的 (open - ref) / ref 平均
  top_dt_buy_count         top-100 中 brench(D) dt_major_net_buy_amount > 0 的檔數 (brench(D) = D-1 分點)
  top_dt_net_amount        top-100 的 dt_major_net_buy_amount 合計 (元)
  all_dt_net_amount        全部有 brench 的股票合計
  limit_touch_rate_lag1    D-1 全市場普通股 high >= limit_up_price 的比例
  limit_close_over_touch_lag1  D-1 收盤 >= 漲停價 檔數 / 摸漲停檔數
  top_limit_touch_rate_lag1    D-1 同上但 universe = 當時 top-100
  limit_*_ma5              上述三者 D-5..D-1 五日均
  top_large_cap_share      top-100 中 MarketCap >= 1e11 的比例 (top_mega_cap_share: >= 5e11)
  top_cap100_share         top-100 中同時是市值前 100 (普通股內排名) 的比例

分析
  A. trailing qcut10: 對每個 D, 用嚴格在 D 之前 N 個交易日 (>= min_obs) 的條件值算十分位, 把 D 歸到 0..9。
  B. in-window qcut10: 損益窗內自身十分位 (描述用, 非因果)。
  C. MA cross: 條件序列 (含 D 已知值) 的短均 vs 長均, golden = 短 > 長, death = 短 < 長; fresh = 交叉後 <= 5 天。

Usage:
    uv run python src/research/timeSeries/test/20260922_daily_regime_conditions/run_daily_regime.py
    uv run python .../run_daily_regime.py --window-tdays 120 --ma-pairs 5:20,5:60
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_storage import load_pipeline_storage  # noqa: E402

STUDY_DIR = Path(__file__).resolve().parent
SIGNAL_PATH = PROJECT_ROOT / "src/research/timeSeries/signalwithRisk_df.parquet"
STOPS_PATH = STUDY_DIR.parent / "20260922_stop_cascade_risk/artifacts/stop1.08/stops.parquet"

BASE_CONDITIONS = [
    "top_open_low_share", "top_open_ret_mean",
    "top_dt_buy_count", "top_dt_net_amount", "all_dt_net_amount",
    "limit_touch_rate_lag1", "limit_close_over_touch_lag1", "top_limit_touch_rate_lag1",
    "top_large_cap_share", "top_mega_cap_share", "top_cap100_share",
]
QCUT_CONDITIONS = BASE_CONDITIONS + ["limit_touch_rate_ma5", "limit_close_over_touch_ma5", "top_limit_touch_rate_ma5"]
LARGE_CAP, MEGA_CAP = 1e11, 5e11
FRESH_DAYS = 5


# ----------------------------------------------------------------------------
# daily pnl
# ----------------------------------------------------------------------------
def daily_pnl(signal_path: Path, fee_bp: float) -> pl.DataFrame:
    df = pl.read_parquet(signal_path, columns=["Date", "QuoteCode", "BidPrice1", "TakerSell_CloseBP"])
    out = df.group_by("Date").agg(
        pl.len().alias("n_trades"),
        pl.col("QuoteCode").n_unique().alias("n_stocks"),
        (pl.col("BidPrice1") / 10).sum().alias("position_wan"),
        (pl.col("BidPrice1") * (pl.col("TakerSell_CloseBP") - fee_bp) / 1e4 / 10).sum().alias("pnl_wan"),
    ).with_columns((pl.col("pnl_wan") / pl.col("position_wan") * 1e4).alias("ret_bp")).sort("Date")
    if STOPS_PATH.exists():
        k = pl.read_parquet(STOPS_PATH).group_by("Date").len().rename({"len": "n_stops"})
        out = out.join(k, on="Date", how="left").with_columns(pl.col("n_stops").fill_null(0))
    return out


# ----------------------------------------------------------------------------
# conditions
# ----------------------------------------------------------------------------
def _daily_ohlc(storage, date: str) -> pl.DataFrame | None:
    """marketData(D) 的 open/high/close/limit_up; 尚未回填時改用 tickData 推."""
    md_path = storage.market_dir / f"{date}_marketData.parquet"
    if not md_path.exists():
        return None
    md = pl.read_parquet(md_path)
    base = md.select(
        pl.col("quote_code").alias("QuoteCode"), "ins_type",
        pl.col("opening_ref_price").alias("ref"), pl.col("limit_up_price").alias("limit_up"),
    )
    if "open_price" in md.columns:
        return base.join(md.select(pl.col("quote_code").alias("QuoteCode"), pl.col("open_price").alias("open"),
                                   pl.col("high_price").alias("high"), pl.col("close_price").alias("close")), on="QuoteCode")
    tick_path = storage.tick_dir / f"{date}_StockTick.parquet"
    if not tick_path.exists():
        return None
    tick = (
        pl.scan_parquet(tick_path).filter(pl.col("marketOpen"))
        .group_by("QuoteCode").agg(pl.col("Open").first().alias("open"), pl.col("FutureHigh").first().alias("high"),
                                   pl.col("Close").first().alias("close"))
        .collect()
    )
    return base.join(tick, on="QuoteCode", how="left")


def build_conditions(storage, dates: list[str]) -> pl.DataFrame:
    rows = []
    for date in tqdm(dates, desc="conditions"):
        pm_path = storage.pre_market_dir / f"{date}_preMarketData.parquet"
        ohlc = _daily_ohlc(storage, date)
        if not pm_path.exists() or ohlc is None:
            continue
        pm = pl.read_parquet(pm_path, columns=["QuoteCode", "day_amount_rank", "MarketCap", "ins_type"])
        cap100 = set(pm.filter(pl.col("ins_type") == "stock").sort("MarketCap", descending=True, nulls_last=True)
                     .head(100)["QuoteCode"].to_list())
        bf_path = storage.brench_feature_dir / f"{date}_brenchFeature.parquet"
        bf = (pl.read_parquet(bf_path, columns=["QuoteCode", "dt_major_net_buy_amount"])
              if bf_path.exists() else pl.DataFrame(schema={"QuoteCode": pl.String, "dt_major_net_buy_amount": pl.Float64}))
        stocks = ohlc.filter((pl.col("ins_type") == "stock") & pl.col("ref").is_not_null() & (pl.col("ref") > 0) & pl.col("high").is_not_null())
        top = pm.filter(pl.col("day_amount_rank") < 100).join(ohlc.drop("ins_type"), on="QuoteCode", how="left").join(bf, on="QuoteCode", how="left")
        top_px = top.filter(pl.col("open").is_not_null() & (pl.col("ref") > 0))
        touch = pl.col("high") >= pl.col("limit_up")
        closed = pl.col("close") >= pl.col("limit_up")
        n_touch = stocks.select(touch.sum()).item() if stocks.height else 0
        rows.append({
            "Date": date,
            "n_top": top.height, "n_top_px": top_px.height, "n_stocks": stocks.height,
            "top_open_low_share": (top_px["open"] < top_px["ref"]).mean() if top_px.height else None,
            "top_open_ret_mean": ((top_px["open"] - top_px["ref"]) / top_px["ref"]).mean() if top_px.height else None,
            "top_dt_buy_count": int((top["dt_major_net_buy_amount"] > 0).sum()) if bf.height else None,
            "top_dt_net_amount": float(top["dt_major_net_buy_amount"].sum()) if bf.height else None,
            "all_dt_net_amount": float(bf["dt_major_net_buy_amount"].sum()) if bf.height else None,
            "limit_touch_rate": stocks.select(touch.mean()).item() if stocks.height else None,
            "limit_close_over_touch": (stocks.select(closed.sum()).item() / n_touch) if n_touch else None,
            "top_limit_touch_rate": top_px.select(touch.mean()).item() if top_px.height else None,
            "top_large_cap_share": (top["MarketCap"] >= LARGE_CAP).mean() if top.height else None,
            "top_mega_cap_share": (top["MarketCap"] >= MEGA_CAP).mean() if top.height else None,
            "top_cap100_share": top["QuoteCode"].is_in(list(cap100)).mean() if top.height else None,
        })
    cond = pl.DataFrame(rows).sort("Date")
    lag = ["limit_touch_rate", "limit_close_over_touch", "top_limit_touch_rate"]
    return cond.with_columns(
        [pl.col(c).shift(1).alias(f"{c}_lag1") for c in lag]
        + [pl.col(c).shift(1).rolling_mean(5, min_samples=3).alias(f"{c}_ma5") for c in lag]
    )


# ----------------------------------------------------------------------------
# A/B deciles
# ----------------------------------------------------------------------------
def trailing_deciles(cond: pl.DataFrame, target_dates: list[str], columns: list[str], window_tdays: int, min_obs: int) -> pl.DataFrame:
    all_dates = cond["Date"].to_list()
    pos = {d: i for i, d in enumerate(all_dates)}
    out = {"Date": target_dates}
    for c in columns:
        vals = cond[c].to_numpy().astype(float)
        deciles = []
        for d in target_dates:
            i = pos.get(d)
            if i is None or np.isnan(vals[i]):
                deciles.append(None)
                continue
            w = vals[max(0, i - window_tdays):i]
            w = w[~np.isnan(w)]
            if len(w) < min_obs:
                deciles.append(None)
                continue
            edges = np.quantile(w, np.arange(0.1, 1.0, 0.1))
            deciles.append(int(np.searchsorted(edges, vals[i], side="right")))
        out[f"{c}_q"] = deciles
    return pl.DataFrame(out).with_columns([pl.col(f"{c}_q").cast(pl.Int32) for c in columns])


def in_window_deciles(panel: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    return panel.select("Date", *[
        (pl.col(c).rank("average") / pl.col(c).count() * 10 - 1e-9).floor().clip(0, 9).cast(pl.Int32).alias(f"{c}_qw")
        for c in columns
    ])


# ----------------------------------------------------------------------------
# C. MA cross
# ----------------------------------------------------------------------------
def ma_cross_states(cond: pl.DataFrame, columns: list[str], pairs: list[tuple[int, int]]) -> pl.DataFrame:
    out = cond.select("Date")
    for c in columns:
        for s, l in pairs:
            tag = f"{c}_x{s}_{l}"
            frame = cond.select("Date", pl.col(c).rolling_mean(s, min_samples=s).alias("ms"),
                                pl.col(c).rolling_mean(l, min_samples=l).alias("ml"))
            missing = (frame["ms"].is_null() | frame["ml"].is_null()).to_numpy()
            diff = np.nan_to_num(frame["ms"].to_numpy().astype(float)) - np.nan_to_num(frame["ml"].to_numpy().astype(float))
            state = np.where(missing, 0, np.sign(diff))  # 1 golden, -1 death, 0 unknown/equal
            # days since last state change (only counting non-zero states)
            since = np.full(len(state), -1, dtype=np.int32)
            run, prev = 0, 0
            for i, st in enumerate(state):
                if st == 0:
                    run, prev = 0, 0
                    continue
                run = run + 1 if st == prev else 1
                prev = st
                since[i] = run
            label = ["golden" if st > 0 else "death" if st < 0 else None for st in state]
            out = out.with_columns(
                pl.Series(f"{tag}_state", label, dtype=pl.String),
                pl.Series(f"{tag}_since", since).cast(pl.Int32),
            )
    return out


def _stats(g: pl.DataFrame) -> dict:
    r = g["ret_bp"]
    return {
        "n_days": g.height,
        "day_ret_bp": float(r.mean()) if g.height else None,
        "t_stat": float(r.mean() / (r.std() / np.sqrt(g.height))) if g.height > 2 and r.std() else None,
        "pos_ret_bp": float(g["pnl_wan"].sum() / g["position_wan"].sum() * 1e4) if g.height else None,
        "win_rate": float((g["pnl_wan"] > 0).mean()) if g.height else None,
        "pnl_wan": float(g["pnl_wan"].sum()),
        "worst_day_wan": float(g["pnl_wan"].min()) if g.height else None,
    }


def welch_t(a: pl.Series, b: pl.Series) -> float | None:
    if a.len() < 3 or b.len() < 3:
        return None
    se = np.sqrt(a.var() / a.len() + b.var() / b.len())
    return float((a.mean() - b.mean()) / se) if se else None


def episode_table(p: pl.DataFrame, state_col: str) -> pl.DataFrame:
    """連續同狀態的區段 (episode) 為單位: 有效樣本數比日數小得多."""
    d = p.sort("Date").with_columns((pl.col(state_col) != pl.col(state_col).shift(1)).fill_null(True).cum_sum().alias("ep"))
    return d.group_by("ep").agg(
        pl.col("Date").min().alias("start"), pl.col("Date").max().alias("end"), pl.col(state_col).first().alias("state"),
        pl.len().alias("n_days"), pl.col("ret_bp").mean().alias("ret_bp"), pl.col("pnl_wan").sum().alias("pnl_wan"),
    ).sort("ep")


def ma_cross_tables(panel: pl.DataFrame, columns: list[str], pairs: list[tuple[int, int]]):
    overview, detail, episodes = [], [], []
    for c in columns:
        for s, l in pairs:
            tag = f"{c}_x{s}_{l}"
            p = panel.filter(pl.col(f"{tag}_state").is_not_null()).with_columns(
                pl.when(pl.col(f"{tag}_since") <= FRESH_DAYS).then(pl.lit("fresh")).otherwise(pl.lit("established")).alias("age"))
            g, d = p.filter(pl.col(f"{tag}_state") == "golden"), p.filter(pl.col(f"{tag}_state") == "death")
            ep = episode_table(p, f"{tag}_state")
            ep_g, ep_d = ep.filter((pl.col("state") == "golden") & (pl.col("n_days") >= 3)), ep.filter((pl.col("state") == "death") & (pl.col("n_days") >= 3))
            overview.append({
                "condition": c, "pair": f"{s}/{l}",
                "golden_n": g.height, "golden_ret_bp": _stats(g)["day_ret_bp"], "golden_pnl_wan": _stats(g)["pnl_wan"],
                "death_n": d.height, "death_ret_bp": _stats(d)["day_ret_bp"], "death_pnl_wan": _stats(d)["pnl_wan"],
                "golden_minus_death_bp": (_stats(g)["day_ret_bp"] or 0) - (_stats(d)["day_ret_bp"] or 0) if g.height and d.height else None,
                "welch_t": welch_t(g["ret_bp"], d["ret_bp"]),
                "ep_golden_n": ep_g.height, "ep_golden_ret_bp": float(ep_g["ret_bp"].mean()) if ep_g.height else None,
                "ep_death_n": ep_d.height, "ep_death_ret_bp": float(ep_d["ret_bp"].mean()) if ep_d.height else None,
                "ep_death_positive_share": float((ep_d["ret_bp"] > 0).mean()) if ep_d.height else None,
            })
            rows = []
            for st in ("golden", "death"):
                for age in ("fresh", "established"):
                    sub = p.filter((pl.col(f"{tag}_state") == st) & (pl.col("age") == age))
                    rows.append({"state": st, "age": age, **_stats(sub)})
            detail.append((c, f"{s}/{l}", pl.DataFrame(rows)))
            episodes.append((c, f"{s}/{l}", ep))
    return pl.DataFrame(overview), detail, episodes


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------
def decile_table(panel: pl.DataFrame, c: str, suffix: str = "_q") -> pl.DataFrame:
    q = f"{c}{suffix}"
    return (
        panel.filter(pl.col(q).is_not_null())
        .group_by(q).agg(
            pl.len().alias("n_days"),
            pl.col(c).mean().alias("cond_mean"),
            pl.col("pnl_wan").mean().alias("mean_pnl_wan"),
            pl.col("pnl_wan").sum().alias("sum_pnl_wan"),
            pl.col("ret_bp").mean().alias("day_ret_bp"),
            (pl.col("ret_bp").mean() / (pl.col("ret_bp").std() / pl.len().sqrt())).alias("t_stat"),
            (pl.col("pnl_wan").sum() / pl.col("position_wan").sum() * 1e4).alias("pos_ret_bp"),
            (pl.col("pnl_wan") > 0).mean().alias("win_rate"),
            pl.col("pnl_wan").min().alias("worst_day_wan"),
            pl.col("position_wan").mean().alias("mean_position_wan"),
        )
        .rename({q: "decile"}).sort("decile")
    )


def drift_table(cond: pl.DataFrame, pnl_dates: list[str], columns: list[str], window_tdays: int) -> pl.DataFrame:
    start, end = min(pnl_dates), max(pnl_dates)
    all_dates = cond["Date"].to_list()
    i0 = all_dates.index(start) if start in all_dates else 0
    prior = cond.slice(max(0, i0 - window_tdays), min(i0, window_tdays))
    inwin = cond.filter((pl.col("Date") >= start) & (pl.col("Date") <= end))
    rows = []
    for c in columns:
        p, w = prior[c].drop_nulls(), inwin[c].drop_nulls()
        rows.append({"condition": c, "prior_n": p.len(), "prior_median": p.median(), "prior_p90": p.quantile(0.9),
                     "window_n": w.len(), "window_median": w.median(),
                     "window_share_above_prior_p90": float((w > p.quantile(0.9)).mean()) if p.len() else None})
    return pl.DataFrame(rows)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 5:
        return float("nan")
    rx = pl.Series(x[m]).rank().to_numpy(); ry = pl.Series(y[m]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def to_md(df: pl.DataFrame) -> str:
    cols = df.columns

    def fmt(v, c):
        if v is None:
            return ""
        if isinstance(v, float):
            if "rate" in c or "share" in c:
                return f"{v:.3f}"
            if abs(v) >= 1e6:
                return f"{v:.3g}"
            return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v, c) for v, c in zip(row, cols)) + " |" for row in df.iter_rows()]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--signal", type=Path, default=SIGNAL_PATH)
    parser.add_argument("--fee-bp", type=float, default=19.3)
    parser.add_argument("--window-tdays", type=int, default=120, help="trailing qcut 視窗 (交易日)")
    parser.add_argument("--min-obs", type=int, default=60)
    parser.add_argument("--history-days", type=int, default=400, help="條件序列往前多少日曆天開始算")
    parser.add_argument("--ma-pairs", default="5:20,5:60")
    parser.add_argument("--out", type=Path, default=STUDY_DIR / "artifacts")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = [tuple(int(x) for x in p.split(":")) for p in args.ma_pairs.split(",")]

    storage = load_pipeline_storage(project_root=PROJECT_ROOT)
    pnl = daily_pnl(args.signal, args.fee_bp)
    pnl.write_csv(args.out / "daily_pnl.csv")

    cond_path = args.out / "conditions.parquet"
    if cond_path.exists() and not args.force and "top_cap100_share" in pl.read_parquet_schema(cond_path):
        cond = pl.read_parquet(cond_path)
    else:
        first = datetime.strptime(pnl["Date"].min(), "%Y%m%d") - timedelta(days=args.history_days)
        dates = sorted(p.name[:8] for p in storage.market_dir.glob("*_marketData.parquet") if p.name[:8] >= first.strftime("%Y%m%d"))
        cond = build_conditions(storage, dates)
        cond.write_parquet(cond_path)
    cond.write_csv(args.out / "conditions.csv")

    targets = pnl["Date"].to_list()
    dec = trailing_deciles(cond, targets, QCUT_CONDITIONS, args.window_tdays, args.min_obs)
    cross = ma_cross_states(cond, BASE_CONDITIONS, pairs)
    panel = (pnl.join(cond.select(["Date", *QCUT_CONDITIONS]), on="Date", how="left")
             .join(dec, on="Date", how="left").join(cross, on="Date", how="left"))
    panel = panel.join(in_window_deciles(panel, QCUT_CONDITIONS), on="Date", how="left")
    panel.write_csv(args.out / "decile_panel.csv")
    drift = drift_table(cond, targets, QCUT_CONDITIONS, args.window_tdays)
    drift.write_csv(args.out / "condition_drift.csv")

    md = [f"# daily regime conditions (fee {args.fee_bp}bp, trailing {args.window_tdays} tdays qcut10, min_obs {args.min_obs}, MA pairs {args.ma_pairs})",
          f"days={pnl.height} ({pnl['Date'].min()}..{pnl['Date'].max()}), total pnl={pnl['pnl_wan'].sum():.0f} 萬, "
          f"day mean={pnl['pnl_wan'].mean():.1f} 萬, day ret={pnl['ret_bp'].mean():.1f} bp (std {pnl['ret_bp'].std():.0f}), "
          f"win rate={(pnl['pnl_wan'] > 0).mean():.3f}"]

    # A/B overview
    y = panel["ret_bp"].to_numpy().astype(float)
    overview, detail = [], []
    for c in QCUT_CONDITIONS:
        tbl, tblw = decile_table(panel, c), decile_table(panel, c, "_qw")
        tbl.write_csv(args.out / f"decile_{c}.csv"); tblw.write_csv(args.out / f"decile_inwindow_{c}.csv")
        rho = spearman(panel[c].to_numpy().astype(float), y)
        big = tbl.filter(pl.col("n_days") >= 10)
        worst = big.sort("day_ret_bp").head(1)
        skew = int(tbl.filter(pl.col("decile") >= 8)["n_days"].sum())
        overview.append({
            "condition": c, "spearman_ret": rho, "trail_days_in_q8-9": skew,
            "trail_worst_decile(n>=10)": int(worst["decile"][0]) if worst.height else None,
            "trail_worst_ret_bp": float(worst["day_ret_bp"][0]) if worst.height else None,
            "trail_worst_t": float(worst["t_stat"][0]) if worst.height else None,
            "trail_worst_n": int(worst["n_days"][0]) if worst.height else None,
        })
        detail += ["", f"## {c}  (spearman vs day ret_bp = {rho:.3f})",
                   f"### trailing-{args.window_tdays}d decile", to_md(tbl), "### in-window decile (descriptive)", to_md(tblw)]
    ov = pl.DataFrame(overview); ov.write_csv(args.out / "overview.csv")

    # C overview
    xov, xdetail, xepisodes = ma_cross_tables(panel, BASE_CONDITIONS, pairs)
    xov.write_csv(args.out / "ma_cross_overview.csv")
    xmd = []
    for (c, pair, tbl), (_, _, ep) in zip(xdetail, xepisodes):
        tbl.write_csv(args.out / f"ma_cross_{c}_{pair.replace('/', '_')}.csv")
        ep.write_csv(args.out / f"ma_cross_episodes_{c}_{pair.replace('/', '_')}.csv")
        xmd += ["", f"### {c} {pair}", to_md(tbl)]

    # monthly
    pm = panel.with_columns(pl.col("Date").str.slice(0, 6).alias("ym"))
    monthly = pm.group_by("ym").agg(pl.len().alias("n_days"), pl.col("ret_bp").mean().alias("day_ret_bp"),
                                    pl.col("pnl_wan").sum().alias("pnl_wan")).sort("ym")
    for c in QCUT_CONDITIONS:
        monthly = monthly.join(pm.group_by("ym").agg((pl.col(f"{c}_qw") <= 1).mean().alias(f"{c}_lo_share")), on="ym", how="left")
    monthly.write_csv(args.out / "monthly_lo_decile_share.csv")

    md += ["", "## A. trailing qcut overview", to_md(ov),
           "", f"## condition drift (prior {args.window_tdays} tdays vs pnl window)", to_md(drift),
           "", "## C. MA cross overview (golden = short MA > long MA)", to_md(xov.sort("welch_t", nulls_last=True)),
           "", "## C. MA cross detail (fresh = within 5 days of cross)", *xmd,
           "", "## monthly pnl", to_md(monthly.select(["ym", "n_days", "day_ret_bp", "pnl_wan"])),
           *detail]
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:md.index("## C. MA cross detail (fresh = within 5 days of cross)")]))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
