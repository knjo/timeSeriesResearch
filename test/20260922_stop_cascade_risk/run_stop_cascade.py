"""Stop cascade risk: 盤中被嘎 (停損) 檔數之後, 新空單損益與既有部位 mid→Close 損益.

輸入: signalwithRisk_df.parquet (風控後成交清單; 每列 = 1 張以 BidPrice1 賣出的空單, 名目 = BidPrice1*1000).
停損: 該檔當日首次進場後, tickData 第一筆成交價 >= RefPrice*stop_ratio 的時間 T_k;
      出場價 = RefPrice*stop_ratio + 2*TickSize (BackTest 慣例). 依 T_k 排序得當日第 k 檔.

Q1  每筆新空單標 stops_before = 進場前已被嘎檔數.
    window:     stops_before == k 的新空單 (第 k 與 k+1 檔停損之間進場)
    cumulative: stops_before >= k 的新空單 (第 k 檔停損之後全部)
    有 K 檔停損的交易日在 day panel 貢獻 k=1..K 各一筆.
Q2  每個 T_k, 仍在手上的部位 (進場 < T_k, 該檔尚未停損, 非 reentry) 以 T_k 的 as-of mid 為基準:
    bp_close  = (mid - Close) / mid * 1e4              (使用者口徑)
    bp_settle = 之後被嘎者改以停損價結算, 其餘同 bp_close  (對照)

平均 BP = 逐筆等權; 部位 BP = sum(bp*notional)/sum(notional); day_* = 先算每日再等日平均.

Usage:
    uv run python src/research/timeSeries/test/20260922_stop_cascade_risk/run_stop_cascade.py
    uv run python src/research/timeSeries/test/20260922_stop_cascade_risk/run_stop_cascade.py --stop-ratio 1.09
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_storage import load_pipeline_storage  # noqa: E402

STUDY_DIR = Path(__file__).resolve().parent
SIGNAL_PATH = PROJECT_ROOT / "src/research/timeSeries/signalwithRisk_df.parquet"
SIG_COLS = [
    "Date", "QuoteCode", "ChannelSeq", "TransTime", "BidPrice1", "RefPrice",
    "TickSize", "Close", "FutureHigh", "TakerSell_CloseBP",
]
BP = 10_000.0
WAN = 10_000.0


# ----------------------------------------------------------------------------
# per-day computation
# ----------------------------------------------------------------------------
def load_ticks(tick_dir: Path, date: str, stocks: list[str]) -> pl.DataFrame:
    return (
        pl.scan_parquet(tick_dir / f"{date}_StockTick.parquet")
        .filter(pl.col("QuoteCode").is_in(stocks) & (pl.col("TrialMatch") == 0))
        .select(["QuoteCode", "TransTime", "FillPrice", "BidPrice1", "AskPrice1"])
        .collect()
        .sort(["QuoteCode", "TransTime"])
    )


def day_stops(sig: pl.DataFrame, ticks: pl.DataFrame, ratio: float) -> pl.DataFrame:
    """每檔當日第一次停損: 首次進場後第一筆成交價 >= RefPrice*ratio."""
    first = sig.group_by("QuoteCode").agg(
        pl.col("TransTime").min().alias("first_entry"),
        pl.col("RefPrice").first(),
        pl.col("TickSize").first(),
    )
    hit = ticks.join(first, on="QuoteCode").filter(
        (pl.col("FillPrice") > 0)
        & (pl.col("FillPrice") >= pl.col("RefPrice") * ratio)
        & (pl.col("TransTime") > pl.col("first_entry"))
    )
    return (
        hit.group_by("QuoteCode")
        .agg(
            pl.col("TransTime").min().alias("stop_time"),
            pl.col("FillPrice").sort_by("TransTime").first().alias("stop_fill"),
            pl.col("RefPrice").first(),
            pl.col("TickSize").first(),
        )
        .with_columns((pl.col("RefPrice") * ratio + 2 * pl.col("TickSize")).alias("stop_price"))
        .sort("stop_time")
        .with_row_index("k", offset=1)
        .with_columns(pl.col("k").cast(pl.Int32))
    )


def q1_rows(sig: pl.DataFrame, stops: pl.DataFrame, ratio: float, fee_bp: float) -> pl.DataFrame:
    own = stops.select(
        "QuoteCode",
        pl.col("stop_time").alias("own_stop_time"),
        pl.col("stop_price").alias("own_stop_price"),
    )
    sig = sig.join(own, on="QuoteCode", how="left").sort("TransTime")
    before = stops.select(pl.col("stop_time").alias("_st"), pl.col("k").alias("stops_before"))
    sig = sig.join_asof(before, left_on="TransTime", right_on="_st", strategy="backward")
    sig = sig.drop([c for c in ("_st",) if c in sig.columns])

    stopped = pl.col("FutureHigh") >= pl.col("RefPrice") * ratio
    settle = pl.when(stopped).then(pl.col("own_stop_price")).otherwise(pl.col("Close"))
    return sig.with_columns(
        pl.col("stops_before").fill_null(0).cast(pl.Int32),
        (pl.col("own_stop_time").is_not_null() & (pl.col("own_stop_time") <= pl.col("TransTime"))).alias("is_reentry"),
        stopped.alias("stopped"),
        (pl.col("BidPrice1") * 1000).alias("notional"),
        ((pl.col("BidPrice1") - settle) / pl.col("BidPrice1") * BP).alias("pnl_bp"),
    ).with_columns(
        (pl.col("notional") * (pl.col("pnl_bp") - fee_bp) / BP).alias("pnl_twd"),
    )


def q2_rows(sig: pl.DataFrame, stops: pl.DataFrame, ticks: pl.DataFrame) -> pl.DataFrame:
    """每個 T_k 當時仍在手上的部位, 以 T_k 的 as-of mid 對收盤."""
    parts = []
    for s in stops.iter_rows(named=True):
        t_k = s["stop_time"]
        pos = sig.filter(
            (pl.col("TransTime") < t_k)
            & (pl.col("own_stop_time").is_null() | (pl.col("own_stop_time") > t_k))
            & ~pl.col("is_reentry")
        )
        if pos.is_empty():
            continue
        quote = (
            ticks.filter(pl.col("TransTime") <= t_k)
            .group_by("QuoteCode")
            .agg(pl.col("BidPrice1").last().alias("bid_k"), pl.col("AskPrice1").last().alias("ask_k"))
        )
        # 漲停/跌停一邊為 0 時用另一邊
        mid = (
            pl.when((pl.col("bid_k") > 0) & (pl.col("ask_k") > 0)).then((pl.col("bid_k") + pl.col("ask_k")) / 2)
            .when(pl.col("bid_k") > 0).then(pl.col("bid_k"))
            .when(pl.col("ask_k") > 0).then(pl.col("ask_k"))
            .otherwise(None)
        )
        later_stopped = pl.col("own_stop_time").is_not_null()
        settle = pl.when(later_stopped).then(pl.col("own_stop_price")).otherwise(pl.col("Close"))
        parts.append(
            pos.join(quote, on="QuoteCode", how="left")
            .with_columns(mid.alias("mid_k"))
            .filter(pl.col("mid_k").is_not_null())
            .with_columns(
                pl.lit(s["k"]).cast(pl.Int32).alias("k"),
                pl.lit(t_k).alias("stop_time"),
                pl.lit(s["QuoteCode"]).alias("stopped_code"),
                later_stopped.alias("later_stopped"),
                ((pl.col("mid_k") - pl.col("Close")) / pl.col("mid_k") * BP).alias("bp_close"),
                ((pl.col("mid_k") - settle) / pl.col("mid_k") * BP).alias("bp_settle"),
            )
            .select([
                "Date", "k", "stop_time", "stopped_code", "QuoteCode", "ChannelSeq", "TransTime",
                "BidPrice1", "notional", "mid_k", "Close", "later_stopped", "bp_close", "bp_settle",
            ])
        )
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts)


def build_rows(signal: pl.DataFrame, tick_dir: Path, ratio: float, fee_bp: float):
    stops_all, q1_all, q2_all = [], [], []
    for date, sig in tqdm(sorted(signal.partition_by("Date", as_dict=True).items()), desc=f"stop{ratio}"):
        date = date[0]
        path = tick_dir / f"{date}_StockTick.parquet"
        if not path.exists():
            tqdm.write(f"  [SKIP] {date}: tickData not found")
            continue
        ticks = load_ticks(tick_dir, date, sig["QuoteCode"].unique().to_list())
        stops = day_stops(sig, ticks, ratio)
        q1 = q1_rows(sig, stops, ratio, fee_bp)
        stops_all.append(stops.with_columns(pl.lit(date).alias("Date")))
        q1_all.append(q1)
        if stops.height:
            q2 = q2_rows(q1, stops, ticks)
            if q2.height:
                q2_all.append(q2)
    return pl.concat(stops_all), pl.concat(q1_all), pl.concat(q2_all)


# ----------------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------------
def _agg(frame: pl.DataFrame, by: str, bp_col: str, pnl_col: str | None = None) -> pl.DataFrame:
    day = frame.group_by(["Date", by]).agg(
        pl.len().alias("n"),
        pl.col(bp_col).mean().alias("mean_bp"),
        ((pl.col(bp_col) * pl.col("notional")).sum() / pl.col("notional").sum()).alias("pos_bp"),
        (pl.col("notional").sum() / WAN).alias("notional_wan"),
        ((pl.col(pnl_col).sum() if pnl_col else (pl.col(bp_col) * pl.col("notional")).sum() / BP) / WAN).alias("pnl_wan"),
    )
    return day.group_by(by).agg(
        pl.col("Date").n_unique().alias("n_days"),
        pl.col("n").sum().alias("n_trades"),
        ((pl.col("mean_bp") * pl.col("n")).sum() / pl.col("n").sum()).alias("mean_bp"),
        ((pl.col("pos_bp") * pl.col("notional_wan")).sum() / pl.col("notional_wan").sum()).alias("pos_bp"),
        pl.col("mean_bp").mean().alias("day_mean_bp"),
        pl.col("pos_bp").mean().alias("day_pos_bp"),
        pl.col("pnl_wan").sum().alias("pnl_wan"),
        pl.col("pnl_wan").mean().alias("day_pnl_wan"),
    ).sort(by), day.sort(["Date", by])


def q1_summaries(q1: pl.DataFrame, include_reentry: bool):
    base = q1 if include_reentry else q1.filter(~pl.col("is_reentry"))
    window, window_day = _agg(base, "stops_before", "pnl_bp", "pnl_twd")
    stopped_rate = base.group_by("stops_before").agg(pl.col("stopped").mean().alias("stopped_rate"),
                                                     pl.col("TakerSell_CloseBP").mean().alias("label_mean_bp"))
    window = window.join(stopped_rate, on="stops_before", how="left")

    kmax = int(base["stops_before"].max())
    cum_parts = []
    for k in range(0, kmax + 1):
        sub = base.filter(pl.col("stops_before") >= k).with_columns(pl.lit(k).cast(pl.Int32).alias("k_ge"))
        s, _ = _agg(sub, "k_ge", "pnl_bp", "pnl_twd")
        cum_parts.append(s)
    cumulative = pl.concat(cum_parts)

    # 對照: 全部 / 無停損日 / 有停損日
    kd = base.group_by("Date").agg(pl.col("stops_before").max().alias("K_d"))
    ref = []
    for name, cond in [("all", pl.lit(True)), ("days_no_stop", pl.col("K_d") == 0), ("days_with_stop", pl.col("K_d") > 0)]:
        sub = base.join(kd, on="Date").filter(cond).with_columns(pl.lit(name).alias("group"))
        s, _ = _agg(sub, "group", "pnl_bp", "pnl_twd")
        ref.append(s)
    return window, cumulative, pl.concat(ref), window_day


def q2_summaries(q2: pl.DataFrame):
    close_s, close_day = _agg(q2, "k", "bp_close")
    settle_s, settle_day = _agg(q2, "k", "bp_settle")
    later = q2.group_by("k").agg(pl.col("later_stopped").mean().alias("later_stopped_rate"))
    summary = (
        close_s.join(settle_s.select("k", pl.col("mean_bp").alias("mean_bp_settle"), pl.col("pos_bp").alias("pos_bp_settle"),
                                     pl.col("day_mean_bp").alias("day_mean_bp_settle"), pl.col("pnl_wan").alias("pnl_wan_settle")), on="k")
        .join(later, on="k")
        .sort("k")
    )
    day = close_day.join(
        settle_day.select("Date", "k", pl.col("mean_bp").alias("mean_bp_settle"), pl.col("pos_bp").alias("pos_bp_settle")),
        on=["Date", "k"],
    ).join(q2.group_by(["Date", "k"]).agg(pl.col("stop_time").first(), pl.col("stopped_code").first()), on=["Date", "k"]).sort(["Date", "k"])
    return summary, day


def to_md(df: pl.DataFrame, digits: int = 1) -> str:
    cols = df.columns

    def fmt(v, c):
        if isinstance(v, float):
            return f"{v:.3f}" if "rate" in c else f"{v:.{digits}f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v, c) for v, c in zip(row, cols)) + " |" for row in df.iter_rows()]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--signal", type=Path, default=SIGNAL_PATH)
    parser.add_argument("--stop-ratio", type=float, default=1.08)
    parser.add_argument("--fee-bp", type=float, default=19.3)
    parser.add_argument("--include-reentry", action="store_true", help="Q1 納入被嘎後同檔再進場的列")
    parser.add_argument("--out", type=Path, default=None, help="預設 artifacts/stop<ratio>/")
    parser.add_argument("--force", action="store_true", help="重算 per-row artifacts")
    args = parser.parse_args()

    out = args.out or STUDY_DIR / "artifacts" / f"stop{args.stop_ratio:g}"
    out.mkdir(parents=True, exist_ok=True)
    rows_paths = {n: out / f"{n}.parquet" for n in ("stops", "q1_rows", "q2_rows")}

    if args.force or not all(p.exists() for p in rows_paths.values()):
        tick_dir = load_pipeline_storage(project_root=PROJECT_ROOT).tick_dir
        signal = (
            pl.read_parquet(args.signal, columns=SIG_COLS)
            .with_columns(pl.col("TickSize").cast(pl.Float64))
            .sort(["Date", "TransTime"])
        )
        stops, q1, q2 = build_rows(signal, tick_dir, args.stop_ratio, args.fee_bp)
        stops.write_parquet(rows_paths["stops"]); q1.write_parquet(rows_paths["q1_rows"]); q2.write_parquet(rows_paths["q2_rows"])
    else:
        stops, q1, q2 = (pl.read_parquet(p) for p in rows_paths.values())

    window, cumulative, ref, q1_day = q1_summaries(q1, args.include_reentry)
    q2_sum, q2_day = q2_summaries(q2)
    kdist = stops.group_by("Date").len().rename({"len": "K_d"})
    n_days = q1["Date"].n_unique()
    kdist_tbl = (
        kdist.group_by("K_d").len().rename({"len": "n_days"}).sort("K_d")
        .vstack(pl.DataFrame({"K_d": [0], "n_days": [n_days - kdist.height]}, schema={"K_d": pl.UInt32, "n_days": pl.UInt32}))
        .sort("K_d")
    )

    for name, frame in [("q1_window_summary", window), ("q1_cumulative_summary", cumulative), ("q1_reference", ref),
                        ("q1_day_panel", q1_day), ("q2_summary", q2_sum), ("q2_day_panel", q2_day), ("stops_per_day", kdist_tbl)]:
        frame.write_csv(out / f"{name}.csv")

    reentry_n = int(q1["is_reentry"].sum())
    md = [
        f"# stop cascade @ ratio {args.stop_ratio:g}, fee {args.fee_bp}bp, reentry {'included' if args.include_reentry else 'excluded'} ({reentry_n} rows)",
        f"days={n_days}, trades={q1.height}, days with >=1 stop={kdist.height}",
        "", "## 每日停損檔數分佈", to_md(kdist_tbl),
        "", "## Q1 對照 (all / 無停損日 / 有停損日)", to_md(ref),
        "", "## Q1 window: stops_before == k 的新空單", to_md(window),
        "", "## Q1 cumulative: stops_before >= k 的新空單", to_md(cumulative),
        "", "## Q2: 第 k 檔停損當下, 既有部位 mid→Close (bp_close) / 之後被嘎者改停損價結算 (settle)", to_md(q2_sum),
    ]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    (out / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print(f"\nartifacts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
