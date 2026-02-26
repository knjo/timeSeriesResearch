"""
dataLoader.py
=============
讀取 data_index.parquet 中的 unikey，
逐日合併 tickData + tickFeature + preMarket + tickBar，
篩選 nextday_allow_day_trade_mark == 'X'，
寫出到 src/research/negFill/data/YYYYMMDD.parquet。

Usage (批次建立):
    python src/research/negFill/dataLoader.py              # 全部日期
    python src/research/negFill/dataLoader.py --force      # 強制重建

Usage (研究時讀取):
    from src.research.negFill.dataLoader import read_negfill
    df = read_negfill("20240105")
    df = read_negfill(dates=["20240105", "20240108"])
"""

import sys
import polars as pl
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR           = PROJECT_ROOT / "data"
TICK_DATA_DIR      = DATA_DIR / "tickData"
TICK_FEATURE_DIR   = DATA_DIR / "tickFeature"
PRE_MARKET_DIR     = DATA_DIR / "preMarket"
TICK_BAR_DIR       = DATA_DIR / "tickBar"

INDEX_PATH  = PROJECT_ROOT / "src" / "research" / "negFill" / "data_index.parquet"
OUTPUT_DIR  = PROJECT_ROOT / "src" / "research" / "negFill" / "data"

UNIKEY_COLS = ["QuoteCode", "ChannelSeq"]


# ================================================================
#  Build: 逐日 merge + sink
# ================================================================

def _read_index(date: str | None = None) -> pl.DataFrame:
    """讀取 data_index，可選只保留特定 Date。"""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"data_index not found at {INDEX_PATH}. "
            "Run build_index.py first."
        )
    idx = pl.read_parquet(INDEX_PATH)
    if date is not None:
        idx = idx.filter(pl.col("Date") == date)
    return idx


def _merge_one_day(date: str, keys: pl.DataFrame) -> pl.DataFrame | None:
    """
    合併單日四份資料，回傳篩選後的 DataFrame。
    如果必要檔案不存在則回傳 None。
    """
    # --- tickFeature ---
    tf_path = TICK_FEATURE_DIR / f"{date}_tickFeature.parquet"
    if not tf_path.exists():
        print(f"  [SKIP] tickFeature not found: {tf_path.name}")
        return None
    tick_feature = (
        pl.scan_parquet(tf_path)
          .join(keys.lazy(), on=UNIKEY_COLS, how="semi")
          .collect()
    )

    # --- tickData ---
    td_path = TICK_DATA_DIR / f"{date}_StockTick.parquet"
    if not td_path.exists():
        print(f"  [SKIP] tickData not found: {td_path.name}")
        return None
    tick_data = (
        pl.scan_parquet(td_path)
          .join(keys.lazy(), on=UNIKEY_COLS, how="semi")
          .collect()
    )

    # 合併 tickData + tickFeature（same tick-level granularity）
    merged = tick_data.join(tick_feature, on=UNIKEY_COLS, how="left")

    # --- preMarket ---
    pm_path = PRE_MARKET_DIR / f"{date}_preMarketData.parquet"
    if pm_path.exists():
        pre_market = pl.read_parquet(pm_path)
        merged = merged.join(pre_market, on="QuoteCode", how="left", suffix="_pm")
    else:
        print(f"  [WARN] preMarket not found: {pm_path.name}, skipping")

    # --- tickBar (join_asof: 找最近的 5min bar) ---
    tb_path = TICK_BAR_DIR / f"{date}_tickBar.parquet"
    if tb_path.exists():
        tick_bar = pl.read_parquet(tb_path)

        # 只保留 QuoteCode + TransTime + Rank 欄位
        rank_cols = [c for c in tick_bar.columns if c.endswith("Rank")] + [c for c in tick_bar.columns if c.endswith("Count")]
        tb_select = ["QuoteCode", "TransTime", "TimeSlot"] + rank_cols
        tb_select = [c for c in tb_select if c in tick_bar.columns]
        tick_bar = tick_bar.select(tb_select)

        # 確保兩邊 TransTime 精度一致 (cast to datetime[us])
        if "TransTime" in merged.columns and "TransTime" in tick_bar.columns:
            merged = merged.sort(["QuoteCode", "TransTime"])
            tick_bar = tick_bar.sort(["QuoteCode", "TransTime"])

            # Rename tickBar TransTime to avoid collision
            tick_bar = tick_bar.rename({"TransTime": "TransTime_tb"})

            merged = merged.join_asof(
                tick_bar,
                left_on="TransTime",
                right_on="TransTime_tb",
                by="QuoteCode",
                strategy="backward",
            )
        else:
            print(f"  [WARN] {date}: TransTime column missing, skipping tickBar join")
    else:
        print(f"  [WARN] tickBar not found: {tb_path.name}, skipping")

    # 加上 Date 欄位
    merged = merged.with_columns(pl.lit(date).alias("Date"))

    # 篩選: 只保留 nextday_allow_day_trade_mark == "X"
    if "nextday_allow_day_trade_mark" in merged.columns:
        merged = merged.filter(pl.col("nextday_allow_day_trade_mark") == "X")
    elif "nextday_allow_day_trade_mark_pm" in merged.columns:
        # 可能被 suffix 改名
        merged = merged.filter(pl.col("nextday_allow_day_trade_mark_pm") == "X")
    else:
        print(f"  [WARN] {date}: nextday_allow_day_trade_mark column not found, no filter applied")

    return merged


def build_merged_data(forcefetch: bool = False):
    """
    逐日讀取 index → merge 四份 → 篩選 → 寫出 per-day parquet。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = _read_index()
    all_dates = idx["Date"].unique().sort().to_list()

    print(f"Total dates in index: {len(all_dates)}")
    print(f"Output directory: {OUTPUT_DIR}")

    saved = 0
    skipped = 0

    for i, date in enumerate(all_dates):
        output_path = OUTPUT_DIR / f"{date}.parquet"

        if output_path.exists() and not forcefetch:
            skipped += 1
            continue

        day_idx = idx.filter(pl.col("Date") == date)
        keys = day_idx.select(UNIKEY_COLS)

        try:
            result = _merge_one_day(date, keys)
            if result is not None and len(result) > 0:
                result.write_parquet(output_path, compression="zstd")
                print(f"  [{i+1}/{len(all_dates)}] {date}: {len(result)} rows saved")
                saved += 1
            else:
                print(f"  [{i+1}/{len(all_dates)}] {date}: 0 rows after filter")
        except Exception as e:
            print(f"  [{i+1}/{len(all_dates)}] {date}: ERROR - {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Saved: {saved}, Skipped: {skipped}")


# ================================================================
#  Read: 研究時讀取已建好的 per-day parquet
# ================================================================

def read_negfill(
    date: str | None = None,
    dates: list[str] | None = None,
) -> pl.DataFrame:
    """
    讀取已建好的 negFill merged parquet。

    Args:
        date:  單日 (e.g. "20240105")
        dates: 指定多日 list
        都不傳 = 讀取 data/ 底下所有日期
    """
    if date is not None:
        target_files = [OUTPUT_DIR / f"{date}.parquet"]
    elif dates is not None:
        target_files = [OUTPUT_DIR / f"{d}.parquet" for d in dates]
    else:
        target_files = sorted(OUTPUT_DIR.glob("*.parquet"))

    parts: list[pl.DataFrame] = []
    for f in target_files:
        if f.exists():
            parts.append(pl.read_parquet(f))
        else:
            print(f"  [WARN] not found: {f.name}")

    if not parts:
        print("No data loaded.")
        return pl.DataFrame()

    return pl.concat(parts, how="diagonal")


# ================================================================
#  CLI
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build negFill merged data (per-day)")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if output exists")
    args = parser.parse_args()
    build_merged_data(forcefetch=args.force)
