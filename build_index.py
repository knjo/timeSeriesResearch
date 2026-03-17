"""
build_index.py
==============
逐日掃描 tickFeature，過濾出符合以下條件的 tick：
    (Spread.diff() != 0)  &  (FillLots_atLow < 0)

輸出：research/negFill/data_index.parquet
      欄位 = ['Date', 'QuoteCode', 'ChannelSeq']
"""

import sys
import polars as pl
from pathlib import Path

# --------------- paths ---------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # HFTResearch/

TICK_FEATURE_DIR = PROJECT_ROOT / "data" / "tickFeature"
OUTPUT_DIR       = PROJECT_ROOT / "src" / "research" / "negFill"
OUTPUT_PATH      = OUTPUT_DIR / "data_index.parquet"

# --------------- filter logic ---------------

def filter_one_day(path: Path) -> pl.DataFrame | None:
    """
    讀取單日 tickFeature，回傳符合條件的 unikey DataFrame。
    如果沒有符合條件的資料就回傳 None。
    """
    date_str = path.stem.split("_")[0]  # e.g. "20240105"

    lf = pl.scan_parquet(path)

    # 確認需要的欄位存在
    schema_cols = lf.collect_schema().names()
    for col in ["QuoteCode", "ChannelSeq", "Spread", "FillLots_atLow","SpreadPairBid"]:
        if col not in schema_cols:
            print(f"  [SKIP] {date_str}: missing column '{col}'")
            return None

    # Spread diff per QuoteCode
    lf = lf.with_columns(
        pl.col("Spread").diff().over("QuoteCode").alias("_spread_diff")
    )
    # Spread diff per QuoteCode
    lf = lf.filter((pl.col("_spread_diff") != 0) &(pl.col("BidPreMove") != 0) & (pl.col("FillLots_atLow") < 0))

    # 只保留 unikey
    lf = lf.select([
        pl.lit(date_str).alias("Date"),
        "QuoteCode",
        "ChannelSeq",
    ])

    df = lf.collect()
    return df if len(df) > 0 else None


def build_index(forcefetch: bool = False) -> pl.DataFrame:
    """
    掃描 tickFeature 日檔，增量更新 data_index。
    若已存在 data_index.parquet，只掃描比最新日期更新的檔案。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    empty_schema = {"Date": pl.Utf8, "QuoteCode": pl.Utf8, "ChannelSeq": pl.Int64}

    # 載入既有 index（如果有的話）
    existing_df = None
    latest_date = None
    if OUTPUT_PATH.exists() and not forcefetch:
        existing_df = pl.read_parquet(OUTPUT_PATH)
        if len(existing_df) > 0:
            latest_date = existing_df["Date"].max()  # e.g. "20260228"
            print(f"Existing index loaded: {len(existing_df)} unikeys, latest date = {latest_date}")

    # 取得所有 tickFeature 檔案
    all_files = sorted(TICK_FEATURE_DIR.glob("*_tickFeature.parquet"))
    if not all_files:
        print(f"No tickFeature files found in {TICK_FEATURE_DIR}")
        return existing_df if existing_df is not None else pl.DataFrame(schema=empty_schema)

    # 只掃描比 latest_date 更新的檔案
    if latest_date is not None:
        files = [f for f in all_files if f.stem.split("_")[0] > latest_date]
        if not files:
            print(f"No new tickFeature files after {latest_date}, index is up-to-date.")
            return existing_df
        print(f"Found {len(files)} new files after {latest_date} (out of {len(all_files)} total)")
    else:
        files = all_files
        print(f"Scanning {len(files)} tickFeature files ...")

    # 掃描新檔案
    parts: list[pl.DataFrame] = []
    for i, f in enumerate(files):
        date_str = f.stem.split("_")[0]
        result = filter_one_day(f)
        if result is not None:
            parts.append(result)
            print(f"  [{i+1}/{len(files)}] {date_str}: {len(result)} unikeys")
        else:
            print(f"  [{i+1}/{len(files)}] {date_str}: 0 unikeys")

    # 合併既有 + 新增
    if existing_df is not None and parts:
        index_df = pl.concat([existing_df] + parts, how="vertical")
    elif existing_df is not None:
        print("No new matching unikeys found.")
        return existing_df
    elif parts:
        index_df = pl.concat(parts, how="vertical")
    else:
        print("No matching unikeys found across all dates.")
        return pl.DataFrame(schema=empty_schema)

    index_df.write_parquet(OUTPUT_PATH, compression="zstd")
    new_count = sum(len(p) for p in parts)
    print(f"\nSaved data_index: {len(index_df)} total unikeys (+{new_count} new) -> {OUTPUT_PATH}")
    return index_df


# --------------- main ---------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build negFill data_index")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if index exists")
    args = parser.parse_args()
    build_index(forcefetch=args.force)
