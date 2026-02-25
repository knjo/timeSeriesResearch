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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

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
    for col in ["QuoteCode", "ChannelSeq", "Spread", "FillLots_atLow"]:
        if col not in schema_cols:
            print(f"  [SKIP] {date_str}: missing column '{col}'")
            return None

    # Spread diff per QuoteCode
    lf = lf.with_columns(
        pl.col("Spread").diff().over("QuoteCode").alias("_spread_diff")
    )

    # 篩選條件
    lf = lf.filter(
        (pl.col("_spread_diff") != 0) & (pl.col("FillLots_atLow") < 0)
    )

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
    掃描所有 tickFeature 日檔，建立整份 data_index。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not forcefetch:
        print(f"data_index already exists at {OUTPUT_PATH}, loading...")
        return pl.read_parquet(OUTPUT_PATH)

    files = sorted(TICK_FEATURE_DIR.glob("*_tickFeature.parquet"))
    if not files:
        print(f"No tickFeature files found in {TICK_FEATURE_DIR}")
        return pl.DataFrame(schema={"Date": pl.Utf8, "QuoteCode": pl.Utf8, "ChannelSeq": pl.Int64})

    print(f"Scanning {len(files)} tickFeature files ...")
    parts: list[pl.DataFrame] = []

    for i, f in enumerate(files):
        date_str = f.stem.split("_")[0]
        result = filter_one_day(f)
        if result is not None:
            parts.append(result)
            print(f"  [{i+1}/{len(files)}] {date_str}: {len(result)} unikeys")
        else:
            print(f"  [{i+1}/{len(files)}] {date_str}: 0 unikeys")

    if not parts:
        print("No matching unikeys found across all dates.")
        return pl.DataFrame(schema={"Date": pl.Utf8, "QuoteCode": pl.Utf8, "ChannelSeq": pl.Int64})

    index_df = pl.concat(parts, how="vertical")
    index_df.write_parquet(OUTPUT_PATH, compression="zstd")
    print(f"\nSaved data_index: {len(index_df)} total unikeys -> {OUTPUT_PATH}")
    return index_df


# --------------- main ---------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build negFill data_index")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if index exists")
    args = parser.parse_args()
    build_index(forcefetch=args.force)
