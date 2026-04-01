# feature_transforms.py 完整更新版
import polars as pl
import numpy as np
from numba import njit
from datetime import timedelta
from tqdm.auto import tqdm

@njit(cache=True)
def get_rank_numba(ref_sorted, targets):
    n_ref = len(ref_sorted)
    if n_ref == 0:
        return np.full(len(targets), np.nan)
    insert_indices = np.searchsorted(ref_sorted, targets, side='right')
    return insert_indices / n_ref

def apply_lookback_rank(df: pl.DataFrame, time_col: str, feature_cols: list, n_days: int) -> tuple[pl.DataFrame, list]:
    print(f"\n⚙️ 開始計算 {len(feature_cols)} 個特徵的 {n_days} 日 Lookback Rank...")
    
    # 確保按時間排序並提取日期
    if df[time_col].dtype == pl.String:
        df = df.with_columns(pl.col(time_col).str.to_datetime(strict=False))
        
    df = df.sort(time_col).with_columns(pl.col(time_col).dt.date().alias("_date_internal"))
    new_cols = []
    
    for val_col in tqdm(feature_cols, desc="Rank 轉換進度"):
        # 🌟 關鍵修正：date_tuple[0] 把日期從 Tuple 中抽出來
        daily_groups = {
            date_tuple[0]: group.get_column(val_col).drop_nulls().to_numpy() 
            for date_tuple, group in df.group_by("_date_internal", maintain_order=True)
        }
        
        unique_dates = sorted(daily_groups.keys())
        results_list = []
        
        for current_date in unique_dates:
            # 現在 current_date 是真正的 date 物件了，可以正常相減！
            start_date = current_date - timedelta(days=n_days)
            ref_arrays = [
                daily_groups[d] for d in unique_dates 
                if start_date <= d < current_date and d in daily_groups
            ]
            
            target_vals = df.filter(pl.col("_date_internal") == current_date).get_column(val_col).to_numpy()
            
            if not ref_arrays:
                ranks = np.full(len(target_vals), np.nan)
            else:
                ref_vals = np.concatenate(ref_arrays)
                ref_vals.sort()
                ranks = get_rank_numba(ref_vals, target_vals)
                
            results_list.append(ranks)
            
        new_col_name = f"{val_col}_rank{n_days}d"
        df = df.with_columns(pl.Series(new_col_name, np.concatenate(results_list)))
        new_cols.append(new_col_name)

    return df.drop("_date_internal"), new_cols