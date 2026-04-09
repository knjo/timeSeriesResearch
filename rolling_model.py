import os
# 強制設定 OpenBLAS / MKL 等底層函式庫預設只使用 1 個 CPU Core
# 避免 32 Process x 32 Threds = 1024 執行緒互相衝突 (Oversubscription)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import gc
import warnings
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed

# 隱藏 Scipy/Sklearn 底層矩陣 condition number 警告
warnings.filterwarnings("ignore", message=".*ill-conditioned.*")


def _numpy_spearman_ic(x, y):
    """
    純 NumPy 實作 Spearman Rank Correlation，避免 scipy.stats.spearmanr
    在多進程環境下造成的 OOM 問題。
    此版本只用 argsort + arange，記憶體占用極低。
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid]
    y_valid = y[valid]
    n = len(x_valid)
    if n < 3:
        return np.nan
    rank_x = np.empty(n, dtype=np.float64)
    rank_y = np.empty(n, dtype=np.float64)
    rank_x[np.argsort(x_valid)] = np.arange(n, dtype=np.float64)
    rank_y[np.argsort(y_valid)] = np.arange(n, dtype=np.float64)
    
    rx = rank_x - rank_x.mean()
    ry = rank_y - rank_y.mean()
    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    if denom == 0:
        return np.nan
    return np.sum(rx * ry) / denom


def _stratified_sample_indices(quotecodes, sample_ratio=0.2, min_per_stock=3, rng_seed=42):
    """
    對商品 (QuoteCode ID) 進行分層取樣 (Stratified Sampling)。
    """
    n = len(quotecodes)
    if n <= 100:
        return np.arange(n)

    rng = np.random.RandomState(rng_seed)
    unique_codes = np.unique(quotecodes)

    selected = []
    for code in unique_codes:
        code_idx = np.where(quotecodes == code)[0]
        n_code = len(code_idx)

        n_take = max(min_per_stock, int(np.ceil(n_code * sample_ratio)))
        n_take = min(n_take, n_code)

        if n_take >= n_code:
            selected.append(code_idx)
        else:
            chosen = rng.choice(code_idx, size=n_take, replace=False)
            selected.append(chosen)

    indices = np.sort(np.concatenate(selected))
    return indices


def _screen_features_for_one_day(i, unique_dates, date_bounds, X_arr, Y_arr, quotes_arr, current_X, n_training_days):
    """
    [Phase 1] 單日的 Feature Screening worker。
    """
    if i == 0:
        return None

    start_idx = max(0, i - n_training_days)
    train_start_date = unique_dates[start_idx]
    train_end_date = unique_dates[i - 1]

    train_row_start = date_bounds[train_start_date][0]
    train_row_end = date_bounds[train_end_date][1]

    if train_row_end <= train_row_start:
        return None

    mid_idx = start_idx + (i - 1 - start_idx) // 2
    split_date = unique_dates[mid_idx]
    split_row_start = date_bounds[split_date][0]

    split_row_start = max(train_row_start, min(split_row_start, train_row_end))
    local_split_idx = split_row_start - train_row_start

    if local_split_idx <= 0 or local_split_idx >= (train_row_end - train_row_start):
        return None

    X_train_chunk = X_arr[train_row_start:train_row_end]
    Y_train_chunk = Y_arr[train_row_start:train_row_end]
    quotes_train_chunk = quotes_arr[train_row_start:train_row_end]

    # ⚠️ 關鍵修改：加上 .copy() 準備進行 Clip
    X_IS_full = X_train_chunk[:local_split_idx].copy()
    Y_IS_full = Y_train_chunk[:local_split_idx]
    quotes_IS_full = quotes_train_chunk[:local_split_idx]

    # ⚠️ 關鍵修改：加上 .copy()
    X_OOS_full = X_train_chunk[local_split_idx:].copy()
    Y_OOS_full = Y_train_chunk[local_split_idx:]
    quotes_OOS_full = quotes_train_chunk[local_split_idx:]

    # ====== 🌟 新增：根據前半段 (IS_full) 計算並 Clip ======
    if len(X_IS_full) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # 忽略 NaN 計算 1% 和 99% 分位數
            lower_bounds = np.nanpercentile(X_IS_full, 1, axis=0)
            upper_bounds = np.nanpercentile(X_IS_full, 99, axis=0)
            
            # Clip IS 和 OOS 資料
            np.clip(X_IS_full, lower_bounds, upper_bounds, out=X_IS_full)
            np.clip(X_OOS_full, lower_bounds, upper_bounds, out=X_OOS_full)

    sample_ratio = 0.2
    is_sample_idx = _stratified_sample_indices(quotes_IS_full, sample_ratio=sample_ratio, min_per_stock=3, rng_seed=i)
    oos_sample_idx = _stratified_sample_indices(quotes_OOS_full, sample_ratio=sample_ratio, min_per_stock=3, rng_seed=i + 1000)

    X_IS = X_IS_full[is_sample_idx]
    Y_IS = Y_IS_full[is_sample_idx]
    X_OOS = X_OOS_full[oos_sample_idx]
    Y_OOS = Y_OOS_full[oos_sample_idx]

    selected_features_idx = []
    n_quantiles = 5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        for f_idx in range(len(current_X)):
            feat_is = X_IS[:, f_idx]
            feat_oos = X_OOS[:, f_idx]

            ic_is = _numpy_spearman_ic(feat_is, Y_IS)
            if np.isnan(ic_is):
                continue

            try:
                q_bins = np.nanquantile(feat_is, np.linspace(0, 1, n_quantiles + 1))
                q_bins[0] = -np.inf
                q_bins[-1] = np.inf
                q_idx = np.digitize(feat_is, q_bins) - 1

                mean_bot = np.nanmean(Y_IS[q_idx == 0])
                mean_top = np.nanmean(Y_IS[q_idx == n_quantiles - 1])
                spread_is = mean_top - mean_bot
            except Exception:
                continue

            ic_oos = _numpy_spearman_ic(feat_oos, Y_OOS)
            if np.isnan(ic_oos):
                continue

            ic_drop = abs(ic_oos) / abs(ic_is)
            if ( abs(ic_drop - 1 ) < 0.5) and (spread_is * np.sign(ic_is) > 20):
                selected_features_idx.append(f_idx)

    if len(selected_features_idx) > 0:
        return (i, selected_features_idx)
    return None


def _process_one_day(i, unique_dates, date_bounds, X_arr, Y_arr, dates_arr, weight_arr, current_X, Y_col, normal_mask, ab_mask, n_training_days, precomputed_features=None, alpha=10.0):
    """
    [Phase 2] 執行單日 rolling 訓練與預測的 worker function。
    """
    current_date = unique_dates[i]
    start_idx = max(0, i - n_training_days)

    if i == 0:
        return None

    train_start_date = unique_dates[start_idx]
    train_end_date = unique_dates[i - 1]

    train_row_start = date_bounds[train_start_date][0]
    train_row_end = date_bounds[train_end_date][1]

    test_row_start = date_bounds[current_date][0]
    test_row_end = date_bounds[current_date][1]

    if test_row_start == test_row_end:
        return None

    train_normal = normal_mask[train_row_start:train_row_end]
    train_ab = ab_mask[train_row_start:train_row_end]

    X_train_chunk = X_arr[train_row_start:train_row_end]
    Y_train_chunk = Y_arr[train_row_start:train_row_end]

    X_test_chunk = X_arr[test_row_start:test_row_end]

    if precomputed_features is not None and i in precomputed_features:
        selected_indices = precomputed_features[i]
    else:
        selected_indices = list(range(len(current_X)))

    if len(selected_indices) == 0:
        selected_indices = list(range(len(current_X)))

    # ⚠️ 關鍵修改：加上 .copy()，避免不同日期的 Worker 互相污染資料
    X_train_chunk_filtered = X_train_chunk[:, selected_indices].copy()
    X_test_chunk_filtered = X_test_chunk[:, selected_indices].copy()

    # ====== 🌟 新增：找出這段 Training Window 的「前半段 (IS)」來計算邊界 ======
    mid_idx = start_idx + (i - 1 - start_idx) // 2
    split_date = unique_dates[mid_idx]
    split_row_start = date_bounds[split_date][0]
    
    # 確保切點在範圍內，並計算相對於 train_chunk 的 local index
    split_row_start = max(train_row_start, min(split_row_start, train_row_end))
    local_split_idx = split_row_start - train_row_start

    if local_split_idx > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # 取出前半段
            X_IS_filtered = X_train_chunk_filtered[:local_split_idx]
            
            # 計算 1% 和 99% 邊界
            lower_bounds = np.nanpercentile(X_IS_filtered, 1, axis=0)
            upper_bounds = np.nanpercentile(X_IS_filtered, 99, axis=0)
            
            # Clip 整個 Training 區塊與 Testing 區塊
            np.clip(X_train_chunk_filtered, lower_bounds, upper_bounds, out=X_train_chunk_filtered)
            np.clip(X_test_chunk_filtered, lower_bounds, upper_bounds, out=X_test_chunk_filtered)


    res_normal = np.full(test_row_end - test_row_start, np.nan)
    res_ab = np.full(test_row_end - test_row_start, np.nan)
    coeff_normal = None
    coeff_ab = None
    std_coeff_normal = None
    std_coeff_ab = None

    if np.any(train_normal):
        model_normal = Ridge(alpha=alpha)
        X_normal = X_train_chunk_filtered[train_normal]
        model_normal.fit(X_normal, Y_train_chunk[train_normal])
        res_normal = model_normal.predict(X_test_chunk_filtered)
        coeff_normal = model_normal.coef_.copy()
        feat_std = np.std(X_normal, axis=0)
        feat_std[feat_std == 0] = 1.0
        std_coeff_normal = coeff_normal * feat_std

    if np.any(train_ab):
        model_ab = Ridge(alpha=alpha)
        X_ab = X_train_chunk_filtered[train_ab]
        model_ab.fit(X_ab, Y_train_chunk[train_ab])
        res_ab = model_ab.predict(X_test_chunk_filtered)
        coeff_ab = model_ab.coef_.copy()
        feat_std = np.std(X_ab, axis=0)
        feat_std[feat_std == 0] = 1.0
        std_coeff_ab = coeff_ab * feat_std

    return {
        'start': test_row_start,
        'end': test_row_end,
        'pred_normal': res_normal,
        'pred_ab': res_ab,
        'date': current_date,
        'coeff_normal': coeff_normal,
        'coeff_ab': coeff_ab,
        'std_coeff_normal': std_coeff_normal,
        'std_coeff_ab': std_coeff_ab,
        'selected_features': selected_indices,
    }

def train_and_predict_ridge_rolling(signal_df: pd.DataFrame, X_list: list, Y_col: str, n_training_days: int = 60, n_jobs: int = -1, feature_screening: bool = False, alpha: float = 10.0) -> pd.DataFrame:
    print(f"🔍 目前預測目標: {Y_col} | Rolling Training 視窗: {n_training_days} 天")

    df = signal_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    print("🔄 為了 O(1) NumPy 加速，強制將資料依照 Date, QuoteCode, TransTime 進行排序...")
    df = df.sort_values(['Date', 'QuoteCode', 'TransTime']).reset_index(drop=True)

    total_rows = len(df)
    threshold = total_rows * 0.01
    print(f"📊 資料總筆數: {total_rows} | 0.1% 容忍閾值: {threshold:.2f} 筆")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("🔄 填補特徵遺失值 (依據前一交易日之中位數)...")
    for c in X_list:
        if df[c].isna().any():
            daily_median = df.groupby('Date')[c].median()
            shifted_daily_median = daily_median.shift(1)
            df[c] = df[c].fillna(df['Date'].map(shifted_daily_median))

    cols_to_drop_rows = []
    cols_to_remove_from_X = []

    for col in X_list:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            if nan_count < threshold:
                cols_to_drop_rows.append(col)
            else:
                cols_to_remove_from_X.append(col)

    current_X = [col for col in X_list if col not in cols_to_remove_from_X]

    if cols_to_remove_from_X:
        print(f"⚠️ 以下特徵 NaN 數量超過閾值，已從 X_list 中移除：")
        for col in cols_to_remove_from_X:
            nan_count = df[col].isna().sum()
            print(f"   - {col} (NaN: {nan_count} 筆)")

    if df[Y_col].isna().sum() > 0 and Y_col not in cols_to_drop_rows:
        cols_to_drop_rows.append(Y_col)

    if cols_to_drop_rows:
        print(f"🧹 以下欄位 NaN 數量極少 (低於閾值)，將直接剔除帶有 NaN 的資料列：")
        for col in cols_to_drop_rows:
            nan_count = df[col].isna().sum()
            print(f"   - {col} (NaN: {nan_count} 筆)")

        df = df.dropna(subset=cols_to_drop_rows).copy()
        df = df.reset_index(drop=True)
        print(f"✅ 剔除後剩餘資料總筆數: {len(df)}")

    # ==========================================
    # 🌟 關鍵修復：將 QuoteCode 從 X_list 抽離
    # ==========================================
    if 'QuoteCode' in current_X:
        current_X.remove('QuoteCode')
        print("🧹 關鍵修復：已將 'QuoteCode' 從 X_list 抽離，確保陣列為純數值 (float32)！")

    print("-" * 40)
    print(f"🚀 最終使用的 X_list (共 {len(current_X)} 個特徵):")
    print(current_X)
    print("-" * 40)

    if not current_X:
        raise ValueError("所有的 X 特徵都因為 NaN 太多被移除了，無法建立模型！")

    print("🔍 檢查是否有「全段期間皆為常數」(Zero Variance) 的特徵...")
    cols_zero_var = []
    for col in current_X:
        if df[col].max() == df[col].min():
            cols_zero_var.append(col)

    if cols_zero_var:
        current_X = [col for col in current_X if col not in cols_zero_var]
        print(f"⚠️ 以下 {len(cols_zero_var)} 個特徵為完全不變的常數，已從 X_list 中剔除：")
        print("   - " + ", ".join(cols_zero_var))

    if not current_X:
        raise ValueError("所有的 X 特徵都因為被判定全為常數且移除，導致清單為空，無法建立模型！")

    print("🔄 計算 Lot Mask (BidPrice1 cumsum) 與準備 NumPy 陣列")
    if 'BidPrice1' in df.columns:
        lot_mask = df.groupby(['Date', 'QuoteCode'])['BidPrice1'].cumcount() < 10
        weight_arr = df['BidPrice1'].to_numpy()
    else:
        lot_mask = pd.Series(True, index=df.index)
        weight_arr = np.ones(len(df))

    normal_mask = (lot_mask & (df['isAbnormalDate'] == 0)).to_numpy()
    ab_mask = (lot_mask & (df['isAbnormalDate'] == 1)).to_numpy()

    # ==========================================
    # 🌟 強制記憶體連續與使用純數值型態 (float32)
    # ==========================================
    X_arr = np.ascontiguousarray(df[current_X].to_numpy(dtype=np.float32))
    Y_arr = np.ascontiguousarray(df[Y_col].to_numpy(dtype=np.float32))

    # ==========================================
    # 🌟 獨立製作 QuoteCode 整數陣列供分層取樣使用
    # ==========================================
    if 'QuoteCode' in df.columns:
        quotes_arr = pd.factorize(df['QuoteCode'])[0].astype(np.int32)
    else:
        quotes_arr = np.zeros(len(df), dtype=np.int32)

    dates_s = df['Date'].dt.date
    dates_arr = dates_s.to_numpy()
    unique_dates = np.unique(dates_arr)

    date_bounds = {}
    for d in unique_dates:
        start_row = np.searchsorted(dates_arr, d, side='left')
        end_row = np.searchsorted(dates_arr, d, side='right')
        date_bounds[d] = (start_row, end_row)

    pred_col_name = f"{Y_col}_pred"
    pred_ab_col_name = f"{Y_col}_ABpred"

    pred_normal_full = np.full(len(df), np.nan)
    pred_ab_full = np.full(len(df), np.nan)

    precomputed_features = None

    if feature_screening:
        screening_n_jobs = min(16, os.cpu_count() or 16)
        print(f"🔬 Phase 1: 預先計算 Feature Screening (worker 數: {screening_n_jobs}, 共 {len(unique_dates)} 天)...")
        print(f"   使用分層取樣 (Stratified Sampling)，每個商品保留 20% 且至少 3 筆")

        screening_results = Parallel(n_jobs=screening_n_jobs, require='sharedmem', verbose=10)(
            delayed(_screen_features_for_one_day)(
                i, unique_dates, date_bounds, X_arr, Y_arr, quotes_arr, current_X, n_training_days
            ) for i in range(len(unique_dates))
        )

        precomputed_features = {}
        screened_count = 0
        for result in screening_results:
            if result is not None:
                day_idx, feat_indices = result
                precomputed_features[day_idx] = feat_indices
                screened_count += 1

        print(f"✅ Phase 1 完成！共 {screened_count}/{len(unique_dates)} 天有篩選出特徵。")

    print(f"▶️ Phase 2: 開始對 {len(unique_dates)} 個交易日進行 純Numpy極速 平行 Rolling Update (Core 數: {n_jobs})...")

    results = Parallel(n_jobs=n_jobs, require='sharedmem', verbose=10)(
        delayed(_process_one_day)(
            i, unique_dates, date_bounds, X_arr, Y_arr, dates_arr, weight_arr, current_X, Y_col, normal_mask, ab_mask, n_training_days, precomputed_features, alpha
        ) for i in range(len(unique_dates))
    )

    screening_log = {}
    coeff_log = {}

    for res in results:
        if res is not None:
            r_start = res['start']
            r_end = res['end']
            pred_normal_full[r_start:r_end] = res['pred_normal']
            pred_ab_full[r_start:r_end] = res['pred_ab']

            date_str = str(res['date'])
            sel_idx = res['selected_features']
            sel_names = [current_X[j] for j in sel_idx]
            screening_log[date_str] = sel_names

            entry = {'selected_features': sel_names}
            if res['coeff_normal'] is not None:
                entry['normal_coeff'] = dict(zip(sel_names, res['coeff_normal'].tolist()))
                entry['normal_std_coeff'] = dict(zip(sel_names, res['std_coeff_normal'].tolist()))
            if res['coeff_ab'] is not None:
                entry['ab_coeff'] = dict(zip(sel_names, res['coeff_ab'].tolist()))
                entry['ab_std_coeff'] = dict(zip(sel_names, res['std_coeff_ab'].tolist()))
            coeff_log[date_str] = entry

    df[pred_col_name] = pred_normal_full
    df[pred_ab_col_name] = pred_ab_full

    print("✅ 平行 Rolling 更新預測完成！")
    
    # --- [釋放龐大記憶體] 發送給 gc，防止 Jupyter 重複執行時把 Swap 塞爆 ---
    del X_arr, Y_arr, dates_arr, quotes_arr, normal_mask, ab_mask, weight_arr
    del pred_normal_full, pred_ab_full, results
    if precomputed_features is not None:
        del precomputed_features
    gc.collect()

    return {
        'df': df,
        'screening_log': screening_log,
        'coeff_log': coeff_log,
        'feature_names': current_X,
    }