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
import warnings
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed

# 隱藏 Scipy/Sklearn 底層矩陣 condition number 警告
warnings.filterwarnings("ignore", message=".*ill-conditioned.*")

def _process_one_day(i, unique_dates, date_bounds, X_arr, Y_arr, normal_mask, ab_mask, n_training_days):
    """
    執行單日 rolling 訓練與預測的 worker function
    全域使用 NumPy 切片與陣列，徹底消除 Pandas 的跨 Process Overhead。
    """
    current_date = unique_dates[i]
    start_idx = max(0, i - n_training_days)
    
    if i == 0:
        return None
        
    train_start_date = unique_dates[start_idx]
    train_end_date = unique_dates[i - 1]
    
    # 利用已經算好的 Boundary 做 O(1) 的超高速切片
    train_row_start = date_bounds[train_start_date][0]
    train_row_end = date_bounds[train_end_date][1]
    
    test_row_start = date_bounds[current_date][0]
    test_row_end = date_bounds[current_date][1]
    
    if test_row_start == test_row_end:
        return None
        
    # 取出對應範圍內的 Boolean Mask
    train_normal = normal_mask[train_row_start:train_row_end]
    train_ab = ab_mask[train_row_start:train_row_end]
    
    # 取出特徵與標籤矩陣的 View (不佔額外記憶體)
    X_train_chunk = X_arr[train_row_start:train_row_end]
    Y_train_chunk = Y_arr[train_row_start:train_row_end]
    
    X_test_chunk = X_arr[test_row_start:test_row_end]
    
    res_normal = np.full(test_row_end - test_row_start, np.nan)
    res_ab = np.full(test_row_end - test_row_start, np.nan)
    
    # 進行訓練與預測 (如果該區間內有符合條件的資料)
    if np.any(train_normal):
        model_normal = Ridge(alpha=10.0)
        # 用 boolean mask 抽出資料，訓練
        model_normal.fit(X_train_chunk[train_normal], Y_train_chunk[train_normal])
        res_normal = model_normal.predict(X_test_chunk)
        
    if np.any(train_ab):
        model_ab = Ridge(alpha=10.0)
        model_ab.fit(X_train_chunk[train_ab], Y_train_chunk[train_ab])
        res_ab = model_ab.predict(X_test_chunk)
        
    return {
        'start': test_row_start,
        'end': test_row_end,
        'pred_normal': res_normal,
        'pred_ab': res_ab
    }

def train_and_predict_ridge_rolling(signal_df: pd.DataFrame, X_list: list, Y_col: str, n_training_days: int = 60, n_jobs: int = -1) -> pd.DataFrame:
    """
    每天 rolling 更新：取最新一天 (預測目標)，並以過去 n_training_days 個交易日做為訓練區間。
    (Numpy + Joblib 極速平行版)
    """
    print(f"🔍 目前預測目標: {Y_col} | Rolling Training 視窗: {n_training_days} 天")
    
    df = signal_df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
        
    print("🔄 為了 O(1) NumPy 加速，強制將資料依照 Date 進行排序...")
    df = df.sort_values('Date').reset_index(drop=True)
        
    # 計算 0.1% 的閾值 (threshold)
    total_rows = len(df)
    threshold = total_rows * 0.01
    print(f"📊 資料總筆數: {total_rows} | 0.1% 容忍閾值: {threshold:.2f} 筆")
    
    # 預先將所有的 inf 替換為 NaN，確保接下來的 NaN 統計包含原先為 inf 的異常值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # ==========================================
    # 1. 檢查 X_list 裡面的 NaN 狀況並分類
    # ==========================================
    cols_to_drop_rows = []
    cols_to_remove_from_X = []
    
    for col in X_list:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            if nan_count < threshold:
                cols_to_drop_rows.append(col)
            else:
                cols_to_remove_from_X.append(col)
                
    # ==========================================
    # 2. 執行移除特徵 (Columns) 動作
    # ==========================================
    current_X = [col for col in X_list if col not in cols_to_remove_from_X]
    
    if cols_to_remove_from_X:
        print(f"⚠️ 以下特徵 NaN 數量超過閾值，已從 X_list 中移除：")
        for col in cols_to_remove_from_X:
            nan_count = df[col].isna().sum()
            print(f"   - {col} (NaN: {nan_count} 筆)")
            
    # ==========================================
    # 3. 處理 Y 的 NaN 與執行 dropna (刪除資料列)
    # ==========================================
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
    # 4. 顯示最終要進模型的 X_list
    # ==========================================
    print("-" * 40)
    print(f"🚀 最終使用的 X_list (共 {len(current_X)} 個特徵):")
    print(current_X)
    print("-" * 40)
    
    if not current_X:
        raise ValueError("所有的 X 特徵都因為 NaN 太多被移除了，無法建立模型！")

    # ==========================================
    # 4.5. 檢查常數變數 (Zero Variance) 並過濾
    # ==========================================
    print("🔍 檢查是否有「全段期間皆為常數」(Zero Variance) 的特徵...")
    cols_zero_var = []
    for col in current_X:
        # 在去除 NaN 後，如果最大值等於最小值代表是不變的常數
        if df[col].max() == df[col].min():
            cols_zero_var.append(col)
            
    if cols_zero_var:
        current_X = [col for col in current_X if col not in cols_zero_var]
        print(f"⚠️ 以下 {len(cols_zero_var)} 個特徵為完全不變的常數，已從 X_list 中剔除：")
        # 避免清單太長，單行顯示
        print("   - " + ", ".join(cols_zero_var))
            
    if not current_X:
        raise ValueError("所有的 X 特徵都因為被判定全為常數且移除，導致清單為空，無法建立模型！")

    # ==========================================
    # 5. 全域事前計算 Mask (加快運算) 並換算為純 NumPy
    # ==========================================
    print("🔄 計算 Lot Mask (BidPrice1 cumsum) 與準備 NumPy 陣列")
    if 'BidPrice1' in df.columns:
        lot_mask = df.groupby(['Date', 'QuoteCode'])['BidPrice1'].cumsum() / 10 < 500
    else:
        lot_mask = pd.Series(True, index=df.index)
        
    # 直接產出最終要給 worker 用的一維布林 NumPy Array 
    # 取代原本 pandas 的多重 DataFrame 切割
    normal_mask = (lot_mask & (df['isAbnormalDate'] == 0)).to_numpy()
    ab_mask = (lot_mask & (df['isAbnormalDate'] == 1)).to_numpy()
    
    # 準備特徵和標籤的 NumPy 陣列矩陣
    X_arr = df[current_X].to_numpy()
    Y_arr = df[Y_col].to_numpy()
        
    # ==========================================
    # 6. 計算每天的在 Numpy 陣列的切片區間界標 (O(1) Boundary Lookup)
    # ==========================================
    dates_s = df['Date'].dt.date
    dates_arr = dates_s.to_numpy()
    unique_dates = np.unique(dates_arr)

    # 用 searchsorted 瞬間抓出每一天資料落在哪幾行 (start, end)
    date_bounds = {}
    for d in unique_dates:
        start_row = np.searchsorted(dates_arr, d, side='left')
        end_row = np.searchsorted(dates_arr, d, side='right')
        date_bounds[d] = (start_row, end_row)

    # ==========================================
    # 7. 利用 Rolling Date 的方式建立並平行訓練 Ridge 模型
    # ==========================================
    pred_col_name = f"{Y_col}_pred"
    pred_ab_col_name = f"{Y_col}_ABpred"
    
    # 預先把結果用 numpy 生成
    pred_normal_full = np.full(len(df), np.nan)
    pred_ab_full = np.full(len(df), np.nan)
    
    print(f"▶️ 開始對 {len(unique_dates)} 個交易日進行 純Numpy極速 平行 Rolling Update (Core 數: {n_jobs})...")
    
    # 將需要跑的 i 和共通的資料傳給並發任務
    # 預設 backend="loky" 且陣列龐大時，Joblib 會自動以 memmap 來避免搬運記憶體拷貝開銷
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_one_day)(
            i, unique_dates, date_bounds, X_arr, Y_arr, normal_mask, ab_mask, n_training_days
        ) for i in range(len(unique_dates))
    )
    
    # 把平行計算完的預測填回預備的 1D numpy array
    for res in results:
        if res is not None:
            r_start = res['start']
            r_end = res['end']
            pred_normal_full[r_start:r_end] = res['pred_normal']
            pred_ab_full[r_start:r_end] = res['pred_ab']

    # 一次性寫回 Pandas，效率最高
    df[pred_col_name] = pred_normal_full
    df[pred_ab_col_name] = pred_ab_full

    print("✅ 平行 Rolling 更新預測完成！")
    return df
