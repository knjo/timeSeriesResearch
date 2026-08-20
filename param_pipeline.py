"""
Time-Series Signal Ridge Regression Parameter Grid Search Pipeline
========================================================
Grid search over:
  - n_training_days: [30, 40, 50, 60, 90]
  - Ridge alpha:     [0.1, 1, 10, 100]

For each (n_training_days, alpha) combination, collects metrics on:
  1. Normal-day prediction quality (qcut-4 mean of TakerSell_CloseBP)
  2. Abnormal-day prediction quality (qcut-4 minus qcut-0 mean)
  3. Backtest performance (total profit, MDD, time-to-high, avg BP, MDD/mean)

Outputs:
  - CSV reports  -> ./report/
  - Heatmap PNGs -> ./report/
"""

import os
import sys
import json
import gc
import warnings

# Force single-threaded BLAS/NumPy to avoid oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.research.rolling_model import train_and_predict_ridge_rolling

# ── constants ────────────────────────────────────────────────────────────────
TEST_PREFIX = "based_line_strick"  # 修改成你想要的測試名稱
REPORT_DIR = os.path.join(SCRIPT_DIR, "report", TEST_PREFIX)
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
os.makedirs(REPORT_DIR, exist_ok=True)

N_TRAINING_DAYS_LIST = [ 5, 10, 20, 30, 40, 50, 60, 90, 120]
ALPHA_LIST           = [ 0.1, 1, 10, 30 ,100]

# columns exactly as defined in the research notebook
BASED_COLUMNS   = ['QuoteCode', 'ChannelSeq', 'InOut', 'Overshoot', 'TotalFillLots',
                   'BidPrice1', 'BidLots1', 'BidLots5', 'AskPrice1', 'AskLots1',
                   'AskLots5', 'FillLots']
FEATURE_COLUMNS = ['ToLow', 'ToHigh', 'Low_High', 'TickSize', 'TickBP', 'ToRef',
                   'ToOpen', 'FillLots_atLow', 'FillLots_atHigh', 'B1_A1B1',
                   'B1_B1B5', 'B12_B1B5', 'A1_A1A5', 'A12_A1A5', 
                   'RemainSeconds', 'B1_Last', 'Total_Last']
CROSS_COLUMNS   = ['AmountRank', 'netAmountRank', 'TotalFillLotsRank',
                   'netTotalFillLotsRank', 'ToRefRank']
DAY_COLUMNS     = ['dealer_hedging_netLots', 'dealer_netLots',
                   'foreign_dealer_netLots', 'foreign_netLots',
                   'investment_netLots', 'margin_netLots', 'short_netLots',
                   'nextday_allow_day_trade_mark', 'trading_volume_noOdd',
                   'big_buy_lots', 'big_buy_ToCloseBP', 'big_buy_300sBP',
                   'big_sell_lots', 'big_sell_ToCloseBP', 'big_sell_300sBP',
                   'hft_participation', 'hft_buy_ToCloseBP', 'hft_buy_300sBP',
                   'hft_sell_ToCloseBP', 'hft_sell_300sBP',
                   'day_amount_rank', 'day_lots_rank']

ALL_FEATURE     = BASED_COLUMNS + FEATURE_COLUMNS + CROSS_COLUMNS + DAY_COLUMNS
LABEL           = 'TakerSell_CloseBP_net'


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_and_prepare_signal_df() -> pd.DataFrame:
    """Replicate the notebook's data-loading & pre-processing steps exactly.

    The notebook:
      1. Reads all parquet files in ./data (skipping the first file)
      2. Filters: BidPrice1 * BidPrice1 > 0
      3. Computes ToRef, TakerSell_CloseBP (with stop-loss at 1.08*RefPrice)
      4. Applies PriceCondition & AmountCondition
      5. Creates TakerSell_CloseBP_net (subtract per Date+QuoteCode mean)
      6. Sets isAbnormalDate from abnormal_dates.json
      7. Creates B1_Last, Total_Last
    """
    print("📂 Loading data from ./data …")

    files = sorted([
        f for f in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, f)) and f.endswith('.parquet')
    ])

    # Notebook skips the first file: `for f in tqdm(file_names[1:]):`
    files = files[-300:]
    print(f"   Found {len(files)} parquet files to process")

    data = []
    targetLabel = 'BidPrice1'
    rankThres = 150

    for f in tqdm(files, desc="Loading parquets"):
        fpath = os.path.join(DATA_DIR, f)
        try:
            stock_df = pd.read_parquet(fpath)
            stock_df = stock_df[stock_df[targetLabel] * stock_df['BidPrice1'] > 0]

            stock_df['ToRef'] = ((stock_df['BidPrice1'] - stock_df['RefPrice'])
                                 / stock_df['RefPrice']).round(4)

            stock_df['TakerSell_CloseBP'] = (
                (stock_df[targetLabel] - stock_df['Close'])
                / stock_df[targetLabel] * 10000
            )
            # Stop-loss at 1.08 * RefPrice
            stop_mask = stock_df['FutureHigh'] > stock_df['RefPrice'] * 1.08
            stock_df.loc[stop_mask, 'TakerSell_CloseBP'] = (
                (stock_df.loc[stop_mask, targetLabel]
                 - stock_df.loc[stop_mask, 'RefPrice'] * 1.08
                 - stock_df.loc[stop_mask, 'TickSize'] * 2)
                / stock_df.loc[stop_mask, targetLabel] * 10000
            )

            PriceCondition = (stock_df['ToRef'] > -0.015) & (stock_df['ToRef'] < 0.05)
            AmountCondition = (
                (stock_df['day_amount_rank'] <= rankThres)
                | (stock_df['AmountRank_canDayTrade'] <= rankThres)
            )

            temp = stock_df[PriceCondition & AmountCondition].reset_index(drop=True)
            temp['Date'] = f.split('.')[0]
            data.append(temp)
        except Exception as e:
            print(f'   [WARN] Failed on {f}: {e}')
            continue

    signal_df = pd.concat(data, ignore_index=True)
    del data
    gc.collect()

    # TakerSell_CloseBP_net (per Date+QuoteCode de-mean)
    signal_df['TakerSell_CloseBP_net'] = (
        signal_df['TakerSell_CloseBP']
        - signal_df.groupby(['Date','QuoteCode'])['TakerSell_CloseBP'].transform('mean')
    )

    # isAbnormalDate flag
    json_path = os.path.join(SCRIPT_DIR, 'abnormal_dates.json')
    with open(json_path, 'r') as fj:
        abnormal_date_list = json.load(fj)
    signal_df['isAbnormalDate'] = signal_df['Date'].isin(abnormal_date_list).astype(int)

    # Derived features
    signal_df['B1_Last']    = signal_df['BidLots1'] / signal_df['avg_bidLots1']
    signal_df['Total_Last'] = signal_df['TotalFillLots'] / signal_df['last_fillLots']

    print(f"✅ signal_df ready – {len(signal_df):,} rows")
    return signal_df


def _apply_risk_filter(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the risk / position filters exactly as specified in the notebook.

    ⚠️ 記憶體優化：不做整份 DataFrame 的 .copy()，
    而是先用條件篩選把大部分資料過濾掉，再對小的子集計算 PosLots / Position。
    """
    # Step 1: 先做 signal condition（大幅縮小資料量）
    condition = (
        (signal_df['TakerSell_CloseBP_net_pred'] * (signal_df['isAbnormalDate'] == 0) > 40)
        | (signal_df['TakerSell_CloseBP_net_ABpred'] * (signal_df['isAbnormalDate'] == 1) > 40)
    )
    df = signal_df.loc[condition].reset_index(drop=True)

    # Step 2: 基本篩選（進一步縮小）
    basic_mask = (df['BidPrice1'] < 1000)
    df = df.loc[basic_mask].reset_index(drop=True)

    # Step 3: 在已經大幅縮小的子集上才計算 PosLots / Position
    df['PosLots']  = df.groupby(['Date', 'QuoteCode'])['BidPrice1'].cumcount() + 1
    df['Position'] = df.groupby(['Date', 'QuoteCode'])['BidPrice1'].cumsum() / 10

    pos_mask = ~((df['PosLots'] > 1) & (df['Position'] > 200))
    return df.loc[pos_mask].reset_index(drop=True)


def _compute_metrics(signal_df: pd.DataFrame) -> dict:
    """Compute all three sets of metrics for a single grid point.

    ⚠️ 記憶體優化：不做大型 DataFrame 的 .copy()，
    改用 pd.qcut 直接產生 Series，避免複製整份資料。
    """
    metrics: dict = {}
    pred_col    = 'TakerSell_CloseBP_net_pred'
    ab_pred_col = 'TakerSell_CloseBP_net_ABpred'

    # ── Metric 1: Normal-day qcut-4 mean ─────────────────────────────────
    #    不用 .copy()，用 boolean mask 做 view，qcut 只產生 Series
    normal_mask = signal_df['isAbnormalDate'] == 0
    try:
        qcut_labels = pd.qcut(signal_df.loc[normal_mask, pred_col], 5,
                              labels=False, duplicates='drop')
        q_max = qcut_labels.max()
        metrics['normal_qcut4_mean'] = float(
            signal_df.loc[normal_mask, 'TakerSell_CloseBP']
                     .loc[qcut_labels == q_max].mean()
        )
    except Exception:
        metrics['normal_qcut4_mean'] = np.nan

    # ── Metric 2: Abnormal-day qcut-4 minus qcut-0 mean ─────────────────
    ab_mask = signal_df['isAbnormalDate'] == 1
    try:
        qcut_labels = pd.qcut(signal_df.loc[ab_mask, ab_pred_col], 5,
                              labels=False, duplicates='drop')
        q_max = qcut_labels.max()
        q_min = qcut_labels.min()
        bp_series = signal_df.loc[ab_mask, 'TakerSell_CloseBP']
        q4_mean = float(bp_series.loc[qcut_labels == q_max].mean())
        q0_mean = float(bp_series.loc[qcut_labels == q_min].mean())
        metrics['abnormal_qcut_diff'] = q4_mean - q0_mean
    except Exception:
        metrics['abnormal_qcut_diff'] = np.nan

    # ── Metric 3: Backtest via risk-filtered DataFrame ───────────────────
    risk_df = _apply_risk_filter(signal_df)

    # Apply totalPosition filter (from notebook)
    risk_df['totalPosition'] = risk_df.groupby('Date')['BidPrice1'].cumsum() / 10
    risk_df = risk_df[risk_df['totalPosition'] <= 40000].reset_index(drop=True)

    FEE = 19.3
    pnl_arr = risk_df['BidPrice1'].values * (risk_df['TakerSell_CloseBP'].values - FEE) / 10
    pos_arr = risk_df['BidPrice1'].values * 1000

    # daily aggregation – 用 numpy 減少 pandas overhead
    dates = risk_df['Date'].values
    daily = pd.DataFrame({'Date': dates, 'PnL': pnl_arr, 'Position': pos_arr})
    daily = daily.groupby('Date', sort=True).agg(
        PnL=('PnL', 'sum'),
        Position=('Position', 'sum'),
    )
    del risk_df
    gc.collect()

    cum_pnl = daily['PnL'].cumsum()
    hwm     = cum_pnl.cummax()
    dd      = cum_pnl - hwm

    total_pnl  = daily['PnL'].sum()
    avg_pos    = daily['Position'].mean()
    mdd_amount = dd.min()  # negative

    # time-to-high (longest drawdown stretch in days)
    is_not_high  = dd < 0
    dd_groups    = (~is_not_high).cumsum()
    dd_durations = is_not_high[is_not_high].groupby(dd_groups).count()
    time_to_high = int(dd_durations.max()) if not dd_durations.empty else 0

    # avg BP per unit capital
    total_pos = daily['Position'].sum()
    avg_bp    = (total_pnl / total_pos * 10000) if total_pos != 0 else 0.0

    # MDD / mean-daily-pnl ratio
    mean_pnl       = daily['PnL'].mean()
    mdd_mean_ratio = (abs(mdd_amount) / mean_pnl) if mean_pnl != 0 else 0.0

    metrics['bt_total_profit']   = total_pnl / 10000
    metrics['bt_mdd']            = mdd_amount / 10000
    metrics['bt_time_to_high']   = time_to_high
    metrics['bt_avg_bp']         = avg_bp
    metrics['bt_mdd_mean_ratio'] = mdd_mean_ratio

    return metrics


def _save_heatmap(df_pivot: pd.DataFrame, title: str, filename: str,
                  fmt: str = ".2f", cmap: str = "RdYlGn"):
    """Draw a seaborn heatmap and save to ./report/."""
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(df_pivot, annot=True, fmt=fmt, cmap=cmap,
                     linewidths=0.5, linecolor='grey')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('n_training_days')
    ax.set_ylabel('alpha')
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   💾 Saved heatmap → {path}")


# ── main pipeline ────────────────────────────────────────────────────────────

def main():
    # 1. Load & merge data ONCE
    signal_df_raw = _load_and_prepare_signal_df()

    # Container: list of dicts (one per grid point)
    results = []

    total_combos = len(N_TRAINING_DAYS_LIST) * len(ALPHA_LIST)
    idx = 0

    for n_days in N_TRAINING_DAYS_LIST:
        for alpha in ALPHA_LIST:
            idx += 1
            print(f"\n{'='*60}")
            print(f"🔄 Grid [{idx}/{total_combos}]  n_training_days={n_days}  alpha={alpha}")
            print(f"{'='*60}")

            # ⚠️ 不需要在這裡做 .copy()！
            # rolling_model.py 內部 (第 285 行) 已經會做 df = signal_df.copy()
            # 如果外面再做一次，記憶體就同時存在 3 份 14M 行的 DataFrame，直接 OOM。
            #
            # 直接傳 signal_df_raw 進去即可 — model 內部的 copy + sort_values
            # 會產生一份獨立的工作副本，不會影響 signal_df_raw。

            result = train_and_predict_ridge_rolling(
                signal_df_raw,
                FEATURE_COLUMNS,
                LABEL,
                n_training_days=n_days,
                feature_screening=True,
                alpha=alpha,
            )

            # 只保下 result['df']，立即釋放其他東西
            out_df = result['df']
            del result
            gc.collect()

            # Compute metrics
            m = _compute_metrics(out_df)
            m['n_training_days'] = n_days
            m['alpha']           = alpha
            results.append(m)

            print(f"   📊 Metrics: {m}")

            # ── 立即釋放本輪的 14M DataFrame ──────────────────────────────
            del out_df
            gc.collect()

    # 2. Collect into DataFrame & save CSV
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(REPORT_DIR, "grid_search_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n📄 Results CSV → {csv_path}")
    print(df_results.to_string(index=False))

    # 3. Build heatmaps
    heatmap_specs = [
        ('normal_qcut4_mean',  'Normal-Day Prediction (qcut-4 TakerSell_CloseBP mean)',
         'heatmap_normal_qcut4.png',   '.2f', 'RdYlGn'),
        ('abnormal_qcut_diff', 'Abnormal-Day Prediction (qcut-4 − qcut-0 diff)',
         'heatmap_abnormal_diff.png',  '.2f', 'RdYlGn'),
        ('bt_total_profit',    'Backtest Total Profit (萬)',
         'heatmap_bt_profit.png',      '.2f', 'RdYlGn'),
        ('bt_mdd',             'Backtest MDD (萬)',
         'heatmap_bt_mdd.png',         '.2f', 'RdYlGn_r'),
        ('bt_time_to_high',    'Backtest Time-to-High (days)',
         'heatmap_bt_time_to_high.png','.0f', 'RdYlGn_r'),
        ('bt_avg_bp',          'Backtest Average BP per Unit',
         'heatmap_bt_avg_bp.png',      '.4f', 'RdYlGn'),
        ('bt_mdd_mean_ratio',  'Backtest MDD / Mean-Daily-PnL',
         'heatmap_bt_mdd_mean.png',    '.2f', 'RdYlGn_r'),
    ]

    for col, title, fname, fmt, cmap in heatmap_specs:
        pivot = df_results.pivot(index='alpha', columns='n_training_days', values=col)
        _save_heatmap(pivot, title, fname, fmt=fmt, cmap=cmap)

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
