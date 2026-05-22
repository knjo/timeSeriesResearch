"""
model_analysis.py
Rolling Model 診斷視覺化工具

使用方式:
    from model_analysis import plot_model_dashboard
    result = train_and_predict_ridge_rolling(...)
    plot_model_dashboard(result)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def _build_coeff_df(coeff_log, coeff_key='normal_std_coeff'):
    """
    將 coeff_log 轉成 DataFrame (日期 × 特徵)。
    coeff_key: 'normal_coeff', 'normal_std_coeff', 'ab_coeff', 'ab_std_coeff'
    """
    rows = {}
    for date_str, entry in coeff_log.items():
        if coeff_key in entry:
            rows[date_str] = entry[coeff_key]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def _build_screening_df(screening_log, feature_names):
    """
    將 screening_log 轉成 binary DataFrame (日期 × 特徵)。
    1 = 被選中, 0 = 未被選中
    """
    rows = {}
    for date_str, sel_names in screening_log.items():
        rows[date_str] = {f: (1 if f in sel_names else 0) for f in feature_names}
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def plot_feature_selection_frequency(result, top_n=25, figsize=(14, 6)):
    """
    特徵被選中的頻率 (所有天數中，該特徵被模型使用的比例)。
    若未啟用 feature_screening 則所有特徵都是 100%。
    """
    screening_log = result['screening_log']
    feature_names = result['feature_names']
    n_days = len(screening_log)

    if n_days == 0:
        print("無篩選記錄")
        return

    # 計算每個特徵的使用次數
    freq = {f: 0 for f in feature_names}
    for sel_names in screening_log.values():
        for f in sel_names:
            if f in freq:
                freq[f] += 1

    # 排序
    sorted_feats = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in sorted_feats]
    counts = [x[1] for x in sorted_feats]
    pcts = [c / n_days * 100 for c in counts]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(names)), pcts, color='steelblue', edgecolor='white', height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Selection Rate (%)')
    ax.set_title(f'Feature Selection Frequency (Top {top_n}, {n_days} days)')
    ax.invert_yaxis()

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%', va='center', fontsize=8)

    ax.set_xlim(0, max(pcts) * 1.15)
    plt.tight_layout()
    plt.show()


def plot_feature_selection_heatmap(result, top_n=30, figsize=(20, 8)):
    """
    特徵篩選熱力圖：日期(X) × 特徵(Y)，被選中=深色。
    只顯示出現頻率最高的 top_n 個特徵。
    """
    screening_log = result['screening_log']
    feature_names = result['feature_names']

    sel_df = _build_screening_df(screening_log, feature_names)
    if sel_df.empty:
        print("無篩選記錄")
        return

    # 按使用頻率排序
    freq = sel_df.sum().sort_values(ascending=False)
    top_feats = freq.index[:top_n].tolist()
    sel_df = sel_df[top_feats]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sel_df.T.values, aspect='auto', cmap='Blues', interpolation='nearest')

    # X 軸：日期 (間隔顯示)
    n_dates = len(sel_df)
    step = max(1, n_dates // 20)
    ax.set_xticks(range(0, n_dates, step))
    ax.set_xticklabels([sel_df.index[i].strftime('%Y-%m-%d') for i in range(0, n_dates, step)],
                       rotation=45, ha='right', fontsize=7)

    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=8)
    ax.set_title(f'Feature Selection Heatmap (Top {top_n} by frequency)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.show()


def plot_standardized_importance(result, model_type='normal', top_n=15, figsize=(16, 10)):
    """
    標準化係數重要性分析。

    標準化係數 = raw_coeff × std(X_feature)
    代表：該特徵變動 1 個標準差時，對預測值的影響量 (以 Y 的單位)。
    這樣不同尺度的特徵可以公平比較。

    圖表:
    - 上: 平均 |標準化係數| 的 Top N 排名 (Bar Chart)
    - 下: Top N 特徵的標準化係數隨時間變化 (Line Chart)
    """
    coeff_key = f'{model_type}_std_coeff'
    coeff_df = _build_coeff_df(result['coeff_log'], coeff_key)
    if coeff_df.empty:
        print(f"無 {model_type} 模型係數記錄")
        return

    # 計算每個特徵的平均 |標準化係數| (未被選中的天數視為係數為 0，藉此懲罰低頻選中的特徵)
    mean_abs = coeff_df.fillna(0).abs().mean().sort_values(ascending=False)
    top_feats = mean_abs.index[:top_n].tolist()

    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 1.5]})

    # ---- 上圖: Bar Chart ----
    ax = axes[0]
    vals = mean_abs[top_feats].values
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_feats)))
    bars = ax.barh(range(len(top_feats)), vals, color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=9)
    ax.set_xlabel('Mean |Standardized Coefficient|  (Y-unit per 1σ change)')
    ax.set_title(f'Feature Importance — {model_type.upper()} Model (Standardized Coefficients)')
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=8)

    # ---- 下圖: 時間序列 ----
    ax2 = axes[1]
    sub_df = coeff_df[top_feats[:8]]  # 只畫前 8 個避免太擁擠
    for col in sub_df.columns:
        ax2.plot(sub_df.index, sub_df[col].interpolate(method='linear').rolling(20, min_periods=1).mean(),
                 label=col, linewidth=1.2, alpha=0.85)
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Standardized Coefficient (20-day MA)')
    ax2.set_title(f'Top 8 Feature Coefficients Over Time — {model_type.upper()} Model')
    ax2.legend(loc='upper left', fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_coefficient_stability(result, model_type='normal', top_n=15, figsize=(14, 7)):
    """
    係數穩定性分析：Boxplot 顯示每個特徵的標準化係數分布。
    穩定的特徵 = 箱子窄且不跨零。
    """
    coeff_key = f'{model_type}_std_coeff'
    coeff_df = _build_coeff_df(result['coeff_log'], coeff_key)
    if coeff_df.empty:
        print(f"無 {model_type} 模型係數記錄")
        return

    mean_abs = coeff_df.abs().mean().sort_values(ascending=False)
    top_feats = mean_abs.index[:top_n].tolist()
    sub_df = coeff_df[top_feats]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot([sub_df[f].dropna().values for f in top_feats],
                    labels=top_feats, vert=False, patch_artist=True,
                    widths=0.6, showfliers=False,
                    medianprops=dict(color='red', linewidth=1.5))

    # 上色：中位數 > 0 藍色, < 0 紅色
    for i, (patch, feat) in enumerate(zip(bp['boxes'], top_feats)):
        median = sub_df[feat].median()
        patch.set_facecolor('#4A90D9' if median > 0 else '#D94A4A')
        patch.set_alpha(0.6)

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Standardized Coefficient Distribution')
    ax.set_title(f'Coefficient Stability — {model_type.upper()} Model (Top {top_n})')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_relative_importance_heatmap(result, model_type='normal', top_n=20, figsize=(20, 8)):
    """
    相對重要性熱力圖：日期(X) × 特徵(Y)。
    每日的 |標準化係數| 被歸一化為 100%，顯示每個特徵佔多少比重。
    """
    coeff_key = f'{model_type}_std_coeff'
    coeff_df = _build_coeff_df(result['coeff_log'], coeff_key)
    if coeff_df.empty:
        print(f"無 {model_type} 模型係數記錄")
        return

    # 歸一化：每一天的 |std_coeff| 加總為 100% (未被選中的特徵係數為 0)
    abs_df = coeff_df.fillna(0).abs()
    row_sum = abs_df.sum(axis=1)
    row_sum[row_sum == 0] = 1.0
    pct_df = abs_df.div(row_sum, axis=0) * 100

    mean_pct = pct_df.mean().sort_values(ascending=False)
    top_feats = mean_pct.index[:top_n].tolist()
    pct_sub = pct_df[top_feats]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pct_sub.T.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Relative Importance (%)', shrink=0.8)

    n_dates = len(pct_sub)
    step = max(1, n_dates // 20)
    ax.set_xticks(range(0, n_dates, step))
    ax.set_xticklabels([pct_sub.index[i].strftime('%Y-%m-%d') for i in range(0, n_dates, step)],
                       rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=8)
    ax.set_title(f'Daily Relative Importance (%) — {model_type.upper()} Model')
    ax.set_xlabel('Date')
    plt.tight_layout()
    plt.show()

def plot_pls_component_weights(result, comp=1, figsize=(14, 6)):
    """
    視覺化 PLS 降維時，各個 Day Feature 對 Component 的貢獻度 (Weights) 變化。
    comp: 1 代表 Component 1 (最大變異且最相關), 2 代表 Component 2
    """
    if 'pls_log' not in result or not result['pls_log']:
        print("無 PLS 權重記錄 (可能沒有輸入 Day Features 或全部被 Screening 剔除)")
        return

    comp_idx = comp - 1
    rows = {}
    for date_str, info in result['pls_log'].items():
        feats = info['features']
        weights = np.array(info['weights'])
        
        # 確保該天有算出這個 Component (有時特徵太少只會出 1 個 Comp)
        if weights.shape[1] > comp_idx:
            day_w = weights[:, comp_idx]
            rows[date_str] = dict(zip(feats, day_w))

    if not rows:
        print(f"無 Component {comp} 的數據")
        return

    df_w = pd.DataFrame.from_dict(rows, orient='index')
    df_w.index = pd.to_datetime(df_w.index)
    df_w = df_w.sort_index()

    fig, ax = plt.subplots(figsize=figsize)

    # 為了避免高頻雜訊，使用 10 天 MA 平滑線條
    smooth_w = df_w.rolling(10, min_periods=1).mean()

    # 使用較為鮮明的顏色庫
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, len(smooth_w.columns))]

    for i, col in enumerate(smooth_w.columns):
        ax.plot(smooth_w.index, smooth_w[col], label=col, linewidth=2, alpha=0.85, color=colors[i])

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax.set_title(f'How PLS Consumes Day Features — Component {comp} Weights (10-day MA)')
    ax.set_xlabel('Date')
    ax.set_ylabel('PLS Weight (Contribution)')
    
    # 將圖例放到圖表外部右側
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, borderaxespad=0.)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

import matplotlib.colors as mcolors

def plot_pls_weights_heatmap(result, comp=1, top_n=20, figsize=(16, 10)):
    """
    使用熱力圖視覺化 PLS Component 的權重變化。
    解決特徵過多時，折線圖變成「義大利麵」無法解讀的問題。
    
    參數:
    comp: 欲觀察的主成分 (1 或 2)
    top_n: 最多顯示平均影響力最大的前 N 個特徵
    """
    if 'pls_log' not in result or not result['pls_log']:
        print("無 PLS 權重記錄")
        return

    comp_idx = comp - 1
    rows = {}
    for date_str, info in result['pls_log'].items():
        feats = info['features']
        weights = np.array(info['weights'])
        if weights.shape[1] > comp_idx:
            day_w = weights[:, comp_idx]
            rows[date_str] = dict(zip(feats, day_w))

    if not rows:
        return

    # 1. 建立 DataFrame 並排序時間
    df_w = pd.DataFrame.from_dict(rows, orient='index')
    df_w.index = pd.to_datetime(df_w.index)
    df_w = df_w.sort_index()

    # 2. 為了減少高頻雜訊，使用 5 天 MA 平滑
    smooth_w = df_w.rolling(5, min_periods=1).mean()

    # 3. 挑選並排序最重要的特徵 (依照全期平均絕對權重)
    mean_abs_weight = smooth_w.abs().mean().sort_values(ascending=False)
    top_feats = mean_abs_weight.index[:top_n].tolist()
    
    # 取出前 N 大特徵，並轉置 (T) 讓特徵變成 Y 軸，日期變成 X 軸
    plot_data = smooth_w[top_feats].T

    # 4. 繪圖設定
    fig, ax = plt.subplots(figsize=figsize)
    
    # 設定發散型色階 (以 0 為中心，紅正藍負)
    vmax = plot_data.abs().max().max()
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap('RdBu_r') # 紅(正) - 白(0) - 藍(負)

    im = ax.imshow(plot_data.values, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    
    # 加入 Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('PLS Weight (Red: Positive, Blue: Negative)')

    # X 軸：日期 (每隔幾天標示一次避免太擠)
    n_dates = len(plot_data.columns)
    step = max(1, n_dates // 15)
    ax.set_xticks(range(0, n_dates, step))
    ax.set_xticklabels([plot_data.columns[i].strftime('%Y-%m-%d') for i in range(0, n_dates, step)],
                       rotation=45, ha='right', fontsize=9)

    # Y 軸：特徵名稱
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=10)

    ax.set_title(f'PLS Component {comp} Regime Shift Heatmap (Top {top_n} Features)', fontsize=14, pad=15)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Day Features (Sorted by Importance)', fontsize=11)
    
    plt.tight_layout()
    plt.show()

def plot_model_dashboard(result, model_type='normal'):
    """
    一次呼叫，產生完整的模型診斷報告：
    1. 特徵選擇頻率
    2. 標準化係數重要性 + 時間序列
    3. 係數穩定性 Boxplot
    4. 每日相對重要性熱力圖
    """
    print("=" * 60)
    print(f"  Rolling Model Dashboard — {model_type.upper()} Model")
    print("=" * 60)

    n_days = len(result['coeff_log'])
    n_feats = len(result['feature_names'])
    print(f"  交易日數: {n_days}  |  特徵數: {n_feats}")

    # 檢查是否有真正的篩選 (不是所有天都用全部特徵)
    all_full = all(
        len(names) == n_feats
        for names in result['screening_log'].values()
    )
    if all_full:
        print("  Feature Screening: OFF (所有天都使用全部特徵)")
    else:
        screened_days = sum(1 for names in result['screening_log'].values() if len(names) < n_feats)
        print(f"  Feature Screening: ON ({screened_days}/{n_days} 天有篩選)")
        print("\n📊 [1/4] Feature Selection Frequency")
        plot_feature_selection_frequency(result)
        print("\n📊 [2/4] Feature Selection Heatmap")
        plot_feature_selection_heatmap(result)

    print(f"\n📊 {'[3/4]' if not all_full else '[1/2]'} Standardized Coefficient Importance")
    plot_standardized_importance(result, model_type=model_type)

    print(f"\n📊 {'[4/4]' if not all_full else '[2/2]'} Coefficient Stability")
    plot_coefficient_stability(result, model_type=model_type)

    if 'pls_log' in result and result['pls_log']:
        print("\n📊 [5] PLS Component 1 Feature Weights")
        plot_pls_weights_heatmap(result, comp=1)
        
        # 如果你想看第二主成分，取消下面註解即可
        print("\n📊 [6] PLS Component 2 Feature Weights")
        plot_pls_weights_heatmap(result, comp=2)

