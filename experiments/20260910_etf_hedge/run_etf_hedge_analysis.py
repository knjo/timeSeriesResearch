"""ETF 開收盤報酬 vs timeSeries Taker 策略日損益：相關性與相似度自相關分析。

輸入
  src/research/intradaySignal/signalwithRisk_df.parquet  (下單點, Taker 模式)
  <pipeline market_dir>/YYYYMMDD_marketData.parquet  (ETF 開/收/成交金額)

步驤
  1. 以 BackTest (Taker) 算每日 PnL / Position，ret_bp = PnL / Position * 1e4
  2. ETF universe：core = 期間平均日成交額 > 50 億；extended = >=5% 交易日 > 50 億
     計算 ETF 開->收報酬與策略日報酬的相關、beta(避險比率)
  3. 相似度是否有自相關：
     a. 非重疊 20 日區塊相關 -> AR(1)、排名持續性 (Spearman)
     b. 20 日滾動相關的 ACF (lag >= 20 才是非重疊)
     c. 樣本外避險：用前一區塊挑 ETF + 估 beta，套到本區塊
        另做每日滾動版 (前 N 日估 beta、前 5 日平均部位換算名目)
所有輸出寫到同資料夾 artifacts/。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_storage import load_pipeline_storage  # noqa: E402
from src.research.backTest import BackTest  # noqa: E402

SIGNAL_PATH = PROJECT_ROOT / "src/research/intradaySignal/signalwithRisk_df.parquet"
OUT = HERE / "artifacts"
OUT.mkdir(exist_ok=True)

FEE_BP = 19.3
CORE_TURNOVER = 5e9            # 50 億
EXT_FRAC_GT50 = 0.05           # 至少 5% 交易日 > 50 億
MIN_DAYS = 60
BLOCK = 20                     # 非重疊區塊長度 (交易日)
ROLL = 20                      # 滾動相關視窗
ROLL_BETA = 40                 # 每日滾動避險估 beta 的回看天數
POS_LOOKBACK = 5               # 避險名目用前 N 日平均部位
ETF_COST_BP = 14.0             # ETF 來回成本：證交稅 0.1% + 手續費 2*0.02%
COST_GRID = (0.0, 7.0, 14.0)   # 成本敏感度

# 色盤 (dataviz reference palette, light)
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
    "grid": "#e5e5e3", "text": "#0b0b0b", "text2": "#52514e", "muted": "#9a9993",
}
import matplotlib.font_manager as _fm
for _f in _fm.findSystemFonts():
    if "NotoSansCJK" in _f and ("Regular" in _f or "TC" in _f):
        _fm.fontManager.addfont(_f)
plt.rcParams.update({
    "font.family": ["Noto Sans CJK TC", "Noto Sans CJK JP", "DejaVu Sans"],
    "axes.edgecolor": C["grid"], "axes.labelcolor": C["text2"], "xtick.color": C["text2"],
    "ytick.color": C["text2"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "axes.titlecolor": C["text"], "axes.titleweight": "bold", "legend.frameon": False,
})


# --------------------------------------------------------------------------- #
# 1. 策略每日損益
# --------------------------------------------------------------------------- #
def build_daily_pnl() -> pd.DataFrame:
    sig = pd.read_parquet(SIGNAL_PATH)
    bt = BackTest(sig, fee=FEE_BP)
    bt.run_analysis(return_daily=True, plot=False)
    daily = bt.get_daily_result()[["Date", "PnL", "Position", "CumPnL", "Drawdown"]].copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["ret_bp"] = daily["PnL"] / daily["Position"] * 1e4
    daily["n_signals"] = sig.groupby(pd.to_datetime(sig["Date"].astype(str))).size().reindex(daily["Date"]).values
    return daily.set_index("Date")


# --------------------------------------------------------------------------- #
# 2. ETF 面板
# --------------------------------------------------------------------------- #
def load_etf_panel(dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_dir = load_pipeline_storage(project_root=PROJECT_ROOT).market_dir
    cols = ["date", "quote_code", "ins_type", "stock_name", "market",
            "open_price", "close_price", "trading_turnover"]
    frames, skipped = [], []
    for d in dates:
        p = market_dir / f"{d:%Y%m%d}_marketData.parquet"
        if not p.exists() or not set(cols) <= set(pl.read_parquet_schema(p)):
            skipped.append(d.strftime("%Y%m%d"))
            continue
        frames.append(pl.read_parquet(p, columns=cols).filter(pl.col("ins_type") == "etf"))
    if skipped:
        print(f"[etf] marketData 缺 OHLC / 不存在，跳過 {len(skipped)} 天: {skipped}")
    etf = pl.concat(frames).filter(
        (pl.col("open_price") > 0) & (pl.col("close_price") > 0)
    ).with_columns(
        ((pl.col("close_price") / pl.col("open_price") - 1) * 1e4).alias("oc_bp")
    ).to_pandas()
    etf["date"] = pd.to_datetime(etf["date"])

    stats = etf.groupby("quote_code").agg(
        name=("stock_name", "last"), market=("market", "last"),
        mean_turnover=("trading_turnover", "mean"), median_turnover=("trading_turnover", "median"),
        frac_gt50=("trading_turnover", lambda s: (s > CORE_TURNOVER).mean()),
        n_days=("trading_turnover", "size"),
    )
    stats["tier"] = np.where(
        stats["mean_turnover"] > CORE_TURNOVER, "core",
        np.where((stats["frac_gt50"] >= EXT_FRAC_GT50) & (stats["n_days"] >= MIN_DAYS), "extended", ""),
    )
    stats = stats[stats["tier"] != ""].sort_values("mean_turnover", ascending=False)
    panel = etf[etf["quote_code"].isin(stats.index)].pivot(index="date", columns="quote_code", values="oc_bp")
    return stats, panel


# --------------------------------------------------------------------------- #
# 3. 相關 / beta
# --------------------------------------------------------------------------- #
def corr_table(daily: pd.DataFrame, panel: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code in panel.columns:
        j = pd.concat([daily[["ret_bp", "PnL"]], panel[code].rename("etf")], axis=1).dropna()
        if len(j) < 20:
            continue
        x, y = j["etf"].values, j["ret_bp"].values
        beta, alpha = np.polyfit(x, y, 1)
        resid = y - (alpha + beta * x)
        se = np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum())
        hedged = y - beta * x                      # 完全對沖 beta 後的殘差報酬
        rows.append({
            "code": code, "name": stats.loc[code, "name"], "tier": stats.loc[code, "tier"],
            "n": len(j),
            "corr_retbp": np.corrcoef(x, y)[0, 1],
            "corr_pnl": np.corrcoef(x, j["PnL"].values)[0, 1],
            "spearman": pd.Series(x).corr(pd.Series(y), method="spearman"),
            "beta": beta, "t_beta": beta / se, "r2": 1 - resid.var() / y.var(),
            "hedge_notional_ratio": -beta,         # 每 1 元空方部位要買多少元 ETF (正=做多 ETF)
            "std_unhedged_bp": y.std(), "std_hedged_bp": hedged.std(),
            "var_reduction": 1 - hedged.var() / y.var(),
            "etf_daily_std_bp": x.std(),
        })
    return pd.DataFrame(rows).set_index("code").sort_values("corr_retbp")


# --------------------------------------------------------------------------- #
# 4. 相似度自相關
# --------------------------------------------------------------------------- #
def block_corr(daily: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = daily.index
    block_id = pd.Series(np.arange(len(idx)) // BLOCK, index=idx)
    # 最後一個不足 BLOCK/2 的區塊併入前一塊
    last = block_id.max()
    if (block_id == last).sum() < BLOCK // 2 and last > 0:
        block_id[block_id == last] = last - 1
    corrs, betas = {}, {}
    for b, dts in block_id.groupby(block_id).groups.items():
        y = daily.loc[dts, "ret_bp"]
        sub = panel.reindex(dts)
        c_row, b_row = {}, {}
        for code in panel.columns:
            j = pd.concat([y, sub[code]], axis=1).dropna()
            if len(j) >= BLOCK * 0.75:
                c_row[code] = j.iloc[:, 0].corr(j.iloc[:, 1])
                b_row[code] = np.polyfit(j.iloc[:, 1].values, j.iloc[:, 0].values, 1)[0]
        label = f"{dts.min():%y%m%d}-{dts.max():%m%d}"
        corrs[label] = c_row
        betas[label] = b_row
    return pd.DataFrame(corrs).T, pd.DataFrame(betas).T, block_id


def similarity_autocorr(bc: pd.DataFrame, bb: pd.DataFrame | None = None) -> dict:
    out = {}
    # 每檔 ETF 區塊相關的 AR(1)
    per = {}
    for code in bc.columns:
        s = bc[code].dropna()
        per[code] = s.autocorr(1) if len(s) >= 5 else np.nan
    out["per_etf_ar1"] = pd.Series(per)
    # pooled：原始 (含 ETF 間水準差) vs 去除各 ETF 平均 (只看時間上的起伏)
    stacked = pd.concat([bc.shift(1).stack(), bc.stack()], axis=1, keys=["prev", "cur"]).dropna()
    out["pooled_ar1_raw"] = stacked["prev"].corr(stacked["cur"])
    dm = bc - bc.mean()
    stacked_dm = pd.concat([dm.shift(1).stack(), dm.stack()], axis=1, keys=["prev", "cur"]).dropna()
    out["pooled_ar1_demeaned"] = stacked_dm["prev"].corr(stacked_dm["cur"])
    out["pooled_n"] = len(stacked)
    if bb is not None:
        st_b = pd.concat([bb.shift(1).stack(), bb.stack()], axis=1, keys=["prev", "cur"]).dropna()
        out["beta_pooled_ar1_raw"] = st_b["prev"].corr(st_b["cur"])
        dmb = bb - bb.mean()
        st_bd = pd.concat([dmb.shift(1).stack(), dmb.stack()], axis=1, keys=["prev", "cur"]).dropna()
        out["beta_pooled_ar1_demeaned"] = st_bd["prev"].corr(st_bd["cur"])
    # 排名持續性：第 t 塊 ETF 排名 vs 第 t+1 塊
    ranks = []
    for i in range(1, len(bc)):
        a, b = bc.iloc[i - 1], bc.iloc[i]
        j = pd.concat([a, b], axis=1).dropna()
        if len(j) >= 4:
            ranks.append(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman"))
    out["rank_persistence"] = pd.Series(ranks, index=bc.index[1:len(ranks) + 1])
    # 最佳 ETF (最負相關) 是否延續
    best = bc.idxmin(axis=1)
    out["best_etf_by_block"] = best
    out["best_same_as_prev"] = (best == best.shift(1)).iloc[1:].mean()
    return out


def rolling_corr(daily: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    rc = {}
    for code in panel.columns:
        j = pd.concat([daily["ret_bp"], panel[code]], axis=1)
        rc[code] = j["ret_bp"].rolling(ROLL, min_periods=int(ROLL * 0.75)).corr(j[code])
    return pd.DataFrame(rc)


def rolling_acf(rc: pd.DataFrame, lags=(1, 5, 10, 20, 40, 60)) -> pd.DataFrame:
    return pd.DataFrame({code: {lag: rc[code].autocorr(lag) for lag in lags} for code in rc.columns}).T


# --------------------------------------------------------------------------- #
# 5. 樣本外避險
# --------------------------------------------------------------------------- #
def hedge_pnl_series(daily, panel, code: pd.Series, ratio: pd.Series, notional_base: pd.Series,
                     cost_bp: float = ETF_COST_BP) -> pd.DataFrame:
    """code/ratio/notional_base 都是以 Date 為 index 的每日決策 (已經是事前資訊)。"""
    rows = []
    for d in daily.index:
        c, h, base = code.get(d), ratio.get(d), notional_base.get(d)
        hp, used, rows_notional = 0.0, None, 0.0
        if isinstance(c, str) and pd.notna(h) and pd.notna(base) and h > 0 and c in panel.columns:
            r = panel.at[d, c] if d in panel.index else np.nan
            if pd.notna(r):
                notional = h * base
                hp = notional * (r - cost_bp) / 1e4
                used = c
                rows_notional = notional
        rows.append({"Date": d, "hedge_pnl": hp, "hedge_code": used, "hedge_notional": rows_notional})
    h = pd.DataFrame(rows).set_index("Date")
    h["pnl_unhedged"] = daily["PnL"]
    h["pnl_hedged"] = daily["PnL"] + h["hedge_pnl"]
    return h


def perf(pnl: pd.Series) -> dict:
    cum = pnl.cumsum()
    dd = cum - cum.cummax()
    return {
        "total_pnl_wan": pnl.sum() / 1e4, "daily_std_wan": pnl.std() / 1e4,
        "sharpe_ann": pnl.mean() / pnl.std() * np.sqrt(240) if pnl.std() > 0 else np.nan,
        "mdd_wan": dd.min() / 1e4, "worst_day_wan": pnl.min() / 1e4,
        "win_rate": (pnl > 0).mean(),
    }


def oos_block_hedge(daily, panel, bc, bb, block_id, stats) -> tuple[pd.DataFrame, pd.DataFrame]:
    """前一區塊挑最負相關 ETF + 該區塊 beta，套到本區塊。名目 = 前 5 日平均部位。"""
    base = daily["Position"].shift(1).rolling(POS_LOOKBACK, min_periods=POS_LOOKBACK).mean()
    labels = list(bc.index)
    code_s, ratio_s, code_0050, ratio_0050, code_core, ratio_core = {}, {}, {}, {}, {}, {}
    core_codes = [c for c in stats.index[stats["tier"] == "core"] if c in bc.columns]
    for i, lab in enumerate(labels):
        dts = block_id.index[block_id == i]
        if i == 0:
            continue
        prev_c, prev_b = bc.iloc[i - 1].dropna(), bb.iloc[i - 1]
        if prev_c.empty:
            continue
        best = prev_c.idxmin()
        best_core = prev_c.reindex(core_codes).dropna().idxmin() if core_codes else None
        for d in dts:
            code_s[d], ratio_s[d] = best, -prev_b[best]
            if "0050" in prev_b:
                code_0050[d], ratio_0050[d] = "0050", -prev_b["0050"]
            if best_core is not None:
                code_core[d], ratio_core[d] = best_core, -prev_b[best_core]
    decisions = {
        "prev_block_best_any": (pd.Series(code_s), pd.Series(ratio_s)),
        "prev_block_best_core": (pd.Series(code_core), pd.Series(ratio_core)),
        "prev_block_0050": (pd.Series(code_0050), pd.Series(ratio_0050)),
    }
    # 每日滾動：前 ROLL_BETA 日挑 ETF + 估 beta
    code_r, ratio_r, code_r50, ratio_r50 = {}, {}, {}, {}
    y = daily["ret_bp"]
    for i, d in enumerate(daily.index):
        if i < ROLL_BETA:
            continue
        win = daily.index[i - ROLL_BETA:i]
        yy = y.loc[win]
        best, best_c, b50 = None, 0.0, np.nan
        for code in panel.columns:
            j = pd.concat([yy, panel[code].reindex(win)], axis=1).dropna()
            if len(j) < ROLL_BETA * 0.75:
                continue
            c = j.iloc[:, 0].corr(j.iloc[:, 1])
            b = np.polyfit(j.iloc[:, 1].values, j.iloc[:, 0].values, 1)[0]
            if code == "0050":
                b50 = b
            if c < best_c:
                best, best_c, bbest = code, c, b
        if best is not None:
            code_r[d], ratio_r[d] = best, -bbest
        if pd.notna(b50):
            code_r50[d], ratio_r50[d] = "0050", -b50
    decisions[f"rolling{ROLL_BETA}_best_any"] = (pd.Series(code_r), pd.Series(ratio_r))
    decisions[f"rolling{ROLL_BETA}_0050"] = (pd.Series(code_r50), pd.Series(ratio_r50))
    # 拆解：OOS 選股/beta 不變，但名目改用當日真實部位 (≈ 盤中隨部位加減碼的理想版)
    decisions[f"rolling{ROLL_BETA}_best_any_sameday_pos"] = (pd.Series(code_r), pd.Series(ratio_r))
    decisions[f"rolling{ROLL_BETA}_0050_sameday_pos"] = (pd.Series(code_r50), pd.Series(ratio_r50))
    # 上界對照：全期 in-sample 0050 beta + 同日真實部位 (不可執行，只看相關性的天花板)
    beta_is = np.polyfit(*pd.concat([panel["0050"], daily["ret_bp"]], axis=1).dropna().values.T, 1)[0]
    decisions["insample_0050_sameday_pos"] = (
        pd.Series("0050", index=daily.index), pd.Series(-beta_is, index=daily.index))

    eval_start = max(block_id.index[block_id == 1].min(), daily.index[ROLL_BETA])
    ev = daily.index[daily.index >= eval_start]
    summaries, merged = [], pd.DataFrame({"pnl_unhedged": daily["PnL"]})
    for cost in COST_GRID:
        summary = {"unhedged": perf(daily.loc[ev, "PnL"])}
        for k, (cd, rt) in decisions.items():
            base_k = daily["Position"] if ("sameday_pos" in k) else base
            v = hedge_pnl_series(daily, panel, cd, rt, base_k, cost_bp=cost)
            summary[k] = perf(v.loc[ev, "pnl_hedged"])
            summary[k]["hedge_days"] = v.loc[ev, "hedge_code"].notna().sum()
            summary[k]["hedge_pnl_wan"] = v.loc[ev, "hedge_pnl"].sum() / 1e4
            summary[k]["avg_hedge_notional_wan"] = v.loc[ev, "hedge_notional"].mean() / 1e4
            if cost == ETF_COST_BP:
                merged[f"pnl_{k}"] = v["pnl_hedged"]
                merged[f"code_{k}"] = v["hedge_code"]
        df_s = pd.DataFrame(summary).T
        df_s.insert(0, "cost_bp", cost)
        summaries.append(df_s)
    summary = pd.concat(summaries)
    return summary, merged.loc[ev]


# --------------------------------------------------------------------------- #
# 圖
# --------------------------------------------------------------------------- #
def plot_daily(daily: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(daily.index, daily["CumPnL"] / 1e4, color=C["blue"], lw=2)
    ax.set_title("timeSeries Taker 策略累積損益 (萬)")
    ax.set_ylabel("萬元")
    ax = axes[1]
    colors = np.where(daily["PnL"] >= 0, C["blue"], C["orange"])
    ax.bar(daily.index, daily["PnL"] / 1e4, color=colors, width=1.0)
    ax.set_title("每日損益 (萬)  藍=正 / 橘=負")
    ax.axhline(0, color=C["muted"], lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_daily_pnl.png", dpi=130)
    plt.close(fig)


def plot_corr_bar(ct: pd.DataFrame):
    ct = ct.sort_values("corr_retbp")
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(ct) + 2))
    colors = [C["blue"] if t == "core" else C["muted"] for t in ct["tier"]]
    ax.barh(ct.index + "  " + ct["name"].str.slice(0, 10), ct["corr_retbp"], color=colors, height=0.7)
    ax.axvline(0, color=C["text2"], lw=0.8)
    for i, (v, t) in enumerate(zip(ct["corr_retbp"], ct["t_beta"])):
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:+.2f} (t={t:+.1f})", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=C["text2"])
    ax.set_xlabel("Pearson corr (策略日報酬 bp vs ETF 開→收 bp)")
    ax.set_title("ETF 開收報酬 vs 策略日報酬相關  藍=core(均量>50億) 灰=extended")
    ax.set_xlim(min(ct["corr_retbp"].min() - 0.15, -0.2), max(ct["corr_retbp"].max() + 0.15, 0.2))
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "fig2_corr_by_etf.png", dpi=130)
    plt.close(fig)


def plot_rolling(rc: pd.DataFrame, ct: pd.DataFrame, daily):
    codes = list(ct.sort_values("corr_retbp").index)
    n = len(codes)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 2.6 * nrow), sharex=True, sharey=True)
    for ax, code in zip(axes.flat, codes):
        s = rc[code]
        ax.plot(s.index, s, color=C["blue"] if ct.loc[code, "tier"] == "core" else C["text2"], lw=1.6)
        ax.axhline(0, color=C["muted"], lw=0.8)
        ax.axhline(ct.loc[code, "corr_retbp"], color=C["orange"], lw=1, ls="--")
        ax.set_title(f"{code} {ct.loc[code, 'name'][:10]}  全期 {ct.loc[code, 'corr_retbp']:+.2f}", fontsize=10)
        ax.set_ylim(-1, 1)
    for ax in axes.flat[n:]:
        ax.axis("off")
    import matplotlib.dates as mdates
    for ax in axes.flat:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    for col in range(ncol):
        visible = [axes[r, col] for r in range(nrow) if r * ncol + col < n]
        if visible:
            visible[-1].tick_params(labelbottom=True)
    fig.suptitle(f"{ROLL} 日滾動相關 (橘虛線=全期相關)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_rolling_corr.png", dpi=130)
    plt.close(fig)


def plot_block_heatmap(bc: pd.DataFrame, ct: pd.DataFrame):
    order = list(ct.sort_values("corr_retbp").index)
    m = bc[order]
    cmap = LinearSegmentedColormap.from_list("div", [C["blue"], "#e5e5e3", C["orange"]])
    fig, ax = plt.subplots(figsize=(0.75 * len(order) + 3, 0.5 * len(m) + 2))
    im = ax.imshow(m.values, cmap=cmap, vmin=-0.8, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(m)), m.index, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7, color=C["text"])
    ax.grid(False)
    ax.set_title(f"非重疊 {BLOCK} 日區塊相關 (藍=負相關=可做多避險)")
    fig.colorbar(im, ax=ax, fraction=0.03, label="corr")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_block_corr_heatmap.png", dpi=130)
    plt.close(fig)


def plot_oos(merged: pd.DataFrame):
    series = {
        "未避險": ("pnl_unhedged", C["blue"]),
        "前區塊最佳ETF": ("pnl_prev_block_best_any", C["orange"]),
        "前區塊 0050": ("pnl_prev_block_0050", C["aqua"]),
        f"滾動{ROLL_BETA}日最佳ETF": (f"pnl_rolling{ROLL_BETA}_best_any", C["yellow"]),
        f"滾動{ROLL_BETA}日 0050 + 當日部位": (f"pnl_rolling{ROLL_BETA}_0050_sameday_pos", C["magenta"]),
    }
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for label, (col, color) in series.items():
        cum = merged[col].cumsum() / 1e4
        axes[0].plot(cum.index, cum, color=color, lw=2, label=label)
        dd = (cum - cum.cummax())
        axes[1].plot(dd.index, dd, color=color, lw=1.6, label=label)
    axes[0].set_title(f"樣本外避險：累積損益 (萬)，ETF 來回成本 {ETF_COST_BP:.0f} bp")
    axes[0].legend(loc="upper left")
    axes[1].set_title("回撤 (萬)")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_oos_hedge.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    daily = build_daily_pnl()
    daily.to_csv(OUT / "strategy_daily_pnl.csv")
    print(f"[daily] {len(daily)} 天, 總損益 {daily['PnL'].sum()/1e4:,.0f} 萬, 平均部位 {daily['Position'].mean()/1e4:,.0f} 萬, "
          f"日報酬 {daily['ret_bp'].mean():.1f} ± {daily['ret_bp'].std():.1f} bp")

    stats, panel = load_etf_panel(daily.index)
    stats.to_csv(OUT / "etf_universe.csv")
    print(f"[etf] core {int((stats.tier=='core').sum())} 檔, extended {int((stats.tier=='extended').sum())} 檔")
    print(stats[["name", "mean_turnover", "frac_gt50", "n_days", "tier"]].to_string())

    ct = corr_table(daily, panel, stats)
    ct.to_csv(OUT / "etf_correlation.csv")
    print("\n[corr] 全期相關 (依 corr_retbp 排序)")
    print(ct[["name", "tier", "n", "corr_retbp", "corr_pnl", "spearman", "beta", "t_beta", "r2",
              "hedge_notional_ratio", "var_reduction"]].round(3).to_string())

    bc, bb, block_id = block_corr(daily, panel)
    bc.to_csv(OUT / "block_corr.csv")
    bb.to_csv(OUT / "block_beta.csv")
    ac = similarity_autocorr(bc, bb)
    rc = rolling_corr(daily, panel)
    rc.to_csv(OUT / "rolling_corr.csv")
    acf = rolling_acf(rc)
    acf.to_csv(OUT / "rolling_corr_acf.csv")

    print(f"\n[block corr] {len(bc)} 個 {BLOCK} 日區塊")
    print(bc.round(2).to_string())
    print(f"\n[block beta] (ret_bp 對 ETF oc_bp 斜率；避險比率 = -beta)")
    print(bb.round(2).to_string())
    print("\n[similarity autocorr]")
    print(f"  pooled AR(1) raw (含 ETF 間水準差)      : {ac['pooled_ar1_raw']:+.3f}  (n={ac['pooled_n']})")
    print(f"  pooled AR(1) demeaned (只看時間起伏)     : {ac['pooled_ar1_demeaned']:+.3f}")
    print(f"  beta pooled AR(1) raw / demeaned         : {ac['beta_pooled_ar1_raw']:+.3f} / {ac['beta_pooled_ar1_demeaned']:+.3f}")
    print(f"  區塊間 ETF 排名持續性 Spearman (mean)    : {ac['rank_persistence'].mean():+.3f}")
    print("  " + ac["rank_persistence"].round(2).to_string().replace("\n", "\n  "))
    print(f"  最佳 ETF 與前一區塊相同的比例            : {ac['best_same_as_prev']:.2f}")
    print("  各區塊最佳 (最負相關) ETF: " + ", ".join(f"{k}:{v}" for k, v in ac["best_etf_by_block"].items()))
    print("  各 ETF 區塊相關 AR(1):")
    print("  " + ac["per_etf_ar1"].round(2).to_string().replace("\n", "\n  "))
    print(f"\n[rolling{ROLL} corr ACF] (lag>= {ROLL} 才是非重疊)")
    print(acf.round(2).to_string())

    summary, merged = oos_block_hedge(daily, panel, bc, bb, block_id, stats)
    summary.to_csv(OUT / "oos_hedge_summary.csv")
    merged.to_csv(OUT / "oos_hedge_daily.csv")
    print(f"\n[OOS hedge] 評估期 {merged.index.min():%Y-%m-%d} ~ {merged.index.max():%Y-%m-%d}")
    for cost, grp in summary.groupby("cost_bp"):
        print(f"  -- ETF 來回成本 {cost} bp --")
        print("  " + grp.drop(columns="cost_bp").round(2).to_string().replace("\n", "\n  "))
    for k in [c for c in merged.columns if c.startswith("code_")]:
        vc = merged[k].value_counts()
        print(f"  {k}: " + ", ".join(f"{i}={v}" for i, v in vc.items()))

    plot_daily(daily)
    plot_corr_bar(ct)
    plot_rolling(rc, ct, daily)
    plot_block_heatmap(bc, ct)
    plot_oos(merged)
    pd.Series({
        "pooled_ar1_raw": ac["pooled_ar1_raw"], "pooled_ar1_demeaned": ac["pooled_ar1_demeaned"],
        "rank_persistence_mean": ac["rank_persistence"].mean(), "best_same_as_prev": ac["best_same_as_prev"],
        "beta_pooled_ar1_raw": ac["beta_pooled_ar1_raw"], "beta_pooled_ar1_demeaned": ac["beta_pooled_ar1_demeaned"],
    }).to_csv(OUT / "similarity_autocorr_summary.csv")
    print(f"\n輸出: {OUT}")


if __name__ == "__main__":
    main()
