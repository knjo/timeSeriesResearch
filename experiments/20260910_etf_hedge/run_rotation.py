"""ETF 輪動避險：回看 L 日挑與策略最負相關的 ETF，下一週/月用它做多避險（不計手續費）。

- universe: 0050,0056,00878,00919,00981A,0052,00631L,00881,00662
- 回看 L ∈ {5,10,20} 交易日，rebalance ∈ {weekly, monthly}
- 排名效果：回看期依 corr 由負到正排 1..9，看下一期每個名次的實際 corr / 變異降幅是否單調
- 名目：前一區間平均策略部位 × (策略日報酬 std / ETF 開收 std)，整個區間固定
- 全域固定 0050 / 00631L 版本同樣 vol 還原
輸出 artifacts/rot_*.png / csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_etf_hedge_analysis as R  # noqa: E402

OUT = R.OUT
UNIVERSE = ["0050", "0056", "00878", "00919", "00981A", "0052", "00631L", "00881", "00662"]
LOOKBACKS = [5, 10, 20]
METRIC = "corr"          # corr | upcapture  (由 --metric 設定)
SUFFIX = ""
GATE_MIN_SCORE = None    # upcapture 門檻：分數 > 此值 (None = 不設)
GATE_MAX_CORR = None     # upcapture 門檻：corr < 此值 (None = 不設)；兩者都不過 → 該期不避險
TOP_N = 1                # 取前 N 名，各 1/N 名目
C = R.C
SERIES = {  # 固定順序色盤
    "策略": C["blue"], "避險腿(最互補ETF)": C["orange"], "策略+最互補ETF": C["aqua"],
    "策略+0050": C["yellow"], "策略+00631L": C["magenta"],
}


def load_panel(dates):
    market_dir = R.load_pipeline_storage(project_root=R.PROJECT_ROOT).market_dir
    cols = ["date", "quote_code", "open_price", "close_price"]
    fr = []
    for d in dates:
        p = market_dir / f"{d:%Y%m%d}_marketData.parquet"
        if not p.exists() or "open_price" not in pl.read_parquet_schema(p):
            continue
        fr.append(pl.read_parquet(p, columns=cols).filter(pl.col("quote_code").is_in(UNIVERSE)))
    df = pl.concat(fr).filter((pl.col("open_price") > 0) & (pl.col("close_price") > 0)).with_columns(
        ((pl.col("close_price") / pl.col("open_price") - 1) * 1e4).alias("oc_bp")).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="quote_code", values="oc_bp").reindex(dates)[UNIVERSE]


def periods(index, freq):
    key = index.to_period("W-FRI") if freq == "weekly" else index.to_period("M")
    return [(str(k), pd.DatetimeIndex(idx.values)) for k, idx in pd.Series(index, index=index).groupby(key)]


def score_upcapture(ys, e):
    """虧日漲 + 獲日跌 (bp)，乘上 策略vol/ETFvol 換算成與策略同波動名目。回傳 (score, volr)。"""
    j = pd.concat([ys, e], axis=1).dropna()
    lo, wi = j[j.iloc[:, 0] < 0], j[j.iloc[:, 0] >= 0]
    if len(lo) < 1 or len(wi) < 1:
        return np.nan, np.nan
    volr = j.iloc[:, 0].std() / j.iloc[:, 1].std()
    return (lo.iloc[:, 1].mean() + wi.iloc[:, 1].mean()) * volr, volr


def run(daily, panel, freq, L):
    """回傳 per-period 表 (ETF × 名次 × 下期 corr / hedge) 與每日避險腿損益 (最互補 ETF, 0050, 00631L)。"""
    idx = daily.index
    pos = daily["Position"]
    y = daily["ret_bp"]
    rows, legs = [], []
    for label, dts in periods(idx, freq):
        start = idx.get_loc(dts[0])
        if start < L or len(dts) < 3:
            continue
        lb = idx[start - L:start]
        base_pos = pos.loc[lb].mean()
        ys = y.loc[lb]
        lb_corr, lb_notional, lb_score = {}, {}, {}
        for code in UNIVERSE:
            e = panel.loc[lb, code]
            j = pd.concat([ys, e], axis=1).dropna()
            if len(j) < max(3, int(L * 0.8)):
                continue
            lb_corr[code] = j.iloc[:, 0].corr(j.iloc[:, 1])
            volr = j.iloc[:, 0].std() / j.iloc[:, 1].std()
            lb_notional[code] = {"vol": base_pos * volr, "beta": base_pos * volr * max(-lb_corr[code], 0.0)}
            if METRIC == "upcapture":
                sc, _ = score_upcapture(ys, e)
                if pd.isna(sc) or (ys < 0).sum() < 3 or (ys >= 0).sum() < 3:
                    del lb_corr[code], lb_notional[code]
                    continue
                lb_score[code] = sc
        if len(lb_corr) < 3:
            continue
        if METRIC == "upcapture":
            rank = (-pd.Series(lb_score)).rank(method="first")  # 1 = 分數最高
            sc_s = pd.Series(lb_score)
            if GATE_MIN_SCORE is not None:
                sc_s = sc_s[sc_s > GATE_MIN_SCORE]
            if GATE_MAX_CORR is not None:
                sc_s = sc_s[[c for c in sc_s.index if lb_corr[c] < GATE_MAX_CORR]]
            picks = list(sc_s.sort_values(ascending=False).index[:TOP_N]) if not sc_s.empty else []
        else:
            rank = pd.Series(lb_corr).rank(method="first")  # 1 = 最負
            picks = list(pd.Series(lb_corr).sort_values().index[:TOP_N])
        best = picks[0] if picks else None
        # 下一期表現
        yn = y.loc[dts]
        pn = pos.loc[dts]
        strat_pnl = daily.loc[dts, "PnL"]
        for code, c_lb in lb_corr.items():
            e = panel.loc[dts, code]
            j = pd.concat([yn, e], axis=1).dropna()
            if len(j) < 3:
                continue
            leg = lb_notional[code]["vol"] * e.fillna(0) / 1e4
            leg_b = lb_notional[code]["beta"] * e.fillna(0) / 1e4
            comb = strat_pnl + leg
            comb_b = strat_pnl + leg_b
            sc_next, _ = score_upcapture(yn, e)
            volr_lb = lb_notional[code]["vol"] / base_pos
            rows.append({
                "period": label, "start": dts[0], "L": L, "freq": freq, "code": code,
                "rank_lb": int(rank[code]), "corr_lb": c_lb,
                "score_lb": lb_score.get(code, np.nan),
                "score_next": sc_next / (j.iloc[:, 0].std() / j.iloc[:, 1].std()) * volr_lb if pd.notna(sc_next) else np.nan,
                "corr_next": j.iloc[:, 0].corr(j.iloc[:, 1]),
                "leg_pnl": leg.sum(), "strat_pnl": strat_pnl.sum(), "comb_pnl": comb.sum(),
                "std_ratio": comb.std() / strat_pnl.std() if strat_pnl.std() > 0 else np.nan,
                "std_ratio_beta": comb_b.std() / strat_pnl.std() if strat_pnl.std() > 0 else np.nan,
                "leg_bp": leg.sum() / pn.sum() * 1e4,
                "notional": lb_notional[code]["vol"], "notional_beta": lb_notional[code]["beta"], "is_best": code in picks,
            })
        if not picks:
            for sizing in ["vol", "beta"]:
                legs.append(pd.DataFrame({"Date": dts, "tag": "best", "code": None, "leg": 0.0, "notional": 0.0, "sizing": sizing}))
        else:
            for sizing in ["vol", "beta"]:
                leg = sum(lb_notional[c][sizing] / len(picks) * panel.loc[dts, c].fillna(0) for c in picks) / 1e4
                legs.append(pd.DataFrame({"Date": dts, "tag": "best", "code": "+".join(picks), "leg": leg.values,
                                          "notional": sum(lb_notional[c][sizing] for c in picks) / len(picks), "sizing": sizing}))
        for tag, code in [("0050", "0050"), ("00631L", "00631L")]:
            if code in lb_notional:
                for sizing in ["vol", "beta"]:
                    leg = lb_notional[code][sizing] * panel.loc[dts, code].fillna(0) / 1e4
                    legs.append(pd.DataFrame({"Date": dts, "tag": tag, "code": code, "leg": leg.values,
                                              "notional": lb_notional[code][sizing], "sizing": sizing}))
    return pd.DataFrame(rows), pd.concat(legs, ignore_index=True)


def rank_effect(tbl):
    """每名次的下期平均 corr / std_ratio，以及每期 Spearman(回看名次, 下期 corr)。"""
    g = tbl.groupby("rank_lb").agg(corr_next_mean=("corr_next", "mean"), corr_next_se=("corr_next", lambda s: s.std() / np.sqrt(len(s))),
                                   score_next_mean=("score_next", "mean"), score_next_se=("score_next", lambda s: s.std() / np.sqrt(max(s.count(), 1))),
                                   score_next_n=("score_next", "count"),
                                   std_ratio_mean=("std_ratio", "mean"), std_ratio_beta_mean=("std_ratio_beta", "mean"), n=("corr_next", "size"))
    tgt = "score_next" if METRIC == "upcapture" else "corr_next"
    sign = -1 if METRIC == "upcapture" else 1   # upcapture: 名次小 應對應 分數高 → 取負讓正值=有序
    sp = tbl.groupby("period").apply(lambda d: sign * d["rank_lb"].corr(d[tgt], method="spearman") if d[tgt].count() >= 4 else np.nan).dropna()
    return g, sp


def main():
    daily = R.build_daily_pnl()
    panel = load_panel(daily.index)
    print("[panel]", panel.notna().sum().to_dict())
    all_tbl, all_legs, effects = [], {}, {}
    for freq in ["weekly", "monthly"]:
        for L in LOOKBACKS:
            tbl, legs = run(daily, panel, freq, L)
            all_tbl.append(tbl)
            all_legs[(freq, L)] = legs
            g, sp = rank_effect(tbl)
            effects[(freq, L)] = (g, sp)
            print(f"\n== [{METRIC}] {freq} L={L}: {tbl.period.nunique()} 期 ==")
            print(g.round(3).to_string())
            print(f"  每期 Spearman(名次, 下期corr) mean={sp.mean():+.3f}, >0 比例={(sp > 0).mean():.2f}, n={len(sp)}")
            print("  最互補 ETF 出現次數:", tbl[tbl.is_best].code.value_counts().to_dict())
    tbl_all = pd.concat(all_tbl)
    tbl_all.to_csv(OUT / f"rot_period_table{SUFFIX}.csv", index=False)

    # ---------- 圖 1：名次效果 ----------
    fig, axes = plt.subplots(2, 6, figsize=(22, 8), sharex=True)
    for ci, (freq, L) in enumerate([(f, l) for f in ["weekly", "monthly"] for l in LOOKBACKS]):
        g, sp = effects[(freq, L)]
        ax = axes[0, ci]
        if METRIC == "upcapture":
            ax.bar(g.index, g["score_next_mean"], yerr=g["score_next_se"], color=C["blue"], width=0.7, capsize=2, error_kw={"ecolor": C["text2"], "lw": 1})
            ax.axhline(0, color=C["muted"], lw=0.8)
            ax.set_title(f"{freq}  回看 {L} 日\n下期『虧日漲+獲日跌』(bp, 同vol) 平均 (名次1=回看最高)", fontsize=10)
            ax.text(0.02, 0.96, f"Spearman(名次,下期分數) 有序=正\nmean {sp.mean():+.2f}, >0 佔 {(sp>0).mean():.0%}, n={len(sp)}",
                    transform=ax.transAxes, fontsize=8, color=C["text2"], va="top")
        else:
            ax.bar(g.index, g["corr_next_mean"], yerr=g["corr_next_se"], color=C["blue"], width=0.7, capsize=2, error_kw={"ecolor": C["text2"], "lw": 1})
            ax.axhline(0, color=C["muted"], lw=0.8)
            ax.set_title(f"{freq}  回看 {L} 日\n下期 corr 平均 (名次1=回看最負)", fontsize=10)
            ax.text(0.02, 0.04, f"Spearman(名次,下期corr)\nmean {sp.mean():+.2f}, >0 佔 {(sp>0).mean():.0%}, n={len(sp)}",
                    transform=ax.transAxes, fontsize=8, color=C["text2"], va="bottom")
            ax.set_ylim(-0.8, 0.3)
        ax = axes[1, ci]
        ax.bar(g.index - 0.2, g["std_ratio_mean"], color=C["orange"], width=0.38, label="名目=vol還原")
        ax.bar(g.index + 0.2, g["std_ratio_beta_mean"], color=C["aqua"], width=0.38, label="名目=vol還原×|corr| (beta)")
        ax.axhline(1, color=C["muted"], lw=0.8)
        ax.set_title("下期 避險後/未避險 日損益 std 比 (<1 才有降波動)", fontsize=10)
        if ci == 0:
            ax.legend(fontsize=8, loc="upper left")
        ax.set_ylim(0.4, 1.4)
        ax.set_xlabel("回看期名次 (1=最負相關)")
        ax.set_xticks(range(1, 10))
    ttl = "回看期挑『虧日漲多、獲日跌少』ETF → 下一期是否仍如此？" if METRIC == "upcapture" else "回看期挑最互補 ETF → 下一期是否仍最互補？"
    fig.suptitle(ttl + " (名次單調=有用, 平坦/亂=沒用)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / f"rot_fig1_rank_effect{SUFFIX}.png", dpi=130)
    plt.close(fig)

    # ---------- 圖 2/3：損益疊圖 ----------
    for freq, sizing in [(f, sz) for f in ["weekly", "monthly"] for sz in ["vol", "beta"]]:
        fig, axes = plt.subplots(len(LOOKBACKS), 2, figsize=(18, 4.2 * len(LOOKBACKS)), sharex=True,
                                 gridspec_kw={"width_ratios": [3, 1.3]})
        for ri, L in enumerate(LOOKBACKS):
            legs = all_legs[(freq, L)]
            legs = legs[legs.sizing == sizing]
            piv = legs.pivot(index="Date", columns="tag", values="leg").reindex(daily.index).fillna(0)
            first = legs["Date"].min()
            d = daily.loc[first:]
            piv = piv.loc[first:]
            cum = pd.DataFrame({
                "策略": d["PnL"].cumsum(),
                "避險腿(最互補ETF)": piv["best"].cumsum(),
                "策略+最互補ETF": (d["PnL"] + piv["best"]).cumsum(),
                "策略+0050": (d["PnL"] + piv["0050"]).cumsum(),
                "策略+00631L": (d["PnL"] + piv["00631L"]).cumsum(),
            }) / 1e4
            ax = axes[ri, 0]
            for k, col in SERIES.items():
                ax.plot(cum.index, cum[k], color=col, lw=2 if k in ("策略", "策略+最互補ETF") else 1.5,
                        ls="--" if k == "避險腿(最互補ETF)" else "-", label=k)
            ax.axhline(0, color=C["muted"], lw=0.8)
            ax.set_title(f"{freq} 換手, 回看 {L} 日：累積損益 (萬)", fontsize=11)
            ax.legend(loc="upper left", fontsize=9)
            # 換手期間標示最互補 ETF
            code_by_period = legs[legs.tag == "best"].groupby(legs[legs.tag == "best"]["Date"].dt.to_period("W-FRI" if freq == "weekly" else "M"))["code"].first()
            # 右欄：統計表
            ax = axes[ri, 1]
            ax.axis("off")
            stats = []
            for k in ["策略", "策略+最互補ETF", "策略+0050", "策略+00631L"]:
                s = cum[k].diff().fillna(cum[k].iloc[0])
                dd = (cum[k] - cum[k].cummax()).min()
                stats.append([k, f"{cum[k].iloc[-1]:,.0f}", f"{s.std():.0f}", f"{s.mean()/s.std()*np.sqrt(240):.2f}", f"{dd:,.0f}"])
            leg_s = cum["避險腿(最互補ETF)"].diff().fillna(0)
            stats.append(["避險腿(最互補)", f"{cum['避險腿(最互補ETF)'].iloc[-1]:,.0f}", f"{leg_s.std():.0f}", "", ""])
            t = ax.table(cellText=stats, colLabels=["", "總損益", "日std", "Sharpe", "MDD"], loc="center", cellLoc="right")
            t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1.1, 1.5)
            vc = legs[legs.tag == "best"].drop_duplicates(["code", "notional"]).code.value_counts()
            avg_not = legs[legs.tag == "best"].notional.mean() / 1e4
            ax.text(0.0, 0.12, f"最互補 ETF 平均名目 {avg_not:,.0f} 萬 (策略均部位 {d['Position'].mean()/1e4:,.0f} 萬)", transform=ax.transAxes, fontsize=8, color=C["text2"])
            ax.text(0.0, 0.02, "最互補 ETF 被選次數: " + ", ".join(f"{c}×{n}" for c, n in vc.items()),
                    transform=ax.transAxes, fontsize=8, color=C["text2"], wrap=True)
        for ax in axes[-1]:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        sz_txt = "前區間均部位×策略vol/ETFvol" if sizing == "vol" else "前區間均部位×策略vol/ETFvol×|corr| (=beta)"
        fig.suptitle(f"{freq} 換手 ETF 避險 (不計手續費；名目={sz_txt}，整區間固定)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / f"rot_fig2_{freq}_{sizing}_cum{SUFFIX}.png", dpi=130)
        plt.close(fig)

    # ---------- 圖 4：每週最互補 ETF 是誰 (L=20 weekly) + 下週 corr ----------
    tbl = tbl_all[(tbl_all.freq == "weekly") & (tbl_all.L == 20)]
    b = tbl[tbl.is_best].set_index("start").sort_index()
    fig, ax = plt.subplots(figsize=(18, 4))
    codes = UNIVERSE
    ymap = {c: i for i, c in enumerate(codes)}
    sc = ax.scatter(b.index, [ymap[c] for c in b.code], c=b["corr_next"], cmap=matplotlib.colors.LinearSegmentedColormap.from_list("d", [C["blue"], "#e5e5e3", C["orange"]]),
                    vmin=-1, vmax=1, s=90, edgecolor="white")
    ax.set_yticks(range(len(codes)), codes)
    ax.set_title(f"weekly, 回看 20 日 [{METRIC}]：每週被選的 ETF，顏色=該週實際 corr (藍=負=真的互補)")
    fig.colorbar(sc, ax=ax, fraction=0.02)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    fig.tight_layout(); fig.savefig(OUT / f"rot_fig3_weekly_pick{SUFFIX}.png", dpi=130); plt.close(fig)
    print("\n輸出:", OUT)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["corr", "upcapture"], default="corr")
    ap.add_argument("--lookbacks", type=int, nargs="+", default=None)
    a = ap.parse_args()
    METRIC = a.metric
    SUFFIX = "" if METRIC == "corr" else "_uc"
    if a.lookbacks:
        LOOKBACKS = a.lookbacks
    main()
