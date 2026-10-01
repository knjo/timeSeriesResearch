"""比較：前 1 名 vs 前 2 名各半；全宇宙 vs 移除高股息 (0056/00878/00919)。weekly 回看 20 日、upcapture、beta 名目。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_etf_hedge_analysis as R  # noqa: E402
import run_rotation as RO  # noqa: E402

C = R.C
FULL = ["0050", "0056", "00878", "00919", "00981A", "0052", "00631L", "00881", "00662"]
HIGH_DIV = ["0056", "00878", "00919"]
NO_HD = [c for c in FULL if c not in HIGH_DIV]
START = pd.Timestamp("2026-01-02")
CONFIGS = [("weekly", 20), ("weekly", 40), ("monthly", 20)]


def perf(p):
    c = p.cumsum()
    return {"總損益": p.sum() / 1e4, "日std": p.std() / 1e4, "Sharpe": p.mean() / p.std() * np.sqrt(240), "MDD": (c - c.cummax()).min() / 1e4,
            "最差日": p.min() / 1e4, "勝率": (p > 0).mean()}


def main():
    daily = R.build_daily_pnl()
    RO.METRIC = "upcapture"
    RO.UNIVERSE = FULL
    panel_full = RO.load_panel(daily.index)
    d = daily.loc[START:]
    rows, store = [], {}
    base = perf(d["PnL"]); base.update({"freq": "-", "L": "-", "宇宙": "-", "TopN": "-", "避險腿": 0, "被選": ""}); rows.append(base)
    for freq, L in CONFIGS:
        for uni_name, uni in [("全部 9 檔", FULL), ("移除高股息", NO_HD)]:
            RO.UNIVERSE = uni
            panel = panel_full[uni]
            for n in [1, 2]:
                RO.TOP_N = n
                _, legs = RO.run(daily, panel, freq, L)
                legs = legs[legs.sizing == "beta"]
                store[(freq, L, uni_name, n)] = legs
                piv = legs.pivot(index="Date", columns="tag", values="leg").reindex(d.index).fillna(0)
                b = legs[legs.tag == "best"]
                per = b.groupby(b.Date.dt.to_period("W-FRI" if freq == "weekly" else "M"))["code"].first()
                per = per[per.index >= START.to_period("W-FRI" if freq == "weekly" else "M")]
                cnt = pd.Series([c for v in per.dropna() for c in v.split("+")]).value_counts()
                r = perf(d["PnL"] + piv["best"]); r.update({"freq": freq, "L": L, "宇宙": uni_name, "TopN": n, "避險腿": piv["best"].sum() / 1e4,
                                                          "被選": ", ".join(f"{k}×{v}" for k, v in cnt.items())}); rows.append(r)
                if n == 1 and uni_name == "全部 9 檔":
                    r0 = perf(d["PnL"] + piv["0050"]); r0.update({"freq": freq, "L": L, "宇宙": "固定 0050", "TopN": "-", "避險腿": piv["0050"].sum() / 1e4, "被選": "0050"}); rows.append(r0)
    res = pd.DataFrame(rows)[["freq", "L", "宇宙", "TopN", "總損益", "日std", "Sharpe", "MDD", "最差日", "勝率", "避險腿", "被選"]]
    res.to_csv(R.OUT / "topn_universe_compare.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    print(f"評估期 {START.date()} ~ {d.index.max().date()}，upcapture 選股，beta 名目，不計手續費")
    print(res.round(2).to_string(index=False))

    # 圖：weekly 20 四個變體 + 未避險 + 0050
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    series = {"未避險": (d["PnL"], C["blue"], 2.2)}
    for (uni_name, n), col in [(("全部 9 檔", 1), C["orange"]), (("全部 9 檔", 2), C["yellow"]), (("移除高股息", 1), C["green"]), (("移除高股息", 2), C["violet"])]:
        piv = store[("weekly", 20, uni_name, n)].pivot(index="Date", columns="tag", values="leg").reindex(d.index).fillna(0)
        series[f"{uni_name} 前{n}名" + ("各半" if n == 2 else "")] = (d["PnL"] + piv["best"], col, 1.6)
    piv = store[("weekly", 20, "全部 9 檔", 1)].pivot(index="Date", columns="tag", values="leg").reindex(d.index).fillna(0)
    series["固定 0050"] = (d["PnL"] + piv["0050"], C["aqua"], 1.6)
    for k, (p, col, lw) in series.items():
        c = p.cumsum() / 1e4
        axes[0].plot(c.index, c, color=col, lw=lw, label=k)
        axes[1].plot(c.index, c - c.cummax(), color=col, lw=1.3)
    axes[0].axhline(0, color=C["muted"], lw=0.8); axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_title("weekly 回看 20 日：前 1 名 vs 前 2 名各半、全宇宙 vs 移除高股息 (累積損益, 萬；beta 名目；不計手續費)")
    axes[1].set_title("回撤 (萬)"); axes[1].axhline(0, color=C["muted"], lw=0.8)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator()); axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    fig.tight_layout(); fig.savefig(R.OUT / "topn_fig1_weekly20.png", dpi=130); plt.close(fig)
    print("輸出:", R.OUT)


if __name__ == "__main__":
    main()
