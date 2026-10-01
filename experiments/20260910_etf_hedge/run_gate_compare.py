"""比較 upcapture 選股加門檻 (分數 > 0 且 corr < −0.3) 前後的避險效果。"""
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
GATES = {"無門檻": (None, None), "分數>0": (0.0, None), "corr<-0.3": (None, -0.3), "分數>0 且 corr<-0.3": (0.0, -0.3)}
CONFIGS = [("weekly", 10), ("weekly", 20), ("weekly", 40), ("monthly", 20)]


def perf(p):
    c = p.cumsum()
    return {"總損益": p.sum() / 1e4, "日std": p.std() / 1e4, "Sharpe": p.mean() / p.std() * np.sqrt(240), "MDD": (c - c.cummax()).min() / 1e4,
            "最差日": p.min() / 1e4, "勝率": (p > 0).mean()}


def main():
    daily = R.build_daily_pnl()
    panel = RO.load_panel(daily.index)
    RO.METRIC = "upcapture"
    results, legs_store = [], {}
    for freq, L in CONFIGS:
        for gname, (ms, mc) in GATES.items():
            RO.GATE_MIN_SCORE, RO.GATE_MAX_CORR = ms, mc
            _, legs = RO.run(daily, panel, freq, L)
            legs = legs[legs.sizing == "beta"]
            legs_store[(freq, L, gname)] = legs
    start = pd.Timestamp("2026-01-02")
    d = daily.loc[start:]
    base = perf(d["PnL"]); base.update({"freq": "-", "L": "-", "gate": "未避險", "避險腿": 0, "無避險週數": 0, "被選": ""})
    results.append(base)
    for (freq, L, gname), legs in legs_store.items():
        piv = legs.pivot(index="Date", columns="tag", values="leg").reindex(d.index).fillna(0)
        b = legs[legs.tag == "best"]
        wk = b.groupby(b.Date.dt.to_period("W-FRI" if freq == "weekly" else "M"))["code"].first()
        wk = wk[wk.index >= start.to_period("W-FRI" if freq == "weekly" else "M")]
        r = perf(d["PnL"] + piv["best"]); r.update({"freq": freq, "L": L, "gate": gname, "避險腿": piv["best"].sum() / 1e4,
                                                  "無避險週數": int(wk.isna().sum()), "被選": ", ".join(f"{k}×{v}" for k, v in wk.value_counts().items())})
        results.append(r)
        if gname == "無門檻":
            r0 = perf(d["PnL"] + piv["0050"]); r0.update({"freq": freq, "L": L, "gate": "固定 0050", "避險腿": piv["0050"].sum() / 1e4, "無避險週數": 0, "被選": "0050"})
            results.append(r0)
    res = pd.DataFrame(results)[["freq", "L", "gate", "總損益", "日std", "Sharpe", "MDD", "最差日", "勝率", "避險腿", "無避險週數", "被選"]]
    res.to_csv(R.OUT / "gate_compare.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print(f"評估期 {start.date()} ~ {d.index.max().date()}，beta 名目，不計手續費")
    print(res.round(2).to_string(index=False))

    # 圖：weekly L=20 疊圖
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    series = {"未避險": (d["PnL"], C["blue"])}
    for gname, col in [("無門檻", C["orange"]), ("分數>0 且 corr<-0.3", C["green"]), ("corr<-0.3", C["yellow"])]:
        piv = legs_store[("weekly", 20, gname)].pivot(index="Date", columns="tag", values="leg").reindex(d.index).fillna(0)
        series[f"最互補 ETF ({gname})"] = (d["PnL"] + piv["best"], col)
    piv = legs_store[("weekly", 20, "無門檻")].pivot(index="Date", columns="tag", values="leg").reindex(d.index).fillna(0)
    series["固定 0050"] = (d["PnL"] + piv["0050"], C["aqua"])
    for k, (p, col) in series.items():
        c = p.cumsum() / 1e4
        axes[0].plot(c.index, c, color=col, lw=2 if k in ("未避險",) else 1.6, label=k)
        axes[1].plot(c.index, c - c.cummax(), color=col, lw=1.4)
    axes[0].axhline(0, color=C["muted"], lw=0.8); axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_title("weekly 回看 20 日：選股門檻比較 (累積損益, 萬；beta 名目；不計手續費)")
    axes[1].set_title("回撤 (萬)"); axes[1].axhline(0, color=C["muted"], lw=0.8)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator()); axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    fig.tight_layout(); fig.savefig(R.OUT / "gate_fig1_weekly20.png", dpi=130); plt.close(fig)
    print("輸出:", R.OUT)


if __name__ == "__main__":
    main()
