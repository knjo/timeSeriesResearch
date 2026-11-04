"""Render a human-readable research report from the frozen replay artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from summarize import table

HERE=Path(__file__).resolve().parent
REPORT=HERE.parent.parent/"report"/"studies"/HERE.name/"REPORT.md"


def write_report(output):
    primary=output/"primary_2026"
    summary=pl.read_csv(primary/"summary.csv")
    lookup={r["candidate"]:r for r in summary.iter_rows(named=True)}
    base=lookup["ref08"]
    entry=lookup["vol_entry_2p5"]
    manifest=json.loads((output/"manifest.json").read_text())
    evaluation=json.loads((primary/"evaluation_manifest.json").read_text())
    literal=json.loads((primary/"notebook_literal.json").read_text())
    q1=[]
    stops=[json.loads(p.read_text()) for p in (output/"days").glob("*_stops.json") if p.name[:8]>="20260101"]
    for k in range(1,7):
        r5,r6=lookup[f"cascade_k{k}_to05"],lookup[f"cascade_k{k}_to06"]
        q1.append({"K":k,"觸發天數":sum(len(v)>=k for v in stops),"5% 損益差_萬":r5["delta_wan"],
                   "5% MDD_萬":-r5["mdd_wan"],"5% Sharpe":r5["sharpe"],
                   "6% 損益差_萬":r6["delta_wan"],"6% MDD_萬":-r6["mdd_wan"],"6% Sharpe":r6["sharpe"]})
    q2=[]
    labels={"vol_entry_2p5":"進場固定 2.5σ","vol_clock_2p5":"進場中心、每分鐘縮 sigma",
            "vol_rolling_2p5":"VWAP 更新兩條線","vol_near_sl_2p5":"只將停損取近",
            "vol_near_tp_2p5":"只將停利取近","vol_near_both_2p5":"停損停利都取近",
            "vol_ratchet_2p5":"都取近且歷史線只能收緊"}
    for name,label in labels.items():
        r=lookup[name]
        q2.append({"版本":label,"淨損益_萬":r["pnl_wan"],"對進場固定差_萬":r["pnl_wan"]-entry["pnl_wan"],
                   "Sharpe":r["sharpe"],"MDD_萬":-r["mdd_wan"],
                   "加2tick後差_萬":r["stress_pnl_wan"]-entry["stress_pnl_wan"]})
    anatomy=[]
    for k in (3,5):
        for pct in (5,6):
            name=f"cascade_k{k}_to{pct:02d}"
            anatomy.append({"版本":name,"既有部位差_萬":lookup[name+"_existing"]["delta_wan"],
                            "新單差_萬":lookup[name+"_new"]["delta_wan"],"改線即退出筆數":lookup[name]["reset_exit"]})
    neighbors=[]
    for span in ("2p0","2p5","3p0"):
        neighbors.append({"倍數":span.replace("p","."),
                          "VWAP兩線差_萬":lookup[f"vol_rolling_{span}"]["delta_wan"],
                          "只SL取近差_萬":lookup[f"vol_near_sl_{span}"]["delta_wan"],
                          "只TP取近差_萬":lookup[f"vol_near_tp_{span}"]["delta_wan"],
                          "兩線取近差_萬":lookup[f"vol_near_both_{span}"]["delta_wan"]})
    uncertainty=[]
    for name in ("cascade_k5_to05_new","vol_rolling_2p5","vol_near_tp_2p5","vol_near_both_2p5"):
        r=lookup[name]
        uncertainty.append({"版本":name,"平均日差_萬":r["delta_day_wan"],"95%CI低":r["delta_ci_low_wan"],
                            "95%CI高":r["delta_ci_high_wan"],"family調整p":r["family_adjusted_p"],
                            "正差月份":r["positive_months"],"負差月份":r["negative_months"]})
    daily=pl.read_csv(primary/"daily.csv",schema_overrides={"Date":pl.String})
    split=[]
    for name in ("cascade_k3_to05","cascade_k5_to05","vol_rolling_2p5","vol_near_tp_2p5","vol_near_both_2p5"):
        r=daily.filter(pl.col("candidate")==name).sort("Date")
        b=daily.filter(pl.col("candidate")==lookup[name]["baseline"]).sort("Date")
        delta=r["pnl_twd"].to_numpy()-b["pnl_twd"].to_numpy()
        mask=r["Date"].to_numpy()<"20260701"
        split.append({"版本":name,"1至6月差_萬":delta[mask].sum()/1e4,"7至9月差_萬":delta[~mask].sum()/1e4})
    near=lookup["vol_near_both_2p5"]
    root_rel=f"../../../test/{HERE.name}/artifacts/{output.name}"
    lines=[
        "# 空單停損檔數縮限與 VWAP 動態出場：tick 重播",
        "", "Status: `scoped` / `RESEARCH_DIAGNOSTIC_ONLY`。只比較既定空單清單的出場政策，未部署。", "",
        "## 結論", "",
        "1. **不支持『停損多檔後，全部縮到 5%／6% 就能增加總獲利』。** K=1..6 的 12 組全為負增益；部分能縮小回撤，但代價是少賺。既有部位的過早回補是重要來源。",
        f"2. **VWAP 動態更新沒有穩健優於進場固定線的證據。** 2.5σ 兩條線取近，相對進場固定增加 {near['delta_wan']:.2f} 萬；額外每次盤中回補加 2 tick 後，差額為 {near['stress_delta_wan']:.2f} 萬。相鄰倍數、月份與配對 CI 應一起看，不能只挑單一正點。",
        f"3. 使用者貼出的 notebook 算法確實重現正增益：{literal['original_label_pnl_wan']:.2f} → {literal['notebook_vol_pnl_wan']:.2f} 萬，增加 {literal['delta_wan']:.2f} 萬。那是 strict touch、TP 按門檻、SL 按門檻+2tick 的口徑；主研究按可觀察 ask 回補，兩者不是同一個執行假設。",
        "", "## 資料範圍與對照", "",
        f"- 完整檔案：2025-10-16～2026-09-22，{manifest['n_days']} 天、{manifest['n_rows']:,} 列；45 組、每列同一筆 1 張空單。",
        f"- 主期間：2026-01-02～2026-09-22，{base['n_days']} 天、{base['n_trades']:,} 筆；沒有缺檔或因候選不同而換樣本。",
        "- 全期只作描述：2025-12-24 前 OTC RefPrice 有既知錯置；本檔該段 536 個上櫃股票日全部 RefPrice=當日 Close。主期間因此使用 notebook 原本的 2026 日期窗；不是依績效選起日。上游模型訓練資料不在本研究 audit 範圍。",
        "- 進場固定 BidPrice1，每列 1000 股，費用 19.3bp；不因提早出場重分配額度、重選股票或刪掉重新進場。所有金額單位均為萬元。",
        f"- 8% baseline：淨損益 **{base['pnl_wan']:.2f}**、名目加權淨 BP **{base['net_pos_bp']:.2f}**、日 PnL Sharpe **{base['sharpe']:.3f}**、MDD **{-base['mdd_wan']:.2f}**、最差日 **{base['worst_day_wan']:.2f}**、最差 5% 日均 **{base['cvar05_wan']:.2f}**。MDD 表格列正的損失幅度，愈小愈好。",
        "", "## 1. 第 K 檔停損後縮限", "",
        "K 計的是不同持倉商品完成首次停損回補；同檔多筆不重複增加。每個政策在觸發前完全等同 8% baseline，所以可以共用 baseline 的第 K 個實際回補時點。觸發後一次永久改成 5% 或 6%，不再利用原本 8% 版本後續的停損事件追加收緊。",
        "", table(pl.DataFrame(q1),list(q1[0])), "",
        "效果不是單純『多停損之後的單比較差，所以縮小停損必然改善』。縮限也會回補後來回落、原本能賺錢的部位；既有／新單拆分如下：", "",
        table(pl.DataFrame(anatomy),list(anatomy[0])), "",
        "改線即退出筆數按 1 張訊號列計算，不是股票檔數。原本已漲到 7% 才把上限改成 5%，按當時 ask（或無 ask 時後續第一個可用 ask）回補；絕不把損失回填成 5%。新單若已超過縮小後的線，保留原進場，於第一個後續有效事件回補。只改新單／只改既有部位的差額相加已逐筆驗證等於全部調整。",
        "", "## 2. 以 30 分鐘 VWAP 更新，並與進場線取近", "",
        "使用者指定 VWAP。每個整分鐘以 [t−30分鐘,t) 正式成交做量加權；09:30 前保留進場線，空窗不更新。sigma_t 是盤前完整時段 residual vol 乘 canonical 剩餘至 13:30 clock，不讀當日未來報酬。", "",
        "```text\nEntrySL = min(entry * (1 + span * sigma_entry), RefPrice * 1.08)\nEntryTP = entry * (1 - span * sigma_entry)\nMovingSL = min(VWAP30_t * (1 + span * sigma_t), RefPrice * 1.08)\nMovingTP = VWAP30_t * (1 - span * sigma_t)\nNearSL = min(EntrySL, MovingSL)\nNearTP = max(EntryTP, MovingTP)\n```", "",
        "所謂進場版，是原始進場時固定的線，沒有隨 t 重新計算進場 sigma。單獨更新 sigma 的 clock 對照另外列出。使用者公式沒有額外 absolute TP，故不臆造 Ref×0.92；取近版以原始 EntryTP 當 floor、EntrySL 當 ceiling。取近版可以回到原進場線內較寬的位置；永遠只能收緊的 ratchet 是另列版本。", "",
        table(pl.DataFrame(q2),list(q2[0])), "",
        "表中損益差都對『進場固定 2.5σ』。加 2 tick 是每次盤中回補的額外成本敏感度，包含 TP；不是把虛構成交價推到漲跌停範圍之外。它與原 notebook 僅 SL 加 2 tick 的定義不同。", "",
        "相鄰倍數（各自對同倍數進場固定線）：", "",table(pl.DataFrame(neighbors),list(neighbors[0])), "",
        f"取近 2.5σ 的 TP 觸發中有 {near['tp_loss_count']} 筆按原始進場成本仍虧損；TP 是觸發線名稱，不保證獲利。另提供 minute-last SMA 版本作均價定義敏感度，結果在完整 summary.csv。",
        "", "## 3. 時序、撮合與統計驗證", "",
        "- **收盤集合競價**：只在 09:30～13:24 更新可即時執行的線；13:25～13:30 不以舊 ask 模擬成交，剩餘部位按來源 Close 結算。依 [TWSE 交易制度](https://www.twse.com.tw/zh/products/system/trading.html)，逐筆交易至 13:25，收盤採集合競價。",
        "- **試撮狀態**：保留非正式 tick 的 timestamp 並令該狀態 ask 不可執行；不得沿用前一筆正式 ask 穿過試撮／暫緩撮合。沒有賣單時停損單等待下個正式有量 ask；到收盤仍無 ask 者只估值、明列 unfilled_close_mark。",
        "- **因果規則**：正常觸發用正式成交 >=SL 或 <=TP，嚴格在進場後；同時觸發 SL 優先。VWAP 不含 t 當下或未來成交。改線時按當下狀態判斷已越界，成交用報價而非新門檻。",
        "- **機器驗證**：19 個 synthetic/integration tests；全筆檢查相同 entry support、出場晚於進場、vol SL 不超過 Ref×1.08、取近不晚於原版、觸發前不改損益、既有／新單分解一致、集合競價無盤中假成交。另抽樣用既有 first_touch_times 核對靜態觸價時間。",
        f"- **多重比較**：45 個固定候選全部留存，差額相關矩陣的 participation-ratio 有效數約 {evaluation['effective_trials_participation_ratio']:.2f}（描述性估計）。使用同一套 5 日 circular moving-block bootstrap、2,000 抽樣；主估計為配對平均日損益差，正向 family max-t 校正。CI 沒有校正多重比較；p 有。",
        "",table(pl.DataFrame(uncertainty),list(uncertainty[0])), "",
        "2026 年前後段診斷（仍是看過的歷史，不能稱 untouched OOS）：", "",table(pl.DataFrame(split),list(split[0])), "",
        "## 限制與可證偽後續", "",
        "這是固定清單的出場比較，不是從原始所有訊號重建下單／容量策略。逐列 1 張、共享 Ask1，未消耗多筆同時回補的深度、未加真實延遲／排隊，Close 是終點結算基準而非保證回補成交；暫緩收盤的實際終止時間也未另外建模。因此不把正 PnL 當成可部署證明。",
        "此題只研究空方 raw trading PnL，未重播 TXF hedge；不把原本到 13:30 的 TXF 報酬錯套到提早出場後的 residual BP／z。訊號選擇與模型有效性視為既定，不宣稱整體策略已因果驗證。",
        "目前適合的判定是：縮限若要採用，必須明確接受少賺換較小回撤；VWAP 取近的微小增益尚不足支持替換。若繼續，先固定單一規則與真實延遲／成本，再收集未參與本次比較的新期間資料，禁止把這次微小最佳值直接當正式參數。",
        "", "## 產物與重現", "",
        f"- [研究程式與公式](../../../test/{HERE.name}/README.md)。",
        f"- [2026 全部 45 組結果]({root_rel}/primary_2026/summary.csv)、[每日結果]({root_rel}/primary_2026/daily.csv)、[每月差額]({root_rel}/primary_2026/monthly_delta.csv)、[既有／新單分解]({root_rel}/primary_2026/cascade_decomposition.csv)。",
        f"- [逐筆出場 ledger]({root_rel}/days/)、[改線即出場例子]({root_rel}/primary_2026/reset_examples.csv)、[invariants]({root_rel}/primary_2026/integrity.json)、[notebook 原式重現]({root_rel}/primary_2026/notebook_literal.json)。",
        f"- [Replay manifest]({root_rel}/manifest.json)、[2026 evaluation manifest]({root_rel}/primary_2026/evaluation_manifest.json)。", "",
        f"![2026 equity]({root_rel}/primary_2026/equity.png)", "",
        f"Signal SHA256: `{manifest['signal_sha256']}`。Replay fingerprint: `{manifest['fingerprint']}`。Tick 分區紀錄 path/size/mtime，沒有計算全部 tick 檔案 SHA256；原檔變動必須另建輸出版本。",
    ]
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text("\n".join(lines)+"\n")
    (primary/"report_manifest.json").write_text(json.dumps({"source_fingerprint":manifest["fingerprint"],
        "report_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256":hashlib.sha256(REPORT.read_bytes()).hexdigest()},indent=2))
    full_summary=(output/"SUMMARY.md").read_text()
    if not full_summary.startswith("> Full-period descriptive"):
        (output/"SUMMARY.md").write_text("> Full-period descriptive output includes the known pre-20251224 OTC RefPrice defect. Use primary_2026/SUMMARY.md and the study REPORT.md for conclusions.\n\n"+full_summary)
    print(REPORT)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    write_report(p.parse_args().output)
