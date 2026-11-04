# Dynamic short exits: tick replay

研究範圍是同一份 `signalwithRisk_df.parquet` 的出場政策比較。正式 pipeline、notebook、
原始訊號與 tickData 均不改寫。主版本採使用者指定的 trailing 30-minute VWAP。

主判讀使用 **2026 年**，與 notebook 的 BackTest 日期窗一致。資料口徑檢查發現
2025-12-24 前 OTC `RefPrice` 的已知錯置也存在於本訊號：該段 536 個 OTC stock-days
全數 `RefPrice` 等於當日 Close。全期重播保留作描述，不能拿那段結果支持因果主張。
2026 窗只評估固定訊號清單的出場差，不構成上游模型／訓練資料已通過 leakage audit。

## 事前規格

- 空單每列 1 張，`BidPrice1` 進場，保留同檔加倉與重新進場；不退還／重分配進場額度。
- 主對照重新 replay `RefPrice * 1.08`，沒有停利；不直接把 parquet 的 label 當成交路徑。
- 第一題：第 K=1..6 個不同持倉商品真的停損回補後，一次永久改為 RefPrice * 1.05/1.06。
  before-switch 全部政策與 8% 對照完全相同，因此其第 K 次回補時間可以共用；
  這不是拿調整後已不存在的 baseline 停損繼續決策。額外報 K=3/5 只改既有／只改新單。
- 第二題：span=2/2.5/3；進場固定、只更新時間波動度、VWAP 中心、只取近 SL、
  只取近 TP、兩者取近；2.5 加單向 ratchet 與 minute-last SMA 敏感度。
- EntrySL=min(entry*(1+span*entry_sigma), Ref*1.08)，EntryTP=entry*(1-span*entry_sigma)。
  RollingSL=min(VWAP*(1+span*sigma_t), Ref*1.08)，RollingTP=VWAP*(1-span*sigma_t)。
  NearSL=min(EntrySL, RollingSL)，NearTP=max(EntryTP, RollingTP)。
  使用者公式未給額外 absolute TP；不臆造對稱的 Ref*0.92。
- 每個整分鐘 09:30..13:24 用 [t-30m,t) 的正式成交 VWAP；前 30 分鐘先用進場固定線。
  sigma_t 使用盤前全時段 residual vol 及 canonical 13:30 clock，不估當日未來波動。
- 正常觸發採有效成交價 >=SL / <=TP，嚴格在進場後，重設時直接檢查當時 Ask1。
  觸發後以可觀察且有量的 Ask1 回補；沒有 ask 就等待下一筆有 ask 的行情。
  同 timestamp 的原始成交先於 scheduled reset；不同商品沒有可共用的 exchange sequence。
  已越過新門檻不會回溯以新門檻價成交。重新進場也不刪除，越界於下一個有效事件平倉。
- 未觸發者持至 13:30、以來源 Close 結算；觸發卻到收盤仍無 ask 者也用 Close **估值**，
  明列 unfilled_close_mark，並非假設漲停買得到。報全日排除此狀況的配對敏感度。
- 13:25..13:30 為收盤集合競價，不可用 13:24 的舊 ask 模擬即時成交；該時段不再
  產生盤中回補，剩餘部位按 Close 結算。來源：[TWSE](https://www.twse.com.tw/zh/products/system/trading.html)。
  `TrialMatch != 0` 的 timestamp 保留作非可成交狀態，避免把暫緩撮合前的 ask 當成
  當下可成交價格。v1-v3 為已取代的開發結果；v4 是完成此撮合狀態處理的正式研究快照。
- 費用 19.3bp；主結果用 Ask1，另加所有盤中出場 2 個進場 TickSize 的成本壓力測試。
  這是成本敏感度，不是可越過交易所漲跌停價的虛構成交價。
- 一筆 1 張不等於模擬大量同時回補可全吃同一 Ask1；未模擬 queue、latency、book depletion。
- TP/SL 是觸發線名稱；moving TP 可能仍虧損出場，moving SL 也可能獲利。

## 判定與統計

主要 estimand 是同日、相同進場的扣費損益差；另報平均 bp、名目加權 bp、等日 bp、
日 Sharpe、由零開始計算的 MDD、最差日與最差 5% 日均損益。改善是否有代價須同時看
總損益與尾部，不把「虧損變小但總收益下降」寫成無條件變好。

固定 45 個 candidate，全部保留逐筆輸出；先按 spec.py 凍結再讀研究結果。
5-session circular moving-block paired bootstrap（2000 draws）估計平均日差 CI；
以共同 centered bootstrap 的 family max-t 檢驗校正多重挑選（正向單尾）。
月份、2025/2026、2026-07-01 前後與 k/span 相鄰值用於穩定性診斷。
資料已被使用者與既有研究看過，時間切片不宣稱 untouched OOS，結果只屬 research diagnostic。
此研究沒有新增 selector，不執行 condition-discovery positive-control 或多方研究；
未重播 TXF hedge，報 raw trading BP，不把持至收盤 TXF return 當成提前出場的殘差。

## Reproduce

```bash
uv run python -m unittest discover -s src/research/timeSeries/test/20260922_dynamic_exit -p 'test_*.py'
uv run python src/research/timeSeries/test/20260922_dynamic_exit/run_study.py
uv run python src/research/timeSeries/test/20260922_dynamic_exit/summarize.py --output src/research/timeSeries/test/20260922_dynamic_exit/artifacts/v4
uv run python src/research/timeSeries/test/20260922_dynamic_exit/evaluate_2026.py --output src/research/timeSeries/test/20260922_dynamic_exit/artifacts/v4
uv run python src/research/timeSeries/test/20260922_dynamic_exit/reproduce_notebook.py --output src/research/timeSeries/test/20260922_dynamic_exit/artifacts/v4
uv run python src/research/timeSeries/test/20260922_dynamic_exit/write_report.py --output src/research/timeSeries/test/20260922_dynamic_exit/artifacts/v4
```

`--start/--end/--output` 可做小樣本快驗；續跑必須 `--resume` 且 fingerprint 相同。
manifest 保留訊號 SHA256、程式 SHA256、全部候選規格與 tick 分區 size/mtime。
tick 未計算全檔 SHA256，這是 source fingerprint 的限制；來源缺檔／缺商品直接失敗。
`primary_2026/` 保存主結果與全筆配對 invariants；根目錄 CSV 是全期描述。
重現到已存在的目錄會被拒絕；重新計算請用 `--output` 指定新的版本目錄，並將同一路徑
傳給 `evaluate_2026.py`。後加的主期間分析另保留 `evaluation_manifest.json` 的 analysis hash。

正式結論連至 [REPORT.md](../../Doc/studies/20260922_dynamic_exit/REPORT.md)。
