# negFill 單檔上限 × 每日總額網格（2026-11-02）

notebook `time_series_research_model.ipynb` 現行流程重建 2023-04-21～2026-10-30（cell 2–27 分三段、各留 65 日暖身：`build_book.py`），
再照 cell 34 風控（停手日、深度、R1/brench、價位、時間、makerSpread、20 日崩跌、2σ 停利停損、停損商品數 ≥3 後 10:05 禁新倉）
只改 `posLimit` ∈ {250…500} 萬與當日累計名目 `cap` ∈ {4,000…7,000} 萬（11/02 由 6,000 延伸到 7,000，原 216 格數字不變）（`grid.py` → `grid.csv`）。

兩種閘門語意：`notebook` = 只排除 stop_dates（現行 cell 34）；`graded` = 正式 RD 語意，cap × scale_by_date（停 0／縮額 0.1／台積電跳空 0.4）。
指標與 BackTest（taker）相同，單格已用 BackTest 驗算逐位一致。

結論（graded）：單檔上限決定 MDD、cap 決定獲利；大單檔 × 小 cap 最差（早盤大部位吃掉額度、後段較好的成交被擋）；
單檔 250 萬在 cap ≈4,750 後飽和。cap 6,000→7,000：獲利續增但遞減（2026 單檔 500 +436、400 +257、350 +174 萬），Sharpe 持平 4.6–4.7、MDD 不動，使用率降到 39–45%。
兩段分歧：2026 仍以 350 × 5,000 最高（4.74／−262）；2023–25 Sharpe 隨 cap 升到網格邊界（350 × 7,000 4.01／−288 vs 350 × 5,000 3.80／−322），7,000 還沒到頂。
兩段合看 350 × 6,500–7,000 優於 350 × 5,000（2023–25 Sharpe +0.15–0.21；2026 −0.10、獲利 +357～+428 萬、MDD −275 vs −262）。
最大獲利 500 × 7,000（2026 5,081 萬／4.64／−372；2023–25 9,586／3.82／−356）。graded 比 notebook 語意 2023–25 Sharpe 高 0.6–0.8。
報告：`make_report.py` → `negfill_poslimit_cap_grid.html`／`.pdf`（A4 橫式 6 頁）。

## 11/03 修正：comp-only 先前誤含出關冷卻
`risk_metrics.py comp`／`hi_grid.py` 原本讀 gate payload 的 `comp_scale`，它已含 2 天出關冷卻（30 天）。
改成 `scale_for_comp(comp)`（純分級）後：MDD、最差單日不變；Sharpe −0.03～−0.05、總獲利 −0～−140 萬、正常日效率 −0.8 pt。
舊結果留在 `hi_grid_withcooldown.csv`。

## 單檔 400 × cap 7,000 停手閘門（11/03）
`build_rows.py <book> <out> 400 keep_stop`（保留停手日成交，才能重放放寬門檻）→ `gate_tune.py`（comp 2,082 組、台積電 1,280 組，各自單獨）
→ `gate_combo.py`（合併 1,073 組、分桶表、2023–25 選 → 2026 驗）→ `make_gate_page.py`（`negfill_gate_400x7000.html`）；CSV 在 `gate400/`。
現行正式 scale_by_date 逐日重現（assert）。gate_daily.csv 要用 `float_precision="round_trip"` 讀，否則跳空 0.020000000000000018 會被讀成 0.02、少算兩天 >2%。
結論：comp 拉 Sharpe（3.16→3.81、MDD 幾乎不動），台積電壓 MDD（−482→−356）；兩層合併 4.04–4.07／−328。
comp −1.5／−0.7 維持（−0.7 是尖峰）；縮額 ×0.05～0.2 等價、建議 ×0.15（1,050 萬）；全停 vs ×0.1 無差；冷卻中性。
台積電 1–2% ×0.1／>2% ×0.4 維持（DD 優先）；1–2% 全停 MDD −312、0.75% 門檻 −275（少 1,350 萬）。剩餘兩段 DD 兩層都沒亮燈。
出關冷卻（`cooldown_variants.py`）：三種語意（不設／現行「限制期結束後再 N 天」／「停手日後 N 天、縮額日可抵」）MDD 全同、Sharpe 4.05–4.08、獲利 ±60 萬；
停手後第 1–2 天 comp 已回復的只有 13 天、日均 ≈ −7 萬（t −0.5）、第 3 天起轉正；現行語意多限制停手後第 3–6 天 16 天（comp 已回復 ≥2 天）。數據分不出來，留的話用停手日錨定版。

## 受限日的單檔上限 × cap（11/03，無冷卻）
`group_poscap.py`（單檔 50–400，keep_stop 列）＋`group_poscap_hi.py`（500–1,000，用 1,000 superset 列；stop 日本來就是 cap 0）→ `group_poscap_grid(_ext).csv`、`group_poscap_uniform(_ext).csv`、`group_poscap_joint.csv`（13,824 組）；
`group_poscap_boot.py`／`verify_bootstrap.csv` 成對區塊 bootstrap。workflow 獨立重寫 replay 5 組全對、code review 無 bug。
結論：三組 cap 都 1,000 時單檔 400 Sharpe 4.07；縮到 300／200／100 掉 0.03／0.09／0.11（200、250、350 顯著）；放大到 1,000 +0.06（區間 +0.01～+0.12、四年都正）。
機制：cap 1,000 在單檔 400 下中位數 09:01 打滿；單檔 p 萬會擋掉股價 ≥10p 元的股票。MDD 差異全來自 2025-07-31（G2）、08-08（G3）兩天，不能拿來挑。
G1 cap 850–1,250 兩段一致；G2 700–1,250 等價。注意：hi 版第一次跑有 bug（pos 與正常日 400 取 min，>400 全被壓回 400），已修。
G3 細網格（`g3_fine.py`／`g3_tranche.py` → `gate400/g3_fine.csv`，每 350 萬一格）：1,000 是 G3 的谷底（組內 −46 萬），3,150–4,200 平台（組內 +200～+320、總獲利 +250～360、MDD −322～−329、Sharpe 不變），>4,500 最差日 −140～−166、Sharpe <4.0。
1,050→3,500 這段 65 天 +304 萬、t 0.94、2024–26 都正；MDD 改善只來自 2025-08-08。建議 G1 1,000／G2 700／G3 3,500：14,144／−325／4.06。

## 主管版摘要 PDF：400 × 7,000（11/03）
[`Doc/negfill_400x7000_summary.pdf`](../../Doc/negfill_400x7000_summary.pdf)（A4 兩頁，給非技術主管；依使用者要求不寫停手判斷方法、天數，無附註）。
最終版口徑：comp 分級＋出關冷卻（gate_daily `comp_scale`）× 台積電跳空 >1% 一律 0.5（使用者指定，避免 0.1/0.4 過擬合；正式 stopTrading 目前在台積電日發 0.1）× cap 7,000。
重放＝`gate400/rows_pos400_keepstop.parquet` + stopLossProd 閘；只全停版與 grid.csv `notebook` 400×7,000 逐位一致（8,934.60／5,177.62）。
最終版：全期 14,212 萬、年化 4,018（×242 日）、MDD −337（2025-07-23→08-18）、近期 −317（2026-09-17→10-14）、最差日 −186、P10/P50/P90 −52／+5／+104、日均交易金額 3,800、Sharpe 3.94。
對照（同重放）：只全停 14,112／−444（近期）；正式發布值（台積電 0.1、量化）13,687／−338、近期 −304；不控管 13,205／−482。
產生：`supervisor_pdf/` 依序 `build_series.py` → `mc.py` → `make_report.py`（PDF 直接寫到 `Doc/`）；輸入是 `supervisor_pdf/inputs/` 的 11/03 快照（rows_pos400_keepstop、停損時間 cache、book 交易日曆、gate_daily.csv），11/04 從快照重跑與原版逐位一致。variants.py 對照表未搬。
11/03 續：報告拿掉「控管 vs 不控管」對照，加蒙地卡羅壓力測試（`mc.py`）：comp 停/縮門檻、縮額比例、台積電門檻/比例、額外成本 0/2/4bp 各 3 值（729 組，cap × scale 真實重放）
× stationary block bootstrap（平均 10 天）20,000 個一年路徑。一年內 MDD 中位 −328、最差 10% −479、5% −540、1% −671（>−500 佔 7.8%）；年獲利中位 3,532、最差 10% 2,320、無整年虧損。
729 組全期 MDD −294～−471（中位 −393），報告版 −337 偏好的一端。bootstrap 產生不出比歷史更差的單日，所以單日最大虧損不放 MC。
