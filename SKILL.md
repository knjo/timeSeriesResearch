---
name: timeseries-signal-research
description: timeSeries 乾淨訊號研究範本（de-beta + U-shape vol normalization）。開發任何日內多空訊號時使用：data merge、label 構造（多空符號規則）、特徵紀律、rolling 模型、16-cell 門檻、評估紀律與上線前查核清單。含 negFill 2025-09~2026-08 實證出處與踩雷紀錄。
---

# timeSeries 訊號研究範本

目標：在「去 beta + U-shape vol 還原」的乾淨框架上開發日內多空訊號。
實證基礎：negFill 空方訊號 2025-09~2026-08（見文末事故紀錄，所有規則都有數字出處）。

## 0. Condition 的角色（框架第一原理）

訊號 condition（如 negFill）的本職是「分佈手術」，四個功能各自獨立：

1. **切偏斜分佈**：把 E[label|C] 從全市場的 ~0 切到顯著偏離（多空皆可），研究從有利底率開始——之後所有層都只是在這個底率上做排序與汰選。
2. **抽樣衛生**：觸發條件中的變動項（如 spread/B1 diff）是去重，避免同一盤口狀態被反覆計入樣本。殘餘問題：同一 stock-day 事件簇共享同一收盤 outcome，有效樣本 << 列數 ⇒ 評估必須 per-day、t-stat 在日級、部位 per-stock cap。
3. **製造 local linearity**：全樣本無效的 feature 在子分佈內可有局部線性 ⇒ 線性 Ridge 才可用。條件只保證「局部」不保證「平穩」——子分佈內的線性關係會漂（ToRef 事故），所以要配 §4 的平穩性提名與門檻分桶（cell = 再切一層局部）。
4. **事件敘事是可選 prior**：統計顯著足以決定要不要上；機制決定盯什麼（哪種 regime 會殺它、該做哪個分解檢驗）。無敘事的 edge 需要更緊的衰減監控。

常設儀表：月度 base-rate 表 E[z|C]（條件切割力）。它比模型 IC 更上游，它歸零時下游全免談。

## 1. 資料流

```
data_index.parquet（訊號點 unikey，由 build_index.py 產生）
  → data_loader.py 逐日 merge：tickData + tickFeature + preMarket + tickBar + TXF
  → src/research/timeSeries/data/YYYYMMDD.parquet
```

- 重建：`uv run python src/research/timeSeries/data_loader.py --start YYYYMMDD --end YYYYMMDD`
- **merge 層只接資料、不算 label**。label 一律在 notebook/研究層算。
- TXF 欄位（`src/features/market_beta.py` + `txfDataLoader.py` 產出）：
  - `txf_beta_60d`：盤前算好的 60 日 beta，`n/(n+20)` 向 1 收縮、clip [-0.5, 2]，嚴格 prior-only。
  - `txf_residual_vol_0900_1330_60d`：盤前 60 日日頻殘差 vol。
  - `txf_residual_vol_to_1330`：上者 × U-clock 時間縮放（`scale(x)=sqrt(x)(1.1177218-1.4889296x+1.3712078x²)`，x=剩餘秒/16200）。事件當下可知。
  - `txf_to_1330_return` / `txf_to_1330_bp`：事件→13:30 已實現期貨報酬。**未來函數，只准進 label/評估，絕不准進特徵。**
  - `RemainSeconds_1330`：到 13:30 的秒數（注意舊欄 `RemainSeconds` 是到 13:25）。
- TXF 資料來源標記 `QuoteSource`：local_tick / nas_txf_only_l1 / sdk_futures_l1_trades（SDK 日只有成交無報價，entry tolerance 10s）。
- 可用窗：20250211 起（beta ramp-up 前 vol 為 null）。TransTime ≥ 13:30 的列 vol=0 → z=inf，時間濾網要擋掉。

## 2. Label 構造（多空符號規則）

```python
# 空方（跌=賺）：label = -r_stock ⇒ 內含市場成分 -β·txf ⇒「加回」對沖腿
residual = (TakerSell_CloseBP + txf_beta_60d * txf_to_1330_bp) / (txf_residual_vol_to_1330 * 1e4)
# 多方（漲=賺）：label = +r_stock ⇒ 內含 +β·txf ⇒「減掉」對沖腿
residual = (TakerBuy_CloseBP  - txf_beta_60d * txf_to_1330_bp) / (txf_residual_vol_to_1330 * 1e4)

net  = residual - groupby(['Date','QuoteCode']).transform('mean')  # 選時點 label
netM = residual - groupby(['Date']).transform('mean')              # 選股 label
```

- z 是「幾個標準差」。**z 不是零均值**，水位隨 regime 漂移（negFill 曾漂到 -0.25σ）⇒ 門檻必須是相對式，絕不能用固定絕對值。
- 停損型 label（如 FutureHigh > Ref×1.08 改用停損價）與「持有到收盤」的對沖腿 horizon 不一致，是已知近似（停損事件的最適對沖 k≈0.12，非停損 k≈0.71，Epps 效應）。
- pred 換回 BP 是「乘」vol 不是除；只在過費用檢查時做，寫新欄位不要覆寫。

## 3. 特徵紀律

- **只用白名單，禁止 regex 選特徵。** 事故：`columns.str.contains('_re')` 掃進 `txf_to_1330_return`（未來函數），
  raw-label 模型直接拿到答案，「residual 比 raw 差」全是假象（修正後 2026+ 損益 74 萬 → 1,145 萬）。
- 快篩法：看模型 dashboard 的 feature importance，出現 txf_*（除兩根 vol）= 紅燈。
- 合法 ex-ante txf 特徵只有三根：`txf_beta_60d`、兩根 vol。

## 4. 座標 vs alpha 特徵（grad × IR 提名法）

用月度 IC 面板把特徵分三類（判準固定、輸入滾動，季頻重審）：

- `grad` = 平均每月 |IC|，`IR` = 月度 IC 均值/標準差，`flip` = 反號月數。
- **大且不穩（grad≥0.03 且 |IR|<1 或 flip>2）→ 結構座標**：交給門檻分桶或滾動基線，不留給固定係數模型。
  實證提名：ToRef（IC 0.28→0.01 死亡）、RemainSeconds（純時鐘，同時刻全股同值，原理上不可選股）、ToHigh/ToOpen/Low_High/FillLots_at* 整個位置座標家族、日頻 vol。
- **大且穩 → alpha 特徵**：MD_L1Rate_30、B1_A1B1、ToLow、QL_BidHHI、L1_BuyBiggestLots、事件層 vol。
- 小梯度 → 一般特徵。

## 5. 模型

- `src/research/rolling_model.py::train_and_predict_ridge_rolling`，每日重訓，net 用 60d、netM 用 20d，alpha=0.1。
- 函式內會 `dropna(subset=[Y_col])`（vol null 列會消失）；特徵缺值用前一日中位數補。
- **不要切 day-type 模型**：AB 日獨立模型與 normal 模型預測 rank 相關 0.83、IC 零增益；混訓 pooled 略優或持平。
  要保守就用同一組 pred 配較嚴門檻，別切模型（AB 門檻 q=0.9/0.5 曾在 2026 掐掉 ~414 萬）。
- Ridge 固定係數對 regime 轉變反應極慢（ToRef IC 1 月死、係數押了一整年）⇒ 結構座標務必移出模型（見 §4）。

## 6. 門檻（16-cell 配方）

```python
TimeBucket  = pd.cut(RemainSeconds, [3300, 6900, 10500, 14100, 99999], labels=False)
ToRefBucket = pd.cut(ToRef, [0, .01, .02, .03, .05], labels=False)
Cell2D = TimeBucket * 10 + ToRefBucket
thr = prior_day_quantile(df, pred_col, q=..., n_days=3, group='Cell2D')  # 開盤前即全部固定
condition = (net_pred > thr_net) & (netM_pred > thr_netM)
```

- 為什麼分桶在門檻層而非 label 層：**改 label 扣條件基線 f̂ 對組成無效**——預測「離散度」的時段梯度仍在，
  pooled 分位數挑尾巴、尾巴胖的時段永遠贏；分桶門檻同時中和均值與離散度。淺樹自動切桶平衡度不如固定 grid。
- 桶的維度 = §4 提名的結構座標（本例：時段 × ToRef）。要開放新價位區（如 ToRef<0）就是加 cell，各 cell 自取 top slice。
- q 必須分窗調參/驗證（本例 tune 2025-12-05~2026-03-31 / holdout 2026-04-01+，rank stability 0.85）。
- 已知殘餘：最終成交仍偏早盤 ~75%，來自 accLots/Position 先到先贏的倉位上限，屬執行層，門檻管不到。

## 7. 評估紀律

- **per-day Spearman IC vs 對應 label**。pooled raw qcut 會被市場成分騙：
  2026 曾出現「漂亮的單調反轉」，分解後 100% 是 hedge 項排序（洩漏特徵在排未來市場方向），拔掉即消失。
- qcut 診斷永遠帶分解：`raw = idio - hedge`，每 bin 看 idio/hedge 各自的斜率。
- net|net IC 會被 RemainSeconds 機械梯度灌水（2026 仍有 0.44）；真選股力看 netM|netM（2026 只剩 0.02~0.05）。
- 費用 19.3bp；報 capw bp（資金加權）+ 筆/日 + Sharpe + MDD/平均部位。

## 8. 上線前查核清單

- [ ] 特徵白名單過目，無任何未來欄位（§3 事故）
- [ ] 門檻為相對式且開盤前可得（prior_day_quantile；盤中不更新）
- [ ] q 調參/驗證分窗，rank stability 報告
- [ ] **hedge 照妖鏡**：新訊號段的 `idio − fee > 0` 才算 alpha。
      事故：負 ToRef 段未避險 holdout +304 萬，掛上 β·txf 對沖只剩 idio 2.3bp → 全是 β，不是選股。
- [ ] day-type / regime 切分需先用「切 vs 不切」對照證明增益
- [ ] 組成審計：選單對母體的 TV 距離、各座標桶佔比（目標貼平母體，本例 TV 0.03）
- [ ] 逐月分解至少含一段純 holdout；死月（訊號無 edge 的月份）門檻救不了，要靠 feature 重建

## 9. 檔案地圖

| 檔案 | 用途 |
|---|---|
| `src/features/market_beta.py` | beta / vol / U-clock（含常數與公式註解） |
| `src/dataloader/txfDataLoader.py` | TXF L1 讀取、to-1330 return、10s tolerance |
| `src/research/timeSeries/data_loader.py` | 逐日 merge、read_time_series |
| `src/research/timeSeries/build_index.py` | 訊號點 index |
| `src/research/rolling_model.py` | rolling Ridge（notebook 用這支，非 multiLabel） |
| `src/research/qcut_analyzer.py` | plot_qcut_analysis / plot_cut_distribution |
| `src/research/backTest.py` | 回測報表 |
| `script/backfill_txfTickData.py` / `validate_txfTickData.py` / `augment_preMarket_beta.py` | TXF 補資料與 beta 增補 |
| `src/research/timeSeries/time_series_research_model.ipynb` | 範本實作（cell 21 門檻 / cell 23 條件） |

## 10. 實證出處（negFill，校準預期用）

- 2025 年條件後 edge z≈0.31-0.64；2026 Feb-May ≈ 0、Jun-Jul 半復活（~0.3）、規模縮至 1/3。
- 2026 regime：vol 215→290bp、+8% 拉板率 4%→11-14%（空方逆風）、ToRef 橫斷面排序力死亡。
- 最終採用（16-cell 0.3/0.1）：全樣本 4,904 萬/Sharpe 5.00；holdout 1,111 萬/2.22；MDD/部位 8.9%。
- 與時間桶 0.5/0.2 配對日檢定：holdout t=0.21（不可分），16-cell 換得組成平衡與座標集中風險免疫。
