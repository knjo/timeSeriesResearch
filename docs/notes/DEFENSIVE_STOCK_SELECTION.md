# 抗跌商品挑選規格

## 1. 目的與正確解讀

這套訊號的目標是，在交易日 D 開盤前，利用 D-1 已完成的市場資料與當時可見的
族群成分，挑出 D 日開盤到收盤期間**預期相對市場較強或較抗跌**的股票組合。

它不是「long-only 絕對報酬必為正」的訊號。市場大跌時，組合仍可能虧損；研究證據
只支持它平均比同日的 capacity-weighted 股票市場少跌或多漲。

目前凍結證據來自
[`news_collector/theme_strength_research.py`](../news_collector/theme_strength_research.py)
與
[`news_collector/plot_selected_theme_pl.py`](../news_collector/plot_selected_theme_pl.py)：

| 2025-01-01 起的 untouched validation | 平均 D 開盤至收盤 P/L |
| --- | ---: |
| 選中族群股票組合 | -11.47 BP/日 |
| Capacity market | 約 -26.41 BP/日 |
| 選中組合減市場 | +14.93 BP/日 |
| 強族群減弱族群 | +14.45 BP/日 |

因此，這個訊號適合當作 daily defensive prior、相對強弱排序或其他日內事件策略的
條件變數；不能單憑上述數字宣稱已有可直接交易的 long-only alpha。

## 2. 日期與可用資訊契約

對每個交易日 D：

1. 檔名為 D 的 `preMarketData` 視為 **D-1 完成資料**。
2. 新聞族群 membership 使用 `trade_date == D` 的 snapshot，但每一列必須滿足
   `evidence_date < D`。
3. 訊號可以使用 D-1 的價格、成交量、法人、融資與委託容量資料。
4. D 的 `open_price`、`close_price` 與 open-to-close P/L 只能作為 outcome，不能進入
   D 的訊號或篩選。
5. `nextDayOpen`、`ToCloseBP`、`300sBP` 等 label-like 欄位不得進入 feature。

最小因果關係如下：

```text
D-1 completed market data + evidence_date < D membership
    -> D defensive score and stock list
    -> D open entry
    -> D close outcome
```

若任何 membership 列不滿足 `evidence_date < D`，或 D-1 `PreviousClosePrice` 與來源
market close 對不上，產生器應 fail closed，不可猜測或用未來 snapshot 回填。

## 3. 凍結的族群評分

### 3.1 股票層橫斷面分數

每個 D 先在股票橫斷面上，以 average rank 將各原始特徵映射到 `[-1, 1]`：

```text
strength(x) = 2 * (average_rank(x) - 1) / (finite_count(x) - 1) - 1
```

至少需要兩個有限值。排名 ties 使用 average rank，避免輸入順序影響分數。

### 3.2 Capacity weight

股票在族群內的 prior-day capacity weight 為：

```text
l1_amount     = ((avg_bidLots1 + avg_askLots1) / 2)
                * PreviousClosePrice * 1000
volume_amount = trading_volume * PreviousClosePrice * 0.003

capacity_weight = clip(max(l1_amount, volume_amount), 0, 5_000_000)
                  / 5_000_000
```

這是盤前已知的可交易容量 proxy，不是 D 日成交結果。

### 3.3 族群層聚合與有效性

每個 `(D, theme_id)` 對成員股票分數做 capacity-weighted mean。族群必須同時滿足：

- 至少 3 檔有效成員；
- capacity effective-N 至少 2.0；
- 同日至少有 10 個有效族群可供橫斷面排序。

目前 `candidate_directional` 是 discovery 期間 2023-2024 凍結後的 signed mean：

```text
candidate_directional = mean(
    -score__turnover_rate,
    -score__short_cover_flow,
    +score__investment_flow,
    -score__open_gap,
)
```

其中 `short_cover_flow` 的原始定義已是
`-(short_netLots / trading_volume)`；實作時不可再憑名稱改寫正負號。

上述四個方向是在 discovery 期間依 family、FDR 與 rank IC 凍結的研究結果。上線後
不得用 2025 以後的 outcome 重新挑方向，再回頭改算同一段 validation。

## 4. 從族群轉成抗跌商品

每日依 `candidate_directional` 由弱到強排序；分數相同時以 `theme_id` 作穩定
tie-break。基準版本選最高 20% 族群：

```text
n_selected_themes = floor(n_valid_themes / 5)，下限為 1
theme_notional     = 1 / n_selected_themes
```

選中族群的有效成員全部進入候選組合；目前證據**沒有**在族群內額外挑「最強 3 檔」。
每個族群內依 `capacity_weight` 正規化，再乘上 `theme_notional`：

```text
member_weight(theme, stock)
    = theme_notional
      * capacity_weight(stock)
      / sum(capacity_weight of valid members in theme)
```

同一股票若屬於多個選中族群，將權重加總，得到該股票最終的
`defensive_weight`。這一步必須保留，否則重疊題材會造成未揭露的集中曝險。

建議的股票清單輸出 schema：

| 欄位 | 意義 |
| --- | --- |
| `decision_date` | 可在 D 使用的交易日 |
| `quote_code` | 四碼股票代號 |
| `defensive_weight` | 跨選中族群加總後的 long-only 目標權重 |
| `selected_theme_count` | 該股票命中的選中族群數 |
| `selected_theme_ids` | 排序後的 theme ID 清單，供稽核 |
| `source_date` | 實際使用的 D-1 市場日期 |
| `universe_sha256` | point-in-time universe 版本 |
| `logic_sha256` | 產生邏輯版本 |

若下游只允許固定檔數，應先另開 discovery/validation 研究決定 top-N 與集中度上限；
不能直接把 `defensive_weight` 最高的 15 檔稱為已驗證版本。

## 5. Optional reliability gate

現有研究另有 causal reliability gate：每個族群使用最近 120 次歷史
`score_rank * realized_relative_oc_bp`，先 `shift(1)`，至少 30 次歷史且平均 edge
大於零才保留。它不會永久封鎖某個族群，族群可隨 strictly-prior 結果重新進出。

但本文件表格中的 `+14.93 BP/日` 與原始累計圖對應的是 **`filtered=false`**，不可在
引用該績效時悄悄加上 reliability gate。若啟用 gate，必須使用自己的績效、版本與
manifest；現有 validation 的 long-minus-market 為 `+16.61 BP/日`，但仍是 gross、
相對報酬證據。

## 6. 交易與回測限制

原始抗跌研究的 label 是 D open-to-close，且報告明確未納入下列項目：

- 手續費、交易稅與滑價；
- D 開盤漲停的可成交性；
- 個股市場衝擊與實際容量限制；
- 固定檔數造成的集中度變化。

因此生產回測至少要把 D 開盤鎖漲停視為未成交現金，不得在看到開盤結果後換成另一檔
並回填當日績效。這是執行層保守修正，不應反過來改變 D 開盤前的候選排序。

也不要把另一份「D 收盤買、D+1 開盤賣」研究的收盤成交假設套用到本訊號；兩者的
持有區間、成交限制和證據不同。

## 7. 接入 timeSeries 的邊界

這份清單是 stock-day prior，不是 `(Date, QuoteCode, ChannelSeq)` 的日內事件。接入
timeSeries 時應：

1. 先產生唯一鍵為 `(decision_date, quote_code)` 的 point-in-time 清單。
2. 將 `decision_date` 對應事件資料的 `Date`，只做 many-events-to-one-stock-day join。
3. join 前檢查右表鍵唯一；join 後檢查事件列數完全不變。
4. 把 `defensive_weight` 當成 feature、eligible universe 或分層欄位；使用哪一種必須
   寫入 study spec。
5. 研究產物只寫到 `timeSeries/test/<study_id>/artifacts/<event>/<version>/` 或
   `/tmp`，不可覆寫 `data_index.parquet`、`data/` 或 frozen baseline。

最小防洩漏檢查：

```text
source_date < decision_date
membership.evidence_date < decision_date
stock-list key is unique(decision_date, quote_code)
event row count before join == event row count after join
no D outcome column is present in the feature manifest
```

## 8. 重現與驗收清單

每次產生候選清單，至少記錄：

- Git commit、Python 版本、`uv.lock` hash；
- theme universe、preMarket 與 market 輸入 hash；
- feature directions、最低成員數、effective-N、top 20% 與 weighting 版本；
- 每日有效族群數、選中族群數、唯一股票數、effective stocks、最大股票權重；
- 每日清單 canonical sort 後的 hash，以及整段輸出的 hash。

兩台機器在相同輸入、lock 與 Git commit 下，逐日股票清單及權重 hash 必須一致。
不同時先比對日期覆蓋、universe hash、logic hash 與套件版本，不可只比較累計 P/L。

建議至少保留以下驗收：

- discovery 僅限 2023-2024，validation 自 2025-01-01 起；
- validation capacity、`filtered=false` 的 long-minus-market 約為
  `+14.93 BP/日`，long-short 約為 `+14.45 BP/日`；
- 股票重建與 theme-level P/L 誤差小於 `1e-8 BP`；
- 原始研究結果是 gross relative edge，不是扣成本後的絕對報酬承諾。

## 9. Source of truth

- 評分、日期對齊、族群聚合與重疊權重：
  [`news_collector/theme_strength_research.py`](../news_collector/theme_strength_research.py)
- 原始累計圖：
  [`news_collector/plot_selected_theme_pl.py`](../news_collector/plot_selected_theme_pl.py)
- 研究報告：
  [`news_collector/artifacts/theme_strength/REPORT.md`](../news_collector/artifacts/theme_strength/REPORT.md)
- 日累計資料：
  [`news_collector/artifacts/theme_strength/selected_theme_daily_pl.csv`](../news_collector/artifacts/theme_strength/selected_theme_daily_pl.csv)

程式碼與 manifest 是事實來源；本文件若與程式衝突，應以程式為準並同步更新本文。
