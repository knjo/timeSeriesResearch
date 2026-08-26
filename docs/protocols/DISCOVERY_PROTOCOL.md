# 長持倉 event-condition 探索契約

本文件定義目前 `timeSeries` 框架的主要研究任務：從日內可觀察資訊找出類似
negFill、能切出乾淨 label bias distribution 的 event condition。這不是只為
negFill 設計；negFill 與 `FillLots_atLow < 0` 是已知現象，用來驗證研究流程能否
先找回答案，再探索未知事件。

## 目前研究目標

- 最長持倉 horizon 是收盤。
- 多、空分別使用自己的 `DirectionalLabelSpec`；entry、停損／停利 barrier、
  adverse fill、beta hedge 與波動標準化都是 label 契約的一部分。
- 「持有到收盤」不等於移除途中 barrier；不得為了做對稱測試而擅自改成 pure-close
  label，也不得用空方 label 乘以負號代替多方 label。
- 當前要找的是相對同一 eligible pool 較有利的候選分布，不要求 event tick 是同股
  附近數分鐘內的局部最佳點。

因此目前固定：

```text
primary_estimand = shared_pool
control_population = eligible_pool
control_sample_size = None  # candidate sweep 預設完整 control
```

正的 `delta_z` 表示對該次 label direction 有利。long 與 short 必須各自建 label
並各自評估，因為 bid/ask entry 與 path-dependent exit 不對稱。

## Stage 0：positive-control recovery gate

任何新研究流程在宣稱能探索未知條件前，必須先使用完全相同的 pool、label、beta、
vol 與 control pipeline 找回事前指定的已知現象：

```text
FillLots_atLow < 0
```

框架入口是：

```python
from src.research.timeSeries.framework.profiles.filllots_at_low import (
    select_filllots_at_low_state_candidates,
)

conditions = select_filllots_at_low_state_candidates(day)
negative_state_keys = conditions["negative_state"]
```

這個 calibration selector 是完整 state condition。此階段不得先加入：

- formal-trade-only；
- state change／onset／epoch entry；
- depth normalization 或 threshold；
- cooldown、每 epoch 一筆或持倉上限；
- same-stock／same-stock-time 候選排序。

以上都是 recovery 成功後的事件形狀、執行與歸因問題。若原始 condition 的 shared
bias 無法回收，先檢查 selector、pool、label fingerprint、來源重建、beta／vol、
control 與符號，不可直接開始挖新候選。

這個條件是事前指定的單一 positive control；驗收看 effect 方向、量級與跨期表現，
不把探索 26 個候選後的 BH q 值倒過來當作 recovery gate。2026 衰退是 stability
資訊，不等於 2025--2026 全期仍可用的已知 bias 消失。

## Stage 1：最小 state partition

對新 feature 先測最少假設的可觀察分割，例如：

```text
state < 0
state > 0
state in fixed economic ranges
```

累積狀態可以在許多 row 持續成立；這是 condition distribution，不等於每個 row 都
下新單。不要為了避免重複交易，先把 selector 改成 crossing 或 state update，否則
可能在尚未確認 bias 前就改掉研究問題。

每個方向至少報：

- condition rows、active days、stocks/day、selection rate；
- shared pooled `delta_z` 與 equal-day `delta_z`；
- raw BP、beta-residual BP uplift；
- daily HAC、positive months；
- 2025／2026 分段方向與量級；
- label profile id、fingerprint、pool id/version、event selector version；
- 與既有 event 的 tick、stock-day 與日損益重疊／相關性。

## Stage 2：事件形狀與 threshold

原始 state 已證明有 shared bias 後，才比較：

- state update；
- sign crossing／re-entry；
- epoch start／entry；
- depth-normalized threshold ladder；
- cooldown 或每 episode 第一筆。

這一層回答「如何把有 bias 的狀態整理成較可執行的事件」，不能把新變體的 matched
效果拿來否決 Stage 0 已回收的 shared bias。掃多個候選時，候選集合與 threshold
先固定，shared family 再做相應的 multiple-testing correction。

## Stage 3：歸因，不是當前主淘汰門檻

same-stock 與 same-stock-time 用來說明 shared bias 來自：

- 股票／股票日選擇；
- 日內時段組成；
- 同一股票內的事件 timing。

在目前 barrier-or-close 的長持倉研究中，它們是 secondary diagnostics。matched
反號表示 bias 來源不是局部 timing，不代表 event gate 對全體 eligible points
沒有可用的長持倉 bias。

未來 label 改成 5m／10m 時，必須建立新 label profile、升 study version，並選擇
與該 horizon 對齊的 control/matching bucket；屆時才可把 `same_stock_time` 設成
`primary_estimand`。不能沿用目前固定 5-minute matching 去裁決收盤 label。

## Stage 4：訓練與交易化

通過 shared-bias gate 的 condition 先形成 candidate universe，再使用：

1. prior beta 去市場成分；
2. event-to-exit 波動標準化；
3. strictly-prior rolling preprocessing/model；
4. prior-only threshold；
5. cost、capacity、position cap 與 executable-fill backtest。

event condition 負責提高候選分布品質；模型負責候選內排序；cooldown、持倉狀態與
資本限制負責把密集 state rows 轉成訂單。三者必須分層，不可把下單規則寫回
positive-control selector。

## 報告判定語言

長持倉 discovery 的主結論依序寫：

1. 是否找回 positive control；
2. 新 condition 的 shared long／short bias；
3. 跨期衰退與經濟量級；
4. 是否和既有事件提供不同股票日分布；
5. matched 歸因；
6. 下一階段是 event shaping、模型或交易化。

不得因 secondary matched estimand 較顯著，就把研究目標從「候選 bias
distribution」事後改寫成「局部 timing alpha」。
