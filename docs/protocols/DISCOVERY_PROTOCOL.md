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

### 門檻參數化與樣本比例

數值門檻不能只記 cutoff；每個 study 還要固定方向、reference population、window
basis、lookback、minimum support 與參數的經濟語意。先依語意選擇以下模式，再用
strictly-prior 分布診斷確認，不得一律 percentile 化：

- `raw`：符號、離散狀態、合法 tick、bounded／normalized 值，或具有固定經濟單位且
  coverage 穩定的門檻；
- `relative`：無界且容易隨市場尺度漂移，或研究問題本來就是取 prior top／bottom q%；
- `hybrid`：同時保留固定經濟 floor／cap 與 strictly-prior relative scale。

relative lower-tail 例：

```text
theta[t] = Q_q(x | target session t 之前 N 個 calendar sessions)
event[t] = x[t] <= theta[t]  # lower-tail example
```

這讓參數表達「取 prior lower q%」，但不能取代有經濟意義的 raw threshold，否則可能
在低量 regime 強迫選入沒有實質量級的樣本。selection rate、active days 與 support 是
所有模式的基本樣本描述；implied percentile 則只在 scale-bearing feature、懷疑尺度
漂移，或需要驗證 raw cutoff 是否仍保有原語意時才是必要 audit。

若同一 raw cutoff 在不同時段從 5% tail 漂到 20% tail，這是診斷訊號，不是自動失敗。
需判斷它是 condition 要捕捉的真實 regime prevalence、名目尺度／denominator／資料定義
漂移，或已破壞原本 selector 語意與 minimum support。第一種可保留 raw 並分 regime
報告；後兩種才考慮 normalization、relative 或 hybrid，且升 selector version。

多條件 conjunction 要另報 joint prevalence、active days、stock-days 與 events/day；
不能用各欄位 marginal percentile 相乘推定。prior support 不足時 fail closed，不以
全樣本或未來分布補值。

### 累積狀態的 fresh sampling

累積 state 與 fresh event 是兩個不同契約。對 `FillLots_atHigh／Low` 類狀態，以平均
depth 正規化只能改變尺度，不能避免同一 persistent state 被 unchanged-price rows 重複
計權。若研究要抽 fresh events，必須事前指定可觀察的新資訊，例如：

- B1／A1 的 MD price change，並明確定義同一 epoch 是否只取新價對 first visit；或
- 當筆 formal `FillLots` 達到 D-1 side-specific `big_buy_lots／big_sell_lots` 等大單
  reference，可再搭配固定最低張數或 D-1 `avg_bidLots1 + avg_askLots1` 尺度。

這裡判斷的是「當筆成交是大單」，不是含糊的「累積到足量新增 FillLots」。若要測
state delta、crossing 或 episode，需在完整 raw stream 上處理 epoch reset 並建立新的
selector version。D-1 liquidity reference 必須有有效 support／lineage；缺值或 producer
fallback 不得默認為合法大單尺度。price-change、big-fill 與兩者聯集是不同 hypothesis，
應事前凍結或分開報告。

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

condition 的 deployment gate 驗收 causal sample definition、prevalence、coverage、
bias 與 learnability 的跨期穩定，不要求 condition 單獨扣成本後獲利。selector 版本或
rolling parameter path 凍結後，模型才可在該 candidate universe 內用 strictly-prior
資料排序；必須另報 downstream model 相對 all-condition baseline 的 outer-OOS lift。
模型 PnL 不得替不穩定的 selector、孤立 threshold peak 或漂移的樣本比例背書。

## Stage 5：多條件 selector 的 deployment 參數驗收

多條件 selector 在 promotion 前必須事前選定以下一種契約。grid、primary objective、
樣本下限、lookback、refit cadence、cost/capacity 口徑與 pass/fail 門檻，都要在 outer
evaluation 前凍結。

### A. 固定全域參數：驗證穩定 plateau

- 參數只能由 development window 選出；若使用全歷史選參數，整段都是 in-sample，
  deployment 證據只能來自之後的新資料。
- 除 nested ablation 與逐參數 ladder 外，還要評估所選參數向量周圍事前定義的低維
  joint neighborhood；只做 one-factor-at-a-time 不能排除 interaction cliff。
- 每個組合都報 equal-period effect、raw／residual BP、joint prevalence、active days、
  events/day、年份／regime、capacity 與 downstream-model learnability。
- 選擇 plateau 內部的代表參數，不選邊界上的單一最高點。孤立尖峰、鄰近組合翻號、
  樣本量崩塌或只有精確 winner 有效，都不得 promotion。

### B. Rolling 參數：驗證下一期預測力

每個 target day／week／block 都執行 nested rolling evaluation：

1. outer target 之前的資料才可進 inner training／validation、preprocessing、參數排名與
   hyperparameter selection；保存 `trained_until`。
2. target 開始前鎖定參數。下一個完整 day／week／block 是剛更新後的 first-live outer
   period；不得用該期結果回頭修改該期參數。
3. 串接所有 outer periods 才形成 deployment estimate。daily refit 時每一天都是新的
   first-live day；較低頻 refit 時整個下一 block 都沿用期初凍結參數。
4. 每個 refit 先用 prior objective 排序 frozen candidate family，再將 prior top
   10--20%、middle、bottom 10--20% 的代表 cohort 原封不動帶到下一期評估，不能依
   next-period outcome 重分組。family 太大時可使用事前固定的 correlation grouping／
   deterministic stratified representatives，但 selected winner、local neighbors、分組法與
   effective-trial 算法都必須在下一期結果前鎖定。

optimizer 不能只報 winner path。至少報：

- next-period top／middle／bottom 的 equal-period 表現與 uncertainty；
- top-minus-bottom spread、正號 outer periods 比率；
- candidate 數足夠時，prior rank 對 next-period performance 的 Spearman rank IC；
- 每期 `top > middle > bottom` 的 monotonicity rate；
- top cohort dispersion、selected winner 相對 top-cohort median 的差距；
- parameter turnover、coverage／prevalence drift 與每組 next-period support。

aggregate 必須呈現方向一致的 top > middle > bottom，且 top-bottom spread 與可估計的
rank IC 具有穩定方向。若精確 winner 特別好但 top cohort 離散很大，或 middle／bottom
沒有順序，代表 optimizer 尚未證明市場延續性；正的 winner 回測也不得 promotion。
實盤合理預期應接近 prior top cohort 的中間狀態，而不是事後 oracle peak。

### 搜尋範圍與 selection bias

事前指定的單一 confirmatory condition 不需要虛構 grid。若研究在同一歷史資料上比較
多個 trial 再選 winner，則需保存可重建 candidate family 的 frozen generator／registry
與 trial ledger，並在看 outer 結果前指定和 objective 相容的 selection-bias audit：

- 已形成共同日／block return series、以 Sharpe 選擇的 strategy family，可使用
  DSR-like、CSCV-PBO 或 moving-block familywise reality check；
- `delta_z`、bias uplift 等非 Sharpe objective，使用 block-bootstrap max statistic、
  family correction 或其他與 estimand 相容的方法，不硬套 DSR。

至少報 raw trial count、dependence-adjusted effective trial count、dependence／grouping
估法、across-trial dispersion、adjusted statistic 與合理敏感度。DSR-like 指標可衡量
multiple-testing 與非常態造成的績效膨脹，但不能取代 causal audit、selected point 的
local plateau、rolling first-live 排序力、成本容量或 untouched future data。

### 最小 machine-readable artifact

不要求每份研究永久保存所有 row-level 中間檔或實體 dense grid，但必須保存足以重建
搜尋範圍、估計 selection bias 並重做 deployment 判定的 artifact：

- frozen candidate generator／grid version 與 trial ledger；每個 trial 至少有 fingerprint、
  family、objective、evaluation window、selected flag、support、active／zero-event period
  數與 candidate-level outcome summary；
- 固定參數 study 的 selected candidate 與事前 local neighborhood 逐期 sample support／
  performance；
- rolling study 的 outer period × candidate prior rank／support／next-period outcome，或可
  確定性重建它們的 registry、代表 cohorts 與 effective-trial artifact；
- threshold mode／raw value，以及 scale diagnostic 適用時的 implied percentile；
- 獨立 deployment decision，逐條列出 gate 與失敗原因。

requested-period calendar 必須完整表示，但不必為每個 trial 實體寫出所有零值 row。
資料完整但無事件的策略期是 flat PnL `0`；condition effect 則是 undefined 並報 support
`0`。兩者都不得被刪除，也不得和 ineligible、missing source 或 run failure 混為一談。

現有 causal rolling model 與 `PriorQuantileSpec` 可提供 strictly-prior prediction 與
score threshold audit，但不等於已有 generic selector-parameter optimizer 或上述
next-period rank-monotonicity evaluator；在共用 API 建立前，必須用 study-local versioned
實作並保留 lineage。通過本節只代表可以準備 deployment handoff，仍不得在沒有使用者
明確要求時修改正式 `build_index.py`、`data_loader.py` 或 production model。

## 報告判定語言

長持倉 discovery 的主結論依序寫：

1. 是否找回 positive control；
2. 新 condition 的 shared long／short bias；
3. 跨期衰退與經濟量級；
4. 是否和既有事件提供不同股票日分布；
5. matched 歸因；
6. fixed plateau 或 rolling first-live 參數驗收；
7. 下一階段是 event shaping、模型、shadow 或 deployment handoff。

不得因 secondary matched estimand 較顯著，就把研究目標從「候選 bias
distribution」事後改寫成「局部 timing alpha」。
