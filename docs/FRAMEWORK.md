# intradaySignal 研究 framework 使用方式

本文件只描述事件研究契約。正式 negFill 點位仍由
`src/research/timeSeries/build_index.py`／`data_loader.py` 維護；framework
不得寫入該目錄，也不得替它產生 manifest。本模組舊的同名入口已於 2026-09-21
移除，將由多訊號 condition registry 與共用 daily store 取代。

框架把一條研究線拆成可獨立版本化的六份契約：event、dataset、label、
feature、model、threshold。事件名稱可以是 negFill、成交量突變、盤口失衡或
其他條件；核心只要求事件能落到 `(Date, QuoteCode, ChannelSeq)`。

## 1. 定義事件，不在 builder 寫死條件

Selector 會收到單日所有 canonical source 的 lazy access，可依需要讀一份或
多份資料。輸出只需 `QuoteCode`、`ChannelSeq`；框架負責加 Date、正規化型別、
排序、唯一性、來源 identity、manifest 與 atomic publish。

```python
from pathlib import Path
import polars as pl

from src.research.intradaySignal.framework import (
    DataRoots,
    EventIndexSpec,
    build_event_index,
)


def select_my_event(day):
    ticks = day.scan_tick_data().select(
        "QuoteCode", "ChannelSeq", "TransTime", "FillLots"
    )
    pre_market = day.scan_pre_market().select("QuoteCode", "day_amount_rank")
    return (
        ticks.join(pre_market, on="QuoteCode", how="inner")
        .filter((pl.col("FillLots") >= 100) & (pl.col("day_amount_rank") <= 200))
        .select("QuoteCode", "ChannelSeq")
    )


roots = DataRoots.from_project_root(Path("."))
artifact_root = Path(
    "src/research/intradaySignal/experiments/20260826_large_trade/"
    "artifacts/large_trade/v1"
)
event_spec = EventIndexSpec(
    event_id="large_trade",
    selector_version="1",
    selector=select_my_event,
    index_path=artifact_root / "index.parquet",
    required_sources=("tick_data", "pre_market"),
    calendar_source="tick_data",
)
build_event_index(event_spec, roots=roots)
```

改 selector 語意時要升 `selector_version`，不可沿用同一版本覆蓋。

## 2. 建立事件資料集

Generic merge 預設保留全部事件列、要求必要來源、使用 completed-slot 的因果
tickBar 對齊，並記錄每個 partition 的 index/source identity。Universe 是
獨立的 `UniverseSpec`；callback 只能選列，不能竄改欄位值。

```python
from src.research.intradaySignal.framework import (
    EventDatasetSpec,
    MergeOptions,
    build_event_dataset,
)

dataset_spec = EventDatasetSpec(
    event_id="large_trade",
    dataset_version="1",
    index_path=event_spec.index_path,
    output_dir=artifact_root / "daily",
    merge_options=MergeOptions(),
)
build_event_dataset(dataset_spec, roots=roots)
```

`legacy_stock_tick_time` 只供歷史重現。新研究不可把 legacy 與 causal partition
混在同一個目錄；manifest 會拒絕這種設定漂移。
`preserve_tick_data_column_order` 也是明列在 manifest 的相容政策；generic
預設關閉，negFill legacy profile 開啟以逐欄重現既有 parquet schema。

## 3. 標籤：方向、執行、出場、hedge、vol 各自表達

`DirectionalLabelSpec` 不使用模糊的 `side` 推導整套交易語意。輸出欄名固定：

- `label_entry_price`、`label_exit_price`
- `label_raw_bp`
- `label_market_component_bp`：`beta × factor return`，尚未做方向調整
- `label_hedge_pnl_bp`：真正 signed hedge leg
- `label_residual_bp = raw + hedge PnL`
- `label_vol_bp`、`label_z`、`label_valid`

```python
from src.research.intradaySignal.framework.labels import (
    Direction,
    DirectionalLabelSpec,
    EntrySpec,
    ExitSpec,
    HedgeSpec,
    LiquidityRole,
    VolSpec,
    add_centered_labels,
    add_directional_labels,
)

label_spec = DirectionalLabelSpec(
    profile_id="large_trade.long_taker_close.v1",
    direction=Direction.LONG,
    entry=EntrySpec("AskPrice1", LiquidityRole.TAKER),
    exit=ExitSpec(fallback_price_col="Close"),
    hedge=HedgeSpec(),
    normalization=VolSpec(),
)
labeled = add_centered_labels(add_directional_labels(events, spec=label_spec))
```

`label_z_within_stock_day` 是選時點 label；`label_z_within_day` 是選股 label。
任何 entry/exit/beta/factor/vol 無效的列會得到 `label_valid=False`，不產生 inf。
`DirectionalLabelSpec.fingerprint` 涵蓋所有會影響輸出的經濟語意；研究與模型
artifact 除了 `profile_id` 也應保存這個 hash，避免同名 profile 被原地改寫。

## 4. 事件研究：condition/control 只分流一次

`EligiblePoolSpec` 版本化靜態 eligibility。框架先建立同一份 eligible、
label-valid pool，再以 event keys semi-join condition，並從同一 pool 取 control。
這避免 event parquet 與 control 現算資料使用不同 beta、vol、filter 或
complete-case cohort。`control_sample_size=None` 使用完整 pool；探索多個候選時，
完整 shared control 是預設建議，能移除每個候選各抽一次 control 的 Monte Carlo
差異。

同一 event family 的候選應先用完整、按事件鍵排序的來源 stream 計算 lag、change
與 onset，之後才和 eligible pool 交集；不能先套時間、universe 或 label validity
再 shift。`compare_conditions_to_shared_control()` 會一次驗證多組 key、一次 join
pool，並讓所有候選共用完全相同的 control。候選可重疊，但每個候選內的 canonical
key 必須唯一。

如果 derived feature table 省略了 raw rows，不能先 inner join derived table 再算
lag 或 epoch；應從最早的 causal source 重建狀態，或先在完整 raw stream 上算完
歷史狀態後才接 derived 欄位。否則每個股票的第一個保留 row 可能被誤判成 onset，
或漏掉第一個真正的 state update。

完整 shared-pool control 回答的是「這個 condition 是否切出不同的 bias」，會同時
保留選到哪些股票、哪些時段與日內 timing 的效果。對目前 barrier-or-close 的
長持倉 event discovery，這正是主要研究問題；`ConditionComparisonSpec` 應固定
`primary_estimand="shared_pool"`，股票與時段組成本身是 event gate 可使用的選擇，
不能因 matched 結果反號就否決 shared bias。若要把結果解讀成 timing event，
研究 adapter 應另報 matched estimand，例如先在同一 `Date × QuoteCode` 比較，並再
以固定的 causal clock bucket 做 `Date × QuoteCode × time bucket` 比較。Matching
欄位、bucket 寬度、每層權重及 control population 都是 study contract 的一部分，
必須進 manifest；不能用 shared-pool 顯著性代替 timing 證據。

反過來也不能以 matched 顯著性事後改寫原研究目標。same-stock／same-stock-time
在目前是 secondary attribution diagnostics；label 改成 5m／10m 並建立相應版本的
control/matching contract 後，才把對齊 horizon 的 matched estimand 設為 primary。

探索新 feature 前，先依 [DISCOVERY_PROTOCOL.md](protocols/DISCOVERY_PROTOCOL.md) 執行
positive-control recovery。`FillLots_atLow < 0` 必須先以完整 state condition 通過
同一 pool/label pipeline；state update、onset、epoch、threshold 與 cooldown 是
後續 event-shaping／execution 問題，不能取代第一階段的 recovery test。

若 matched baseline 使用同一個完整 eligible pool，它會包含 condition rows；這是
`event mean − unconditional pool mean` estimand，不是 event-vs-non-event causal
control。在單一 matched cell 內，self-inclusion 只會以 `(1 - event share)` 衰減
disjoint-control contrast；跨 cell 聚合時各 cell 的衰減率不同，因此不能宣稱
aggregate 一定不翻號。若研究問題需要 disjoint control，必須另立並版本化 control
population，不能只改報告措辭。

可恢復執行的 daily cache 也屬於 artifact contract。Cache identity 必須涵蓋當日
實際讀取的 raw source identities、event family/version、candidate subset、pool、
label fingerprint、control mode 與 matching 設定；只用日期或來源路徑不足以偵測
同名 parquet 被原地替換。HAC lag 的單位也要寫清楚；本框架 report 的 lag 是移除
null 後依日期排序的 active observations，不應寫成 calendar days。

參考命令：

```bash
uv run python src/research/intradaySignal/tools/condition_benchmark.py \
  --event-index src/research/intradaySignal/data_index.parquet \
  --label-profile short-taker-stop108-v1 \
  --start 20250211 --end 20260817 --step 3 --sample 25000
```

這支 CLI 是 negFill calibration adapter；新的 event 應建立自己的 pool spec，
再呼叫 `build_eligible_labeled_pool()` 與 `compare_condition_to_control()`；批次探索
則呼叫 `compare_conditions_to_shared_control()`。事件 family、pool、label 與 control
設定都必須分開保存版本與 fingerprint。

## 5. 特徵與 rolling model

模型只能讀 `FeatureManifest` 的明確有序欄位。欄位順序、role、missing policy、
availability 都進 SHA-256 fingerprint；regex 或 dataframe schema discovery 不屬於
模型契約。`causal_v2` 的缺值只使用 strictly-prior finite daily median，無歷史時
用 manifest 固定 fallback；未來追加資料不能改寫歷史輸入。

```python
from src.research.intradaySignal.framework import (
    FeatureManifest,
    FeatureSpec,
    train_and_predict_causal_v2,
)

features = FeatureManifest(
    name="large_trade",
    version="1",
    features=(
        FeatureSpec("imbalance"),
        FeatureSpec("prior_vol", availability="pre_market"),
    ),
)
result = train_and_predict_causal_v2(
    pandas_frame,
    features,
    Y_col="label_z_within_day",
    n_training_days=20,
    alpha=0.1,
)
```

`src/research/rolling_model.py` 的預設仍是 `legacy_v1`，讓舊 notebook 可重現；
新研究應透過 wrapper 明確使用 `causal_v2`。

Wrapper 預設 estimator 仍是 Ridge。若研究要做同一 rolling／feature contract 下的
ElasticNet ablation，可顯式傳入 `model_family="elastic_net"`、`elastic_net_l1_ratio`
與 `elastic_net_alpha_mode`。`ridge_l2_equivalent` 會依每次 fit 的樣本數換算 sklearn
兩種 objective 的尺度，使 L2 係數對齊原 Ridge alpha，再增加 L1；這只是一個固定
model-family 診斷，不是 alpha／l1-ratio 自動選型。模型 family 與 penalty 設定必須
進 study manifest，不能用同名 artifact 覆寫 Ridge 結果。

## 6. Prior-only threshold

正式契約建議使用 `calendar_sessions`：前 N 個全市場交易日內沒有該 bucket，也
算窗口的一天。`bucket_observation_sessions` 保留歷史 notebook 的「往前找 N 個
有該 bucket 的日子」語意。兩者不能混稱 prior-3-day。

```python
from src.research.intradaySignal.framework import (
    PriorQuantileSpec,
    WindowBasis,
    build_prior_session_thresholds,
)

threshold_spec = PriorQuantileSpec(
    score_col="label_z_within_day_pred",
    threshold_col="threshold",
    q=0.9,
    bucket_cols=("time_bucket", "price_bucket"),
    lookback_sessions=3,
    window_basis=WindowBasis.CALENDAR_SESSIONS,
)
audit = build_prior_session_thresholds(
    scores,
    sessions=trading_calendar,
    spec=threshold_spec,
)
```

Audit table會保存 history start/end、window sessions、實際 support sessions、
observations、quantile、window basis 與 score unit。缺門檻時 gate 為 False。

## 7. negFill reference 與正式輸出的邊界

- `framework/profiles/negfill.py` 保存研究校準用的 negFill selector、short label、
  feature manifest 與 merge factories；它不是正式入口的 runtime dependency。
- 共用台股 universe 在 `profiles/tw_equity.py`；benchmark long 與 FHL maker
  labels 分別在 `profiles/benchmark.py`、`profiles/fhl.py`，不屬於 negFill。
- 本模組舊的根目錄入口（`build_index.py`、`data_loader.py`、`data/`、
  `data_index.parquet`）已移除；正式 negFill 點位仍在
  `src/research/timeSeries/`，重建完成前不得從本模組寫入。
- 新版 reference merge 必須建立新的 `EventDatasetSpec`，並將 `index_path` 與
  `output_dir` 指向 study-local versioned artifact 目錄或 `/tmp`。
- `baseline/timeseries_points.parquet` 是 frozen legacy artifact；修改框架、
  文件或測試都不得原地重建它。新的結果以新的 dataset/model/study version
  並存，先做 parity／差異歸因後才可宣告取代。

## 8. 多訊號：SignalConfig 與四層

`framework/registry.py` 放契約（每層一個 Spec + `SignalConfig`），`layers/` 放四層邏輯，
`signals/<name>.py` 只宣告一個 `SignalConfig`，`config.py` 註冊。

```python
from src.research.intradaySignal.framework.registry import GateSpec, RollingSelection, SignalConfig
from src.research.intradaySignal.signals._short_common import short_day_selector, short_model_spec, short_point_filter_spec

CONDITION = GateSpec(
    gate_id="my_condition", version="1",
    params={"rank_max": 150},                # 預設值；不在 grid 裡的參數固定
    grid={"rank_max": (100, 150, 200)},      # 要 rolling 的參數；空 dict = 固定
    apply=my_condition_fn,                   # (pl.DataFrame, params) -> pl.Series[bool]
    stage="condition",
)
MY_SIGNAL = SignalConfig(
    st_name="my_signal", version="1", stage="research",
    # 1 envelope + condition
    envelope=my_envelope,                    # EventDaySources -> keys（可帶 envelope_columns）
    envelope_required_sources=("tick_feature",),
    envelope_columns=(),                     # envelope 算出、tickFeature 沒有、condition 要用的欄
    universe=NEXTDAY_ALLOW_DAY_TRADE_UNIVERSE_V1,
    gate=CONDITION,
    selection=RollingSelection(window_days=60, objective="stockday_mean_label"),
    selection_pool=my_label_fn,              # 候選 rows -> 帶 label 的 rows，只給選參評分
    # 2 固定 tick 過濾（無 grid；None = 不過濾）
    point_filter=short_point_filter_spec("my_points", md_l1rate_min=0.2),
    # 3 學習型 stock-day gate（None = 不用）
    day_selector=short_day_selector("my_day", exclude_share=0.35),
    # 4 模型（None = 沒有模型層）
    model=short_model_spec(),
)
```

- `layers/envelope.py`：`build_envelope_index`／`build_store`（在 `framework/store.py`）→
  `build_condition_params`（每個 store 日對 grid 全部格子算 per (Date, param_set, QuoteCode) 聚合並快取；
  評分池先套 point filter；target day 的分數只用 strictly-prior `window_days` 個 store 日；保留全 grid
  分數與 `is_winner`）→ `build_signal_index`（`params_for_date`：當日 → 最近一次 winner → 預設）。
  `evaluate_rolling_selection` 是「每日微調有沒有價值」的判定器。
- `layers/point_filter.py`：`build_points` 對 index 每日套 `point_filter.mask`，寫 keys；`load_rows`
  是第 3、4 層共用的候選列讀法（store ∩ index ∩ points）。
- `layers/day_selector.py`：`build_stock_day_table`（candidate rows → `spec.label` → 每 stock-day 的
  D-1 特徵與 target 均值）→ `build_day_scores`（strictly-prior 視窗重訓、依 `exclude_share` 切、
  `hard_rules.mask` 永遠擋）→ `apply_day_scores_pl/pd`。
- `layers/model.py`：`build_signal_frame`（long format，`signal_id`；列已過 day selector）→
  `train_by_signal`、`threshold_by_signal`、`select_by_signal`；模型引擎仍是
  `src/research/rolling_model.train_and_predict_ridge_rolling`。
- 每個產物有 manifest 記定義 hash；定義變了自動重建，上游變了下游跟著重做。
