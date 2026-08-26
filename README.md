# timeSeries：泛用日內事件研究框架

這個模組研究「任意可落在 `(Date, QuoteCode, ChannelSeq)` 的日內事件」，
不是 negFill 策略本身。框架要驗證兩件事：

1. event condition 能否從同一個 eligible control pool 切出穩定的 bias；
2. 這個 bias 經 prior beta 去市場成分、事件 horizon 波動標準化後，能否用
   prior-only rolling model 與 threshold 訓練成可交易排序。

negFill 只是第一個 reference adapter，用來找回已知 bias 並凍結歷史基準。
新的 event family 不應繼承它的 selector、空方方向、B1/A1 進場、1.08 停損、
16-cell 或可當沖 universe。

## 當前探索目標

目前優先找的是「最長持有到收盤、途中依方向別 label 執行 barrier」的多空事件。
event condition 的任務是從全體 eligible points 切出較有利的候選分布，因此正式
探索的 `primary_estimand` 是 `shared_pool`。same-stock 與 same-stock-time 是
bias 來源歸因；等 label 改成 5m／10m，才用對齊 horizon 的 matched estimand
重新定義主驗收。

開始找未知事件前，框架必須先通過 `FillLots_atLow < 0` 的 positive-control
recovery。先測原始 state，不得預先改成 state update、epoch entry、深度門檻或
下單 cooldown。完整研究順序與報告規格見
[DISCOVERY_PROTOCOL.md](docs/protocols/DISCOVERY_PROTOCOL.md)。

## 核心與 reference 的邊界

| 層 | 泛用契約 | reference profile／adapter |
|---|---|---|
| 事件 | `framework/events.py`：多來源 selector、canonical keys、manifest | `build_index.py` |
| 資料 | `framework/dataset.py`：row-preserving merge、顯式 universe、lineage | `data_loader.py` |
| 標籤 | `framework/labels.py`：方向、執行、exit、hedge、vol 分離 | `framework/profiles/` |
| 事件驗收 | `framework/condition_study.py`：共用 eligible labeled pool | `condition_benchmark.py` |
| 特徵／模型 | `framework/features.py`、`model.py`：有序 manifest、causal preprocessing | `time_series_research_model.py` |
| 門檻 | `framework/prior_thresholds.py`：prior-session quantile 與 audit | 研究自行提供 versioned spec |

詳細 API、最小範例與版本規則見 [FRAMEWORK.md](docs/FRAMEWORK.md)。negFill 的完整
資料契約、研究歷史與執行規格已獨立到
[NEGFILL_REFERENCE.md](docs/references/NEGFILL_REFERENCE.md)。

研究數字與判定不放在入口文件；由 [report index](report/README.md) 進入對應的
versioned study report。`test/` 保存可重現的研究程式，`report/studies/` 保存結論，
兩者以相同的日期主題 id 配對。

## 核心資料流

```text
EventIndexSpec(selector + version)
  → canonical event keys
  → EventDatasetSpec(universe + source/alignment contract)
  → DirectionalLabelSpec(raw → market component → residual → z)
  → FeatureManifest + causal rolling model
  → PriorQuantileSpec
  → cost/cap backtest
```

每一層都能獨立換 spec；事件名稱不會改變 merge、label、feature 或 threshold
的核心行為。

## 最小的新事件入口

```python
from pathlib import Path

import polars as pl

from src.research.timeSeries.framework import (
    DataRoots,
    EventIndexSpec,
    build_event_index,
)


def select_event(day):
    return (
        day.scan_tick_feature()
        .filter(pl.col("my_event_state"))
        .select("QuoteCode", "ChannelSeq")
    )


roots = DataRoots.from_project_root(Path("."))
spec = EventIndexSpec(
    event_id="my_event",
    selector_version="1",
    selector=select_event,
    index_path=Path("data/events/my_event/v1/index.parquet"),
    required_sources=("tick_feature",),
)
build_event_index(spec, roots=roots)
```

接著用 `EventDatasetSpec`、`DirectionalLabelSpec`、`FeatureManifest` 與
`PriorQuantileSpec` 組合研究；不要複製 compatibility adapter 再改條件。

## 框架的不變條件

- `side` 不再同時暗示方向、maker/taker 與出場規則；它們是獨立 spec，
  完整設定會產生 stable fingerprint。
- generic merge 預設不套可當沖、rank 或 negFill universe。
- tickBar 預設只接已完成的 `TimeSlot`；`legacy_stock_tick_time` 僅供重現。
- feature 只能由明確、有序、可 hash 的 manifest 宣告；不以 regex 掃 schema。
- causal preprocessing 只用 strictly-prior finite median；未來資料不能改寫過去。
- condition 與 control 先共用同一份 eligible、label-valid pool，再分流。
- 缺來源、缺 eligibility、重複鍵、schema/version 漂移都 fail closed。
- prior threshold 明確區分 calendar sessions 與 sparse bucket observations。

## 相容與凍結範圍

- `build_index.py`、`data_loader.py`、`condition_benchmark.py` 與
  `time_series_research_model.py` 是 reference／legacy adapters。
- `data/` 與 `baseline/timeseries_points.parquet` 是 frozen legacy artifacts；
  新 selector、merge、label、model 或 study 版本必須寫到新目錄，不能原地覆寫。
- frozen baseline 的契約與 hash 見 [baseline/BASELINE.md](baseline/BASELINE.md)。
