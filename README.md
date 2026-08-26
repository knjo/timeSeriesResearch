# timeSeries：正式點位管線與泛用日內事件研究框架

這個模組有兩條互相獨立的路徑：

- **正式點位**：根目錄 `build_index.py` 與 `data_loader.py` 維護已確定使用的
  negFill event，逐日更新 `data_index.parquet` 與 `data/YYYYMMDD.parquet`。
- **泛用研究**：`framework/` 研究任意可落在
  `(Date, QuoteCode, ChannelSeq)` 的日內事件，輸出只能進 study-local versioned
  artifact 目錄或 `/tmp`。

泛用框架要驗證兩件事：

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

## 正式點位與研究框架的邊界

| 用途 | 入口 | 輸出 | 契約 |
|---|---|---|---|
| 正式事件 index | `build_index.py` | `data_index.parquet` | 固定 selector、依最新 Date 增量更新、不寫 manifest |
| 正式每日點位 | `data_loader.py` | `data/YYYYMMDD.parquet` | 依 index 逐日合併、不寫 manifest |
| 未來事件研究 | `framework/`、`framework/profiles/` | `test/<study_id>/artifacts/<event>/<version>/` 或 `/tmp` | versioned spec、lineage 與 manifest |
| 研究驗收／模型 | `condition_benchmark.py`、`time_series_research_model.py` | study-local artifacts | 不得改寫正式輸出 |

正式兩支入口不得 import `framework/`；framework 也不得把 artifact 指到上述正式
index 或 daily 目錄。`src/research/negFill` 只供查看既有外層管線形態，不形成程式
或資料依賴。

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
    index_path=Path(
        "src/research/timeSeries/test/20260826_my_event/"
        "artifacts/my_event/v1/index.parquet"
    ),
    required_sources=("tick_feature",),
)
build_event_index(spec, roots=roots)
```

接著用 `EventDatasetSpec`、`DirectionalLabelSpec`、`FeatureManifest` 與
`PriorQuantileSpec` 組合研究；不要修改正式 builder 來試尚未確定的條件。

## 正式點位的日常使用

```bash
uv run python src/research/timeSeries/build_index.py
uv run python src/research/timeSeries/data_loader.py
```

第一支只 append 比目前 `data_index.parquet` 最大 Date 更新的 tickFeature 日期；
第二支略過既有舊日、刷新最新既有日期，並建立新日期的 daily parquet。需要限定或
重建區間時使用 `--start YYYYMMDD --end YYYYMMDD`，並搭配 `--force` 才會覆寫
已存在的指定日期。

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

## 正式輸出與凍結範圍

- `build_index.py` 與 `data_loader.py` 是正式營運入口，不是 framework adapter；
  `data_index.parquet` 與 `data/` 會隨新交易日增量更新。
- 只有 `baseline/timeseries_points.parquet` 是 frozen legacy artifact。
- 新 selector、merge、label、model 或 study 版本必須寫到 study-local 新目錄，
  不能原地覆寫正式輸出。
- frozen baseline 的契約與 hash 見 [baseline/BASELINE.md](baseline/BASELINE.md)。
