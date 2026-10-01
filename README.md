# intradaySignal：多訊號日內事件架構

所有日內訊號共用同一條四層流程；新增一個訊號只要新增 `signals/<name>.py`（一個
`SignalConfig`，四段設定）並在 `config.py` 註冊，不再各自維護一套管線。

## 四層

```
1 envelope + condition  →  2 point_filter  →  3 day_selector  →  4 model
   (index)                    (points)            (day scores)       (thresholds → selection)
```

| 層 | 做什麼 | 設定段 | 模組 | 產物 |
|---|---|---|---|---|
| 1 envelope + condition | envelope：無參數的事件定義，決定哪些 tick 進 store（改語意就升版）。condition：可調的子集，參數逐日由 strictly-prior 60 天視窗從凍結 grid 選出；評分池已先套第 2 層過濾 | `envelope` / `gate` / `selection` / `selection_pool` | `layers/envelope.py`（store 合併在 `framework/store.py`） | `envelope/<st>_index`、`store/YYYYMMDD`、`params/<st>_condition`、`index/<st>_index` |
| 2 point_filter | 固定不要的 tick：時間窗、試撮、Bid1>0，加訊號自己的 tick 過濾（negfill：OTC\|mdRate、elapsed、rank）。無 grid | `point_filter`（`GateSpec`, stage=`point_filter`） | `layers/point_filter.py` | `points/<st>_points` |
| 3 day_selector | 學習型 stock-day 風險過濾：幾個簡單模型各自預測一個 `ds_*` 目標（明日停損機率、嘎高幅度、開盤在日內區間的位置），每天用前 60 天的股票日重訓，樣本權重 = clip(昨日成交額 × 0.3% / 500 萬, 0, 1)，各自用絕對門檻多砍；`hard_rules`（價位 3000）不管模型都擋。七條舊規則只當 `baseline_rules` 比較用 | `day_selector`（`DaySelectorSpec` + `RiskModelSpec`） | `layers/day_selector.py` | `params/<st>_stock_days`（cache）、`params/<st>_day_scores` |
| 4 model | label、特徵、rolling Ridge A(60d)/M(20d)、16-cell prior-3-day 門檻；只在前三層留下的列上訓練與定門檻。`use_model_gate=False` 的靜態 book 只算診斷、全數放行 | `model`（`ModelSpec`） | `layers/model.py` | `model/<st>_selected` |

之後是 execution（部位簿記、收單時間、BackTest），在 notebook／trader，不屬於訊號定義。

原則：index 只放 envelope + condition，越寬越好；參數若只是在已 materialize 的 rows 上做子集 →
condition 或 point filter；若會選出 envelope 以外的 row → 升版 envelope。stock 級的刪除交給
day selector，tick 級的刪除交給 point filter，兩者不混。

## 目錄結構

```
intradaySignal/
├── config.py             REGISTRY（SignalConfig 們）與 artifact 目錄，唯一要改的註冊入口
├── signals/              一個訊號一個設定檔：negfill.py、fh_neg.py；_short_common.py 是 taker-short 共用區塊
├── layers/               四層各一個模組：envelope、point_filter、day_selector、model
├── framework/            契約與資料底層：registry（各層 Spec + SignalConfig）、store、dataset、events、features、labels、
│                         overlap（訊號相似度四把尺）；profiles/ 為 envelope selectors；其餘為舊 study 用（見下）
├── pipeline/run.py       唯一 CLI：run.py <envelope|points|day_selector|model|all>
├── artifacts/            產物（gitignore）：envelope/ store/ params/ index/ points/ model/
├── notebooks/            intraday_signal_research.ipynb（薄 driver）；legacy/ 為舊 timeSeries notebook
├── tests/                unittest（uv run python -m unittest discover -s src/research/intradaySignal/tests -t .）
├── docs/                 FRAMEWORK.md、protocols/、references/、notes/
├── experiments/ report/ baseline/ tools/ legacy/   舊 timeSeries 時代的 study 程式、報告、凍結資料與工具；
│                         只有它們還 import framework/{condition_study,prior_thresholds,profiles/benchmark|fhl|filllots_at_low}、tools/
```

## 每日流程

```bash
uv run python src/research/intradaySignal/pipeline/run.py all                      # 四層依序、增量
uv run python src/research/intradaySignal/pipeline/run.py envelope --next-date 20261001   # 隔日 condition 參數
uv run python src/research/intradaySignal/pipeline/run.py envelope --evaluate       # grid 的 rolling first-live 評估
uv run python src/research/intradaySignal/pipeline/run.py day_selector --evaluate   # 次日 realized z 依分數分桶
uv run python src/research/intradaySignal/pipeline/run.py model --last-n-days 230
```

每層都支援 `--only <st ...>`、`--output-root`（測試重導）。`envelope` 另有 `--start/--end`、
`--rescan`（重掃 raw envelope，需給範圍或 `--all-history`）、`--force`（重算參數聚合、重建 index 範圍）、
`--no-store`；`points` 有 `--start/--end/--force`；`day_selector` 有 `--family`、`--force-table`。
產物都帶 manifest：定義 hash 變了自動整段重建，index 變了 points 跟著重做，points 變了 stock-day 表跟著重做。

研究端用 `notebooks/intraday_signal_research.ipynb`：`build_signal_frame`（long format，`signal_id`
區分訊號，列 = store ∩ index ∩ points ∩ day selector 保留的 stock-day）→ `train_by_signal` →
`threshold_by_signal` → `select_by_signal` → 部位簿記 → BackTest。

## 資料來源政策

特徵一律讀 `/media/kevin/SSD2/Data` 交付的 tickData／tickFeature／preMarket／brenchFeature 欄位，管線不重算來源已有的值。
2026-09-30 檢核後改掉的：`ToRef`、`ToOpen`、`TickBP` 改讀 tickFeature（modelUpdate 自己用 Bid1 重算 ToRef，與來源的
Mid 口徑中位差 0.17%）。negfill 因為要與正式 modelUpdate 同門檻，保留 Bid1 口徑，但以明確名稱 `ToRef_bid1`／`ToOpen_bid1`
材料化（`to_ref_source="bid1"`），來源 `ToRef` 原樣不動；fh 用來源口徑。day selector 特徵改用來源的 `top1_net_buy_share`、
`trading_turnover`、`turnover_rate`、`avg_bidLots1/avg_askLots1`，不再自己算比率。

管線仍自己算、且有意為之的：
- 策略 label：`TakerSell_CloseBP` 加 1.08 停損（modelUpdate 契約；會覆蓋 tickFeature 同名、無停損的欄位）、TXF beta 殘差與 `label_residual_bp`。
- 模型特徵轉換（`short_features_pd`，modelUpdate 契約）：`B1_B12`、`A1_A12`、`B45_AB45`、`MD_ElaspeTime_30_re` 等；來源沒有這些欄，長期應落地到 tickFeature registry 再回補，不在讀取時算。
- merge 時算的 TXF 欄（`txf_to_1330_bp`、`txf_residual_vol_to_1330`），與 timeSeries data_loader 相同。
- envelope 狀態欄（fh：`pair_changed`、`high_time`、`visit_index`、`large_fill`）與 day selector 的 `ds_*` 目標、樣本權重、`log_trading_turnover`。

day selector 實驗紀錄（2026-09-30，negfill，2026 避險 BackTest，四層順序）：舊「z 回歸砍底部 46%」比七條規則少 500 萬、風報比腰斬；
風險 target（停損機率／嘎高幅度／開盤位置）+ 淺層樹 + 訓練窗分位數門檻，能砍到規則漏掉的負值組、比不濾商品好（Sharpe 2.49→2.77），
但留下規則會砍的那群（21 檔/日、6 bp），Sharpe 2.8 對規則 3.4、MDD 692 對 408。結論：negfill 規則放第 2 層、不用 day selector；
框架保留 `DaySelectorSpec`（多模型、樣本權重、absolute／window_quantile 門檻、additive）給 fh 與後續 target（損益尾巴）實驗。

## 已註冊訊號

| st_name | stage | 1 envelope／condition | 2 point_filter | 3 day_selector | 4 model | 檔案 |
|---|---|---|---|---|---|---|
| `negfill` | deployed | `\|Spread.diff\|>0 & \|BidPreMove\|>0 & FillLots_atLow<0`；condition = `ToRef_bid1` ∈ (−1.5%, 5%) & rank≤150（固定，grid 空；Bid1 口徑與正式 modelUpdate 一致，明確命名不覆蓋來源 `ToRef`） | 時間窗 09:00:30–12:30、試撮、`OTC\|mdRate>.2`、`elapsed>.1`、`rank≤100`，**加七條日級規則**（R1、市值、分點三條、價位 3000、maker spread） | 無（學習型 selector 對規則沒有風險調整後的優勢，見下） | 14 特徵 Ridge A/M（`ToRef_bid1`/`ToOpen_bid1`）+ 16-cell 門檻，**qA 0.05**（四層順序下 0.30 會在已過濾列上再砍 30%）；2026 BackTest ≈ 原 timeSeries 流程 | `signals/negfill.py` |
| `fh_neg` v3 | research | `FillLots_atHigh<0 &（SpreadPairID 相對前一筆改變 \| 大單）& −1.5%<ToRef<7%`，帶 `pair_changed/large_fill/visit_index/high_time`；condition = rank≤150 + 距高點 ≤5 分鐘 & ToOpen ≥ 0，grid 時間 {5,7.5,10,15} × ToOpen {不限,+1%,0,−1%,−2%} 每日 rolling | 時間窗、試撮（執行過濾全關） | 同 negfill：三個風險模型 + 價位 3000 | 靜態 book（`use_model_gate=False`），A/M 只當診斷 | `signals/fh_neg.py` |

`stage` 只是標籤：`research` 訊號照樣跑四層做研究，但正式端只發布 `deployed`。

`framework/overlap.py::overlap_report` 是訊號相似度的四把尺（exact tick、stock-day、±N 秒鄰近、日 label 相關 + disjoint 子集），
任一層的兩組點位都能比；notebook 3b 有範例。

## 新增訊號

1. `signals/<name>.py`：一個 `SignalConfig`，四段——`envelope`／`gate`（`GateSpec` 含預設參數與 grid）／
   `selection`／`selection_pool`；`point_filter`（無 grid 的 `GateSpec`，可 None）；`day_selector`
   （`DaySelectorSpec`，可 None）；`model`（`ModelSpec`，可 None）。taker-short 類直接用
   `_short_common.py` 的工廠：`short_point_filter_spec`、`short_day_selector`、`short_model_spec`。
2. `config.py` 的 `REGISTRY` 加一行。
3. `run.py all --only <name>`；store 會自動 patch，不重建歷史。

`signals/negfill.py` 是逐字對照 `src/strategy/negFill/modelUpdate.py` 的參考實作（差異只有層的順序）。

## 與正式 timeSeries 的邊界

- **正式 negFill 點位仍在 `src/research/timeSeries/`**（`daily_pipeline.sh` stage、
  `src/strategy/negFill/modelUpdate.py` 消費），本模組不得寫入該目錄。
- 本模組的 store 對 negfill rows 與 `timeSeries/data` 同日檔逐欄相等（含 brenchFeature 26 欄），
  第 1 層 index 與 modelUpdate 的 pre-gate keys 相等；第 2–4 層改為先過濾再訓練，與正式 modelUpdate 的訓練集不再逐字相等（切換正式路徑時以 `--output_dir` 重新對點）。
- `src/research/negFill` 為凍結 legacy，已無任何引用；不得新增。
- `framework/` 研究任意可落在 `(Date, QuoteCode, ChannelSeq)` 的日內事件，study 輸出只進
  `experiments/<study_id>/artifacts/` 或 `/tmp`。API 見 [FRAMEWORK.md](docs/FRAMEWORK.md)；negFill 完整契約見
  [NEGFILL_REFERENCE.md](docs/references/NEGFILL_REFERENCE.md)。

研究數字與判定不放在入口文件；由 [report index](report/README.md) 進入對應的
versioned study report。`experiments/<YYYYMMDD_topic>/` 保存可重現的研究程式與 study-local artifacts，`report/studies/<同 id>/` 保存結論。

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

- 正式營運入口在 `src/research/timeSeries/`，本模組不得寫入它的
  `data_index.parquet` 與 `data/`；本模組自己的 `envelope/`、`store/`、`params/`、
  `index/`、`points/`、`model/` 由 `pipeline/run.py` 增量維護，不進 git。
- 只有 `baseline/timeseries_points.parquet` 是 frozen legacy artifact。
- 新 selector、merge、label、model 或 study 版本必須寫到 study-local 新目錄，
  不能原地覆寫正式輸出。
- frozen baseline 的契約與 hash 見 [baseline/BASELINE.md](baseline/BASELINE.md)。
