# `fh_pos_breakout` maker 多單：fill gate + 模型 + 純多 BackTest

狀態：`scoped`  
Decision：`RESEARCH_CLOSED_NEGATIVE`（maker 進場路線關閉；不 promotion、不 handoff）  
研究日期：2026-09-16  
資料區間：2025-02-11 至 2026-08-10（366 sessions；受抗跌清單覆蓋限制）  
方向／進場：long，maker BidPrice1，訊號後 100ms 掛單，`makerFill` 佇列估計 `< 600s` 才算成交  
Label：`fhl.long_maker_tp109.v1`（1.09 停利否則 13:30 Close；β×TXF 與 residual vol 只作模型 target）  
BackTest：`BackTest(side='long', entry_price_col='BidPrice1', bp_col='label_raw_bp', fee=19.3)`，無避險

## 結論先講

1. **Maker 進場在這個事件上有嚴重的成交逆選擇，把「省 spread」全部吃掉還倒賠。**
   同一批 `fh_pos_breakout` 事件，掛 Bid1 立即成交的樂觀 maker label 全期 2025 `-7.6bp`／2026
   `+19.7bp`；但 600 秒內真的會成交的子集是 2025 `-40.8bp`／2026 `-23.3bp`，
   成交不了的那群是 `+77.4`／`+134.9bp`。成交越快越差（60s 內：`-44.7`／`-19.6`）。
   在相同可成交列上 maker 比 taker 剛好好一個 spread（~25bp），但可成交母體本身比全事件
   差 40～60bp，所以 maker 路線的淨結果**比 taker 全吃還差**。
2. **模型層沒有可持續的排序力。** timing 模型對 raw bp 的日 Spearman IC 開發期 `+0.13`、
   holdout `+0.05`；selection 模型 `+0.09 → +0.02`（holdout t=1.4）。四個 gate 在開發期的淨
   bp 幾乎相同（`-90 ～ -105`），模型沒有把 static 母體推向正值。
3. **書層全部為負；只有 top-list universe 在 2026 holdout 出現小樣本正值。** 事前規則選出
   的 `timing_only__top_list`：開發期 `-44.1bp`（344 筆）、holdout `+8.1bp`（347 筆、+4.6 萬、
   Sharpe 0.40）、全 model era `-15.5bp`；holdout 最好的 `selection_only__top_list` `+22.1bp`
   也只有 457 筆、開發期 `-100bp`。這是每日 2～3 筆、平均部位 <100 萬的切片，不能成書。
4. **不是短線鷹架害的。** 沿用的 static gate 砍掉 82% model-era 列（`SpreadPairElapsed>0.1`
   單獨砍 61%），但把這兩個空方 hygiene gate 拿掉之後 all-universe 仍 `-40～-74bp`、
   top-list holdout `-1～-12bp`。fill horizon 60/300/600s 與剔除 `undercut_immediate`
   的敏感度都同向。

因此：`fh_pos_breakout` 多方線的 maker 版本關閉。抗跌清單仍是唯一穩定的正向 universe 效果，
但它 +15～22bp 的平移填不掉 breakout 事件的成交逆選擇。

## 問題與事前口徑

- 問題：上一輪 taker 版 `fh_pos_breakout × in-top-list` 的高權重三分位接近損益兩平，若改
  maker 進場省 spread、再由 prior-only 模型排序，能否過 19.3bp？
- 事前決策（與使用者確認）：泛化 taker-mode BackTest 跑純多；成交假設用 `makerFill`
  佇列 gate；抗跌清單當執行層 filter 並進特徵，模型訓練在全部事件；不避險，label 的
  β/vol 調整只作模型 target。
- 事前凍結：selector、label、fill horizon 600s、q=0.30/0.10、static gate、capacity 全部沿用
  `20260826_filllots_at_high_signal`；變體選擇只用開發期（model_start～2025-12-31）Sharpe。
- 這是 retrospective；holdout 2026 YTD 不是 prospective first-live。

## Lineage

| 項目 | 值 |
|---|---|
| Event index | `filllots_at_high.fh_pos_breakout.long`，family v1，139,515 事件／366 sessions |
| Universe | daytrade 允許、rank ≤150、09:00:30–12:30、Bid-to-Ref ∈ (−1.5%, 5%) |
| Label valid | 139,117；fill<600s 90,673（65.2%） |
| Fill branch（gate 後） | bid1_queue 51.4%、bid2_queue 41.0%、undercut_immediate 7.7%（unknown 10.6% 已排除） |
| Top-list share（gate 後） | 11.5% |
| Model era | 2025-05-13 起（60 個 event date 暖機）；development 161 日、holdout 144 日 |
| Feature manifest | `fh_pos_breakout_long_maker_defensive/1`：短線 38 根 + `in_top_list`、`defensive_weight` |
| Fill 資料 | `/media/kevin/SSD2/Data/makerFill/{d}_makerFill.parquet`，`Bid1/Bid2_FillSeconds` |
| BackTest | `src/research/backTest.py` taker-mode `side/entry_price_col/bp_col` 泛化（本次新增，預設行為不變，`tests/test_backtest_taker_side.py`） |

Artifacts 與 hash 見 `experiments/20260916_fh_pos_breakout_long_maker/artifacts/backtest_v1/run_manifest.json`。

## 成交逆選擇（全事件，未套 static gate）

樂觀 maker label = 掛 Bid1 立即成交、1.09 停利否則 Close；taker = 同列 Ask1 進場。

| year | 群 | n | maker bp | taker bp（同列） |
|---|---|---:|---:|---:|
| 2025 | 全事件 | 83,387 | −7.6 | — |
| 2025 | fill < 60s | 32,512 | −44.7 | — |
| 2025 | fill < 600s | 55,209 | −40.8 | −66.8 |
| 2025 | fill ≥ 600s | 21,249 | +77.4 | — |
| 2026 | 全事件 | 56,128 | +19.7 | −8.8 |
| 2026 | fill < 60s | 20,549 | −19.6 | — |
| 2026 | fill < 600s | 35,698 | −23.3 | −48.6 |
| 2026 | fill ≥ 600s | 12,511 | +134.9 | +102.8 |
| 2026 | fill < 600s ∧ top-list | 5,040 | +5.4 | −18.0 |
| 2025 | fill < 600s ∧ top-list | 5,382 | −33.3 | −56.4 |

解讀：breakout 事件的 Bid1 掛單只在價格回落時成交（bid1/bid2 queue 兩個分支都一樣，
2026 分別 `-19.2`／`-29.5`），真正續漲的事件從不成交。這是事件本身的性質，不是 fill
horizon 的參數問題。

## 模型層

| period | 模型 | vs 自身 target IC | vs raw bp IC (t) |
|---|---|---:|---:|
| development | timing（60d, within-stock-day z） | +0.545 | +0.134 (9.0) |
| development | selection（20d, within-day z） | +0.093 | +0.109 (7.0) |
| holdout | timing | +0.514 | +0.047 (3.9) |
| holdout | selection | +0.021 (t 1.4) | +0.017 (1.1) |

timing 對自身 target 的高 IC 是 RemainSeconds 機械梯度（legacy 已記錄），對 raw bp 沒有
可用的選股力；selection 在 holdout 歸零。

## 書層（static gate + capacity，19.3bp，日曆補零）

| variant | dev trades | dev net bp | dev Sharpe | holdout trades | holdout net bp | holdout Sharpe | holdout 正月 |
|---|---:|---:|---:|---:|---:|---:|---:|
| static_only__all | 8,255 | −96.7 | −13.3 | 5,372 | −46.8 | −4.0 | 2/8 |
| timing_only__all | 3,926 | −90.6 | −7.9 | 2,622 | −56.5 | −4.0 | 1/8 |
| selection_only__all | 5,075 | −105.2 | −9.9 | 3,481 | −48.8 | −3.7 | 2/8 |
| dual__all | 3,540 | −90.6 | −7.6 | 2,310 | −57.5 | −3.9 | 1/8 |
| static_only__top_list | 840 | −79.7 | −4.5 | 875 | +2.6 | +0.1 | 4/8 |
| **timing_only__top_list**（selected） | 344 | −44.1 | −1.3 | 347 | +8.1 | +0.3 | 3/8 |
| selection_only__top_list | 439 | −100.0 | −3.5 | 457 | +22.1 | +0.8 | 4/8 |
| dual__top_list | 265 | −64.6 | −1.6 | 305 | +9.1 | +0.3 | 3/8 |

Selected variant 的 exact BackTest：full model era `-16.0 萬`、平均部位 66 萬、年化 Sharpe
`-0.70`；holdout `+4.6 萬`、Sharpe `+0.40`、MDD `-12.0 萬`。圖：
`artifacts/backtest_v1/backtest_full.png`、`backtest_2026_holdout.png`。

### 敏感度

- Fill horizon（selected / static_only）：60s 與 300s 全部比 600s 差；剔除
  `undercut_immediate` 再差 5～20bp（`fill_horizon_sensitivity.parquet`）。
- 拿掉 `SpreadPairElapsed>0.1` 與 `OTC ∨ MD_L1Rate>0.25`（`relaxed_gate_diagnostic.parquet`）：
  all-universe dev `-69～-74`、holdout `-40～-49`；top-list holdout `-1～-12`，dev `-66～-95`。
  Gate 不是主因。
- 費用 grid（0/10/19.3/25bp）對 selected variant：0bp 時 full model era 也只有 `+3.8bp` gross，
  不改變結論。

## 限制

- 事後選出的 top-list holdout 正值切片是 outcome-informed；multiplicity 未校正；每日 2～3 筆。
- Fill 估計是 followHFTMaker 的 100ms 延遲查表慣例：`undercut_immediate` 當 0 秒是樂觀
  假設（已列敏感度）；`unknown`（價位已不在前兩檔）直接視為不成交是保守假設。
- Label 的 1.09 停利用訊號 tick 的 FutureHigh 判斷，成交前先碰到 1.09 再回落到 bid 的
  情況會被高估，實務上罕見。
- 抗跌清單只到 2026-08-10；沒有 8/11 之後的新日期。

## 下一步邊界

- Maker 進場的多方線到此為止；同一事件若還要做多，唯一合理的方向是**確認型 taker 進場**
  （事件後價格續漲 N tick 才追），因為本研究顯示「掛單成交不了」的那群才是正報酬，
  但這等於重回 taker 的 −65bp 起點加上追價成本，需要另立 objective 與新 label 才能測。
- 抗跌清單的 stock-day prior 仍成立（見 `20260916_defensive_long_universe`），較合理的用途
  是空方書的 universe／曝險調整或期貨對沖層的權重，而不是獨立多單。

## Reproduce

```bash
uv run python src/research/intradaySignal/experiments/20260916_defensive_long_universe/defensive_list.py
uv run python src/research/intradaySignal/experiments/20260916_fh_pos_breakout_long_maker/build_dataset.py --stage all
uv run python src/research/intradaySignal/experiments/20260916_fh_pos_breakout_long_maker/run_model_backtest.py --n-jobs 8
uv run --with pytest python -m pytest -q tests/test_backtest_taker_side.py
```
