# `fh_neg_rejection` static BackTest extension through 2026-09-14

Status: `RESEARCH_SHADOW_ONLY` (static companion; not deployment evidence)  
Classification: `scoped extension`  
Direction: short  
Frozen baseline: `20250211–20260825`  
New complete sessions: `20260826–20260914` (14 sessions)  
Run contract: `fh_neg_rejection_static_backtest_extension/v1`

## 結論與日期界線

已使用原本固定的 `fh_neg_rejection` conservative D+G+B selector、short
`SHORT_TAKER_STOP108_V1` label、no-model-gate static execution/capacity，以及每筆
`19.3bp` 成本，將 [原凍結快照](STATIC_BACKTEST.md)延伸到目前最後一個有五種完整來源的
收盤交易日 `20260914`。今天 `20260915` 尚無 StockTick、tickFeature、preMarket、
tickBar、TXF tick 的同日檔案，因此不能產生今天的到收盤 label；這不是把 9/15 視為
零訊號日。

8/25 的 pipeline parity：凍結與重建各 143 筆，同一批 143 個事件鍵，兩者
成本後 PnL 都是 `-64,340.565` 元（浮點誤差低於 `1e-6` 元）。原先延伸至 9/2 的
六個重疊交易日也逐日完全相同（交易筆數差 0、net PnL 差 0）。本次結果寫到新的
[versioned artifacts](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260914_v1/)，
沒有改寫凍結快照或舊 9/2 延伸檔。

新增 14 日總計 1,271 筆，成本後 `+16.53 萬`、capital-weighted net
`+7.030bp`、calendar-filled daily Sharpe `0.451`。但這個總數混合了不同證據類別：
8/26 是 non-pristine artifact-OOS，8/27 是 freeze-day excluded；兩日合計
`-80.65 萬`。只看日曆上 `Date > 20260827` 的 12 個 first-live session，則是
1,157 筆、成本後 `+97.18 萬`、net `+47.967bp`、daily Sharpe `3.142`。
這 12 日利潤主要來自 9/3 單日 `+125.40 萬`；排除 9/3，其餘 11 日合計
`-28.22 萬`。樣本短且集中，不能說已驗證穩定 OOS 或可直接部署。

| Period | Sessions | Trades | Net BP | Net PnL（萬元） | Daily Sharpe | Daily win | MDD（萬元） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 新增全段 `20260826–20260914` | 14 | 1,271 | +7.030 | +16.53 | 0.451 | 42.86% | -81.04 |
| 日曆 first-live `20260828–20260914` | 12 | 1,157 | +47.967 | +97.18 | 3.142 | 50.00% | -35.00 |
| 2026/09 `20260901–20260914` | 10 | 911 | +81.427 | +131.76 | 4.847 | 60.00% | -30.69 |
| 2026 YTD `20260102–20260914` | 169 | 25,841 | +30.455 | +1,100.09 | 2.421 | 49.11% | -158.29 |
| 全窗 `20250211–20260914` | 391 | 72,928 | +65.925 | +5,831.62 | 6.579 | 68.29% | -161.59 |

Net BP 是 `net_pnl / Σ(每筆 BidPrice1 × 1000) × 10000`；每天未出訊號的
requested session 補零後計算 daily Sharpe。各 period 的 MDD 從該 period 自身
重新起算，非全窗 drawdown 的裁切。

![2026 extended BackTest](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260914_v1/backtest_2026_extended.png)

## 新增日逐日 ledger

| Date | Evidence class | Trades | Net PnL（萬元） |
|---|---|---:|---:|
| 20260826 | artifact-OOS, non-pristine | 74 | -44.64 |
| 20260827 | freeze-day excluded | 40 | -36.01 |
| 20260828 | first-live calendar | 138 | -10.03 |
| 20260831 | first-live calendar | 108 | -24.56 |
| 20260901 | first-live calendar | 74 | -10.44 |
| 20260902 | first-live calendar | 65 | +17.45 |
| 20260903 | first-live calendar | 177 | +125.40 |
| 20260904 | first-live calendar | 148 | -30.69 |
| 20260907 | first-live calendar | 77 | +2.87 |
| 20260908 | first-live calendar | 111 | +28.30 |
| 20260909 | first-live calendar | 156 | +1.88 |
| 20260910 | first-live calendar | 32 | -1.89 |
| 20260911 | first-live calendar | 41 | +16.77 |
| 20260914 | first-live calendar | 30 | -17.87 |

9/14 最新完整日是虧損 `-17.87 萬`，不是已能用 9/15 的日內資料推估的結果。

## Lineage 與限制

本延伸重建 event index、causal event dataset、model-input features、同一個止損
short label、robustness pool 合法 tick label、固定 static gate 和同檔容量。相較
凍結 baseline，本 runner **未套用 factorial lineage fail-closed audit**，是
selector-only extension；該 audit 在整段歷史中隔離過約 `0.28%` model-input rows，
所以 post-freeze ledger 可能多保留少數原本會被 quarantine 的 event。8/25 parity
通過不等於後續每天也通過這個額外 audit。日曆 first-live 的分類也不等於完成了 parent
study 的 prospective shared-pool bias/threshold/部署 gates。

9/10、9/11、9/14 的 TXF tick 檔 `QuoteSource` 均為
`nas_txf_only_snapshot`，9/9 為 `nas_txf_only_l1`。snapshot 約 8 L1 samples/s，
因此 TXF as-of、residual label/beta context 的資料品質和較早完整 L1 日不同；raw
stock short PnL 仍用股票 Bid 與 stop/close label，但這些新日期不宜當成完全同質
的資料窗。

這仍是固定每 event 一張、Bid 進場、`FutureHigh > Ref × 1.08` 時以
`Ref × 1.08 + 2 legal ticks` 止損，否則到收盤 cover 的理想化 replay。沒有
portfolio-wide capital/concurrent exposure cap、borrow availability、market impact、
latency 或真正的 fill simulator；同檔 D-1 深度/200 萬容量只約束 accepted event。
不能因圖上累積 PnL 正值就更改 parent 的 `RESEARCH_SHADOW_ONLY`。

## 重現與 artifacts

以下是新 version 的初次 materialization 指令；同一輸出目錄已有 artifact 時 runner
會拒絕覆寫。要重跑請使用新的 study-local output version。

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache MPLCONFIGDIR=/tmp/matplotlib-hft \
uv run --no-sync python \
  src/research/intradaySignal/experiments/20260827_fh_neg_rejection_robustness/extend_static_backtest.py \
  --end 20260914 \
  --output-dir src/research/intradaySignal/experiments/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260914_v1
```

- [run manifest](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260914_v1/run_manifest.json)
- [accepted event ledger](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260914_v1/accepted_signals_extension.parquet)
- [new-session daily ledger](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260914_v1/daily_extension.parquet)
- [combined daily ledger](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260914_v1/daily_combined.parquet)
- [full extended chart](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260914_v1/backtest_full_extended.png)

期間統計沿用 [static runner 的 `_period_summary`](../../../test/20260827_fh_neg_rejection_robustness/run_static_backtest.py)：
凍結和新增 accepted ledgers 合併，依表內 requested session calendar 分別計算；
資料檢查和 4 個 static contract unit tests 均通過。
