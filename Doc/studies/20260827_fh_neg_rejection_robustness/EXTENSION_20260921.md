# `fh_neg_rejection` static BackTest extension through 2026-09-21

Status: `RESEARCH_SHADOW_ONLY` (static companion; not deployment evidence)  
Classification: `scoped extension`  
Direction: short  
Frozen baseline: `20250211–20260825`  
New complete sessions: `20260826–20260921` (19 sessions)  
Run contract: `fh_neg_rejection_static_backtest_extension/v1`  
Supersedes for calendar coverage: [EXTENSION_20260917.md](EXTENSION_20260917.md)

## 結論與日期界線

沿用固定的 `fh_neg_rejection` conservative D+G+B selector、short
`SHORT_TAKER_STOP108_V1` label、no-model-gate static execution/capacity，以及每筆
`19.3bp` 成本，已延伸到今天 `20260921`。9/21 的 StockTick 到 13:33、TXF tick
到 13:45，且 tickFeature、preMarket、tickBar 五種來源均存在，因此可作到收盤回測，
不是未完成盤中日。

8/25 parity 仍通過：凍結與重建各 143 筆、143 個事件鍵全部一致、成本後 PnL 都是
`-64,340.565` 元。8/26～9/17 與前一版的 17 日也逐日完全相同，最大 net PnL 差 0、
交易筆數差 0。本次寫入新的
[`static_backtest_extension_20260921_v1/`](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260921_v1/)，
沒有覆寫凍結快照或 9/17 版。

新增兩日結果：

- 9/18：111 筆、22 檔，成本後 **`-51.08 萬`**，5 筆觸發 stop108（4.50%）。
- 9/21：62 筆、9 檔，成本後 **`+0.58 萬`**，沒有事件觸發 stop108；residual net
  為 `+7.49 萬`。

今天的 raw PnL 接近打平：3189 `+5.57 萬`、3374 `+1.55 萬`，主要被 4991
`-4.23 萬`、3481 `-3.24 萬` 抵銷；8111 雖有 21 筆，合計僅 `-0.21 萬`。

## 更新後統計

| Period | Sessions | Trades | Net BP | Net PnL（萬元） | Daily Sharpe | Daily win | MDD（萬元） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 新增全段 `20260826–20260921` | 19 | 1,777 | +7.980 | +26.19 | 0.514 | 47.37% | -81.04 |
| 日曆 first-live `20260828–20260921` | 17 | 1,663 | +36.132 | +106.84 | 2.358 | 52.94% | -51.08 |
| 2026/09 `20260901–20260921` | 15 | 1,417 | +55.478 | +141.43 | 3.391 | 60.00% | -51.08 |
| 2026 YTD `20260102–20260921` | 174 | 26,347 | +29.951 | +1,109.75 | 2.368 | 49.43% | -158.29 |
| 全窗 `20250211–20260921` | 396 | 73,434 | +65.346 | +5,841.29 | 6.479 | 68.18% | -161.59 |

Net BP 是 `net_pnl / Σ(每筆 BidPrice1 × 1000) × 10000`；requested session 未出訊號
補零後算 daily Sharpe，各 period MDD 從該 period 自身起算。

First-live 累積從 9/17 版的 `+157.34 萬` 降為 `+106.84 萬`。而且仍高度集中：
9/3 `+125.40 萬`、9/17 `+66.20 萬`；排除這兩天，其餘 15 日合計 `-84.76 萬`。
因此新增資料沒有改變「短樣本、單日集中、尚未證明穩定 OOS」的判讀。

![2026 extended BackTest](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260921_v1/backtest_2026_extended.png)

## Lineage 與限制

本 extension 仍是 selector-only，未套 factorial lineage fail-closed audit（歷史上隔離約
`0.28%` model-input rows）。它是一張／event、Bid taker short、
`FutureHigh > Ref × 1.08` 時用 `Ref × 1.08 + 2 legal ticks` 回補，否則收盤 cover
的理想化 replay；沒有 borrow、impact、latency、同時曝險或 portfolio-wide capital cap。

9/18、9/21 TXF 的 `QuoteSource` 都是 `nas_txf_only_snapshot`（約 8 L1 samples/s）。
這會影響 residual/beta context 的資料同質性；raw short PnL 本身仍由股票 Bid 與
stop/close label 計算。結果不更動 parent 的 `RESEARCH_SHADOW_ONLY`。

## 重現與 artifacts

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache MPLCONFIGDIR=/tmp/matplotlib-hft \
uv run --no-sync python \
  src/research/timeSeries/test/20260827_fh_neg_rejection_robustness/extend_static_backtest.py \
  --end 20260921 \
  --output-dir src/research/timeSeries/test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260921_v1
```

- [run manifest](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260921_v1/run_manifest.json)
- [accepted event ledger](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260921_v1/accepted_signals_extension.parquet)
- [new-session daily ledger](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260921_v1/daily_extension.parquet)
- [combined daily ledger](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260921_v1/daily_combined.parquet)
- [full extended chart](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260921_v1/backtest_full_extended.png)

四個 static contract unit tests、逐筆 PnL 對帳、來源日曆、重複鍵與所有文件連結均驗證通過。
