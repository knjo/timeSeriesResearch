# `fh_neg_rejection` static BackTest extension through 2026-09-16

Status: `RESEARCH_SHADOW_ONLY` (static companion; not deployment evidence)  
Classification: `scoped extension`  
Direction: short  
Frozen baseline: `20250211–20260825`  
New complete sessions: `20260826–20260916` (16 sessions)  
Run contract: `fh_neg_rejection_static_backtest_extension/v1`  
Supersedes for calendar coverage: [EXTENSION_20260914.md](EXTENSION_20260914.md)（其數字與判讀不改寫）

## 結論與日期界線

沿用固定的 `fh_neg_rejection` conservative D+G+B selector、short `SHORT_TAKER_STOP108_V1`
label、no-model-gate static execution/capacity、每筆 `19.3bp`，把凍結快照延伸到
`20260916`。9/15、9/16 的 StockTick（至 13:33）、tickFeature、preMarket、tickBar、TXF tick
（至 13:45）五種來源都已完整，因此兩日都是到收盤的 first-live session。

Parity：8/25 凍結與重建各 143 筆、同一批事件鍵、成本後 PnL 都是 `-64,340.565` 元。
8/26～9/14 的 14 個交易日逐日與 9/14 版延伸完全相同（筆數差 0、net PnL 差 0）。
本次寫到新的 versioned 目錄
[`static_backtest_extension_20260916_v1/`](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260916_v1/)，
沒有改寫凍結快照或舊延伸檔。

新增兩日：**9/15 `+36.40 萬`（122 筆）、9/16 `-42.43 萬`（50 筆）**，兩日合計 `-6.03 萬`。
把它們接在 9/14 版之後，各 period 變成：

| Period | Sessions | Trades | Net BP | Net PnL（萬元） | Daily Sharpe | Daily win | MDD（萬元） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 新增全段 `20260826–20260916` | 16 | 1,443 | +3.956 | +10.49 | 0.252 | 43.75% | -81.04 |
| 日曆 first-live `20260828–20260916` | 14 | 1,329 | +39.158 | +91.14 | 2.528 | 50.00% | -42.43 |
| 2026/09 `20260901–20260916` | 12 | 1,083 | +65.490 | +125.73 | 3.871 | 58.33% | -42.43 |
| 2026 YTD `20260102–20260916` | 171 | 26,013 | +30.037 | +1,094.06 | 2.381 | 49.12% | -158.29 |
| 全窗 `20250211–20260916` | 393 | 73,100 | +65.633 | +5,825.59 | 6.531 | 68.19% | -161.59 |

Net BP 是 `net_pnl / Σ(每筆 BidPrice1 × 1000) × 10000`；requested session 未出訊號補零
後算 daily Sharpe；各 period 的 MDD 從該 period 自身起算。first-live 14 日的利潤仍然
主要來自 9/3 的 `+125.40 萬`；排除 9/3，其餘 13 日合計 `-34.26 萬`。9/16 是新增段的
最大單日虧損，也把 first-live 段的 MDD 從 `-35.00` 拉到 `-42.43 萬`。樣本短且集中，
判讀與 9/14 版相同：不能說已驗證穩定 OOS 或可直接部署。

![2026 extended BackTest](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260916_v1/backtest_2026_extended.png)

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
| **20260915** | first-live calendar | 122 | **+36.40** |
| **20260916** | first-live calendar | 50 | **-42.43** |

## Lineage 與限制

與 9/14 版相同：selector-only extension，未套 factorial lineage fail-closed audit
（歷史上隔離約 `0.28%` model-input rows）；一張／event、Bid 進場、`FutureHigh > Ref × 1.08`
時 `Ref × 1.08 + 2 legal ticks` 止損否則收盤 cover 的理想化 replay；沒有 portfolio cap、
borrow、impact、latency 或 fill simulator。9/15、9/16 的 TXF tick `QuoteSource` 都是
`nas_txf_only_snapshot`（約 8 L1 samples/s），與 9/10～9/14 相同，residual/beta context 的
資料品質與早期完整 L1 日不同；raw short PnL 只用股票 Bid 與 stop/close label。

## 重現與 artifacts

```bash
MPLCONFIGDIR=/tmp/matplotlib-hft \
uv run python \
  src/research/intradaySignal/experiments/20260827_fh_neg_rejection_robustness/extend_static_backtest.py \
  --end 20260916 \
  --output-dir src/research/intradaySignal/experiments/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260916_v1
```

- [run manifest](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260916_v1/run_manifest.json)
- [accepted event ledger](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260916_v1/accepted_signals_extension.parquet)
- [new-session daily ledger](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260916_v1/daily_extension.parquet)
- [combined daily ledger](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260916_v1/daily_combined.parquet)
- [full extended chart](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260916_v1/backtest_full_extended.png)
