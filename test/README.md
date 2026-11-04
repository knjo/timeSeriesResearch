# timeSeries study code index

`test/` 放研究腳本與 study-local artifacts；正式結論統一由
[report index](../Doc/README.md) 進入。`tests/` 是自動化 unit/integration tests，
兩者用途不同。

## Studies

2026-11-04 清理：08/19～09/21 的 17 個研究（leak_audit … fh_neg_rejection_model_ablation）程式與 artifacts 已刪除；
改寫版在 [`intradaySignal/experiments/`](../../intradaySignal/experiments/)（同名資料夾），結論仍在 [`Doc/studies/`](../Doc/studies/)。
`20260921_fh_neg_rejection_execution_gate_ablation` 的腳本 import 已刪除的 `20260827_fh_neg_rejection_robustness`／`20260826_filllots_at_high_signal`，目前無法直接重跑。
其餘研究的 `artifacts/` 也已清掉（`20261014_negfill_market_attribution` 除外，intradaySignal 實驗仍在讀）。

| Study code | 研究範圍 | Report |
|---|---|---|
| [`20261007_stop_vs_borrow/`](20261007_stop_vs_borrow/) | 固定已接受空單進場，保留 2-sigma TP，比較原 SL 與取消 SL、收盤超過 RefPrice × 1.09 時隔日開盤加 3% 出場；含 notebook 獨立比較 cell | [exit comparison](../Doc/studies/20261007_stop_vs_borrow/REPORT.md) |
| [`20260922_stop_cascade_risk/`](20260922_stop_cascade_risk/) | 風控後成交清單：盤中被嘎 k 檔之後新空單損益（window／cumulative）與既有部位 mid→Close；tickData 還原停損時間 | [stop cascade report](../Doc/studies/20260922_stop_cascade_risk/REPORT.md) |
| [`20260922_daily_regime_conditions/`](20260922_daily_regime_conditions/) | signalwithRisk 日損益 vs 六組開盤前日級條件（top-100 開低比例、隔日沖分點買進檔數／淨額、漲停率 MA、權值股比例）的 trailing-1y 與窗內 qcut10 | [daily regime report](../Doc/studies/20260922_daily_regime_conditions/REPORT.md) |
| [`20260923_beta_vol_lookback/`](20260923_beta_vol_lookback/) | preMarket beta／residual vol 改用 60／20／5 日窗口重算（沿用正式 estimator），`attach_lookback_variants` 把 `txf_beta_{N}`、`txf_residual_vol_to_1330_{N}` 接到事件；跨窗口一致性、次日樣本外避險／QLIKE、事件 z label 穩定度 | [lookback report](../Doc/studies/20260923_beta_vol_lookback/REPORT.md) |
| [`20260924_vwap_feature/`](20260924_vwap_feature/) | vwap 窗格特徵（時間／前日量時鐘）接到 negFill 點位：訓練前逐日 IC、notebook replica 模型＋16-cell 門檻＋風控＋BackTest 的四組特徵比較 | [vwap feature report](../Doc/studies/20260924_vwap_feature/REPORT.md) |

| [`20261005_dd_market_state_gate/`](20261005_dd_market_state_gate/) | 回撤能否事前判定：模型升降冪水位/速度無效；近 5 日漲停率＋開高股走勢－negFill 商品日均 合成 z<−0.7 停手（Sharpe 2.80→3.51、MDD −746→−449）；含每日 `--update` 判定腳本 | [DD gate report](../Doc/studies/20261005_dd_market_state_gate/REPORT.md) |
| [`20261102_poslimit_cap_grid/`](20261102_poslimit_cap_grid/) | negFill 單檔上限 × 每日總額網格、400 × 7,000 停手閘門調參；`supervisor_pdf/` 產主管版 PDF（台積電 0.5、蒙地卡羅壓力測試，輸入為 11/03 快照） | [主管版 PDF](../Doc/negfill_400x7000_summary.pdf) · [README](20261102_poslimit_cap_grid/README.md) |

另見 [`20260922_dynamic_exit/`](20260922_dynamic_exit/)：同一空單清單的 tick 重播，
比較停損檔數觸發縮限、30 分鐘 VWAP 動態停損停利及進場／更新線取近；
包含集合競價與非正式撮合狀態處理。[研究報告](../Doc/studies/20260922_dynamic_exit/REPORT.md)。

## 儲存契約

- 方法：`test/<study_id>/*.py`，進版控。
- 結論：`Doc/studies/<study_id>/REPORT.md`，進版控。
- Frozen point baseline：`baseline/`；不得原地覆寫。
- 可重建 artifacts/cache：`test/<study_id>/artifacts/<event>/<version>/` 或 `/tmp`，
  不進版控，也不得指向正式 `data_index.parquet` 或 `data/`。
- 正式每日點位由根目錄 `build_index.py`／`data_loader.py` 維護，不屬於 study
  artifact。
- 所有 Python 執行使用 `uv run`；新增 study 時同步更新本索引與 report index。
