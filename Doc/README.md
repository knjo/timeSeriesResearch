# timeSeries research reports

主管版摘要：[`negfill_400x7000_summary.pdf`](negfill_400x7000_summary.pdf)（negFill 單檔 400 萬 × 每日 7,000 萬、台積電跳空 0.5、蒙地卡羅壓力測試；產生：`../test/20261102_poslimit_cap_grid/supervisor_pdf/`，build_series → mc → make_report）。

這裡只保存有日期、有研究問題與 lineage 的結論（11/04 由 `report/` 改名為 `Doc/`）。研究程式放在 `../test/`，框架與
protocol 放在 `../docs/`；不要把 p-value、PnL、Sharpe 或候選勝負寫回 module
README 或 skill。

## Study index

| Study | Status | Report | Reproducible code |
|---|---|---|---|
| `20261014_negfill_market_attribution` | `scoped / RESEARCH_DIAGNOSTIC_ONLY`；現行 negFill notebook 2024 起重播、近期實際持倉市場因子歸因與候選池比較 | [report](studies/20261014_negfill_market_attribution/REPORT.md) | [study code](../test/20261014_negfill_market_attribution/) |
| `20261007_stop_vs_borrow` | `RESEARCH_DIAGNOSTIC_ONLY`；固定進場與 TP，配對原 SL／取消 SL 含指定券差成本，隔日價來源驗證與缺值補齊；entry-date PnL，未模擬隔夜資金 | [report](studies/20261007_stop_vs_borrow/REPORT.md) | [study code](../test/20261007_stop_vs_borrow/) |
| `20260819_20260821_legacy_digest` | archived summary-only；舊 skill 與舊 test index 的完整封存 | [digest](studies/20260819_20260821_legacy_digest/REPORT.md) · [experiment table](studies/20260819_20260821_legacy_digest/EXPERIMENT_TABLE.md) | [08-19～21 code（intradaySignal）](../../intradaySignal/experiments/) |
| `20260825_filllots_at_low_framework` | scoped/canonical event-shaping 與 matched attribution；不是 raw-state Stage 0 gate | [report](studies/20260825_filllots_at_low_framework/REPORT.md) | [study code（intradaySignal）](../../intradaySignal/experiments/20260825_filllots_at_low_framework/) |
| `20260826_filllots_at_low_state_bias` | canonical raw-state long/short positive-control recovery | [report](studies/20260826_filllots_at_low_state_bias/REPORT.md) | [study code（intradaySignal）](../../intradaySignal/experiments/20260826_filllots_at_low_state_bias/) |
| `20260826_filllots_at_high_feature_bias` | scoped shared-pool long/short bias；dense 與 calendar complement 通過，但同股／時段 attribution 反向 | [report](studies/20260826_filllots_at_high_feature_bias/REPORT.md) | [study code（intradaySignal）](../../intradaySignal/experiments/20260826_filllots_at_high_feature_bias/) |
| `20260826_filllots_at_high_signal` | scoped/research-only；空方 standalone BackTest 過成本，但 model gate 無 holdout 增益且非 negFill-independent stock-day book | [report](studies/20260826_filllots_at_high_signal/REPORT.md) | [study code（intradaySignal）](../../intradaySignal/experiments/20260826_filllots_at_high_signal/) |
| `20260827_fh_neg_rejection_robustness` | `RESEARCH_SHADOW_ONLY`；D/B 有實質支持、G 僅統計正向且邊際；27-cell 回顧式 plateau 通過；static BackTest 已延至 2026-10-13，但 selector-only first-live calendar 不是完整 prospective validation | [report](studies/20260827_fh_neg_rejection_robustness/REPORT.md) · [static BackTest](studies/20260827_fh_neg_rejection_robustness/STATIC_BACKTEST.md) · [10/13 extension](studies/20260827_fh_neg_rejection_robustness/EXTENSION_20261013.md) · [9/23 extension](studies/20260827_fh_neg_rejection_robustness/EXTENSION_20260923.md) · [9/21 extension](studies/20260827_fh_neg_rejection_robustness/EXTENSION_20260921.md) · [9/17 extension](studies/20260827_fh_neg_rejection_robustness/EXTENSION_20260917.md) | [study code（intradaySignal）](../../intradaySignal/experiments/20260827_fh_neg_rejection_robustness/) |
| `20260827_netfill_long_cover` | `RESEARCH_DIAGNOSTIC_ONLY`；六條 signed-FillLots long 在 0bp 已全負，雖有 short-tail complement shape，但固定 cover frontier 無 Sharpe/MDD joint improvement | [report](studies/20260827_netfill_long_cover/REPORT.md) | [study code（intradaySignal）](../../intradaySignal/experiments/20260827_netfill_long_cover/) |
| `20260916_defensive_long_universe` | scoped；抗跌清單對多方 taker pool 有 +15～22bp 的 bp 平移但 Δz≈0，絕對水位仍負；疊加 `fh_pos_breakout` 後 2026 切片首次扣費為正，樣本小、無 first-live | [report](studies/20260916_defensive_long_universe/REPORT.md) | [study code（intradaySignal）](../../intradaySignal/experiments/20260916_defensive_long_universe/) |
| `20260916_fh_pos_breakout_long_maker` | `RESEARCH_CLOSED_NEGATIVE`；maker Bid1 掛單在 breakout 事件有成交逆選擇（可成交子集比全事件差 40～60bp），模型 holdout IC 歸零，八個 variant 只有 top-list holdout 小樣本微正 | [report](studies/20260916_fh_pos_breakout_long_maker/REPORT.md) | [study code（intradaySignal）](../../intradaySignal/experiments/20260916_fh_pos_breakout_long_maker/) |
| `20260921_fh_neg_rejection_model_ablation` | `RESEARCH_DIAGNOSTIC_ONLY`；2026 static 全下優於 development-selected timing gate；L2-matched Elastic Net 與 Ridge 分數及交易近乎相同，無 holdout 改善 | [report](studies/20260921_fh_neg_rejection_model_ablation/REPORT.md) | [study code（intradaySignal）](../../intradaySignal/experiments/20260921_fh_neg_rejection_model_ablation/) |
| `20260921_fh_neg_rejection_execution_gate_ablation` | `RESEARCH_SHADOW_ONLY`；保留 ToRef、移除 L1 rate＋spread elapsed 的 `T1L0S0` 已凍結為擴量候選；shadow 已延至 10/13；仍為研究觀察，不因近期結果重選 | [report](studies/20260921_fh_neg_rejection_execution_gate_ablation/REPORT.md) · [10/13 extension](studies/20260921_fh_neg_rejection_execution_gate_ablation/EXTENSION_20261013.md) · [9/23 extension](studies/20260921_fh_neg_rejection_execution_gate_ablation/EXTENSION_20260923.md) | [study code](../test/20260921_fh_neg_rejection_execution_gate_ablation/) |
| `20260922_stop_cascade_risk` | `RESEARCH_DIAGNOSTIC_ONLY`；signalwithRisk 成交清單的盤中停損級聯診斷：被嘎 k 檔後的新空單與既有部位 mark-to-close | [report](studies/20260922_stop_cascade_risk/REPORT.md) | [study code](../test/20260922_stop_cascade_risk/) |
| `20260922_daily_regime_conditions` | `RESEARCH_DIAGNOSTIC_ONLY`；日損益對六組日級條件的十分位診斷 | [report](studies/20260922_daily_regime_conditions/REPORT.md) | [study code](../test/20260922_daily_regime_conditions/) |
| `20260923_beta_vol_lookback` | `RESEARCH_DIAGNOSTIC_ONLY`；beta 對窗口敏感（Spearman 60/20 0.71、60/5 0.36）、vol 較穩；次日樣本外避險與 QLIKE 四年皆 60 最好，20／5 只會變差；事件 z 排序與 60 版 ρ 0.98／0.92 | [report](studies/20260923_beta_vol_lookback/REPORT.md) | [study code](../test/20260923_beta_vol_lookback/) |
| `20260924_vwap_feature` | `RESEARCH_DIAGNOSTIC_ONLY`；vwap 窗格特徵對 `net` 有穩定大 IC 但屬 ToRef 同類結構，對 `netM` IC 很小；放進 14 特徵模型後 Sharpe 3.94→3.94～4.03、日損益差 t≤1.66，無增量，不當 gate | [report](studies/20260924_vwap_feature/REPORT.md) | [study code](../test/20260924_vwap_feature/) |
| `20261005_dd_market_state_gate` | `RESEARCH_DIAGNOSTIC_ONLY`；回撤事前判定：模型升降冪水位/速度無效；行情狀態合成 z（近5日漲停率＋開高股走勢－negFill 商品日均）<−0.7 停手，Sharpe 2.80→3.51、MDD −746→−449、前四大 DD 進入前可見；規則為樣本內選擇，未接 pipeline | [report](studies/20261005_dd_market_state_gate/REPORT.md) | [study code](../test/20261005_dd_market_state_gate/) |

- `20260922_dynamic_exit`：`RESEARCH_DIAGNOSTIC_ONLY`；停損檔數縮限與 VWAP 動態
  出場的配對 tick 重播。[report](studies/20260922_dynamic_exit/REPORT.md) ·
  [study code](../test/20260922_dynamic_exit/)。

## 新報告慣例

- Study id 使用 `<YYYYMMDD>_<topic>`，code 與 report 目錄同名。
- 主文件固定為 `Doc/studies/<study_id>/REPORT.md`。
- 報告連到 `test/<study_id>/` 的 manifests/artifacts；大型或可重建 artifacts 不搬進
  report tree。
- Report 必須標記 `canonical`、`scoped` 或 `archived`，並寫清楚 objective、日期窗、
  versions/fingerprints、primary estimand、uncertainty、限制與 reproduce 指令。
- 產生式 CSV/PNG 仍可寫在 `Doc/` 的 legacy output 目錄，但不進版控，也不能取代
  Markdown report。
