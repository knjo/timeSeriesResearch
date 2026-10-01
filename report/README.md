# timeSeries research reports

這裡只保存有日期、有研究問題與 lineage 的結論。研究程式放在 `../test/`，框架與
protocol 放在 `../docs/`；不要把 p-value、PnL、Sharpe 或候選勝負寫回 module
README 或 skill。

## Study index

| Study | Status | Report | Reproducible code |
|---|---|---|---|
| `20260819_20260821_legacy_digest` | archived summary-only；舊 skill 與舊 test index 的完整封存 | [digest](studies/20260819_20260821_legacy_digest/REPORT.md) · [experiment table](studies/20260819_20260821_legacy_digest/EXPERIMENT_TABLE.md) | [08-19～21 study index](../test/README.md) |
| `20260825_filllots_at_low_framework` | scoped/canonical event-shaping 與 matched attribution；不是 raw-state Stage 0 gate | [report](studies/20260825_filllots_at_low_framework/REPORT.md) | [study code](../test/20260825_filllots_at_low_framework/) |
| `20260826_filllots_at_low_state_bias` | canonical raw-state long/short positive-control recovery | [report](studies/20260826_filllots_at_low_state_bias/REPORT.md) | [study code](../test/20260826_filllots_at_low_state_bias/) |
| `20260826_filllots_at_high_feature_bias` | scoped shared-pool long/short bias；dense 與 calendar complement 通過，但同股／時段 attribution 反向 | [report](studies/20260826_filllots_at_high_feature_bias/REPORT.md) | [study code](../test/20260826_filllots_at_high_feature_bias/) |
| `20260826_filllots_at_high_signal` | scoped/research-only；空方 standalone BackTest 過成本，但 model gate 無 holdout 增益且非 negFill-independent stock-day book | [report](studies/20260826_filllots_at_high_signal/REPORT.md) | [study code](../test/20260826_filllots_at_high_signal/) |
| `20260827_fh_neg_rejection_robustness` | `RESEARCH_SHADOW_ONLY`；D/B 有實質支持、G 僅統計正向且邊際；27-cell 回顧式 plateau 通過；static BackTest 已延至 2026-09-17，但 selector-only first-live calendar 不是完整 prospective validation | [report](studies/20260827_fh_neg_rejection_robustness/REPORT.md) · [static BackTest](studies/20260827_fh_neg_rejection_robustness/STATIC_BACKTEST.md) · [9/17 extension](studies/20260827_fh_neg_rejection_robustness/EXTENSION_20260917.md) · [9/16 extension](studies/20260827_fh_neg_rejection_robustness/EXTENSION_20260916.md) · [9/14 extension](studies/20260827_fh_neg_rejection_robustness/EXTENSION_20260914.md) | [study code](../test/20260827_fh_neg_rejection_robustness/) |
| `20260827_netfill_long_cover` | `RESEARCH_DIAGNOSTIC_ONLY`；六條 signed-FillLots long 在 0bp 已全負，雖有 short-tail complement shape，但固定 cover frontier 無 Sharpe/MDD joint improvement | [report](studies/20260827_netfill_long_cover/REPORT.md) | [study code](../test/20260827_netfill_long_cover/) |
| `20260916_defensive_long_universe` | scoped；抗跌清單對多方 taker pool 有 +15～22bp 的 bp 平移但 Δz≈0，絕對水位仍負；疊加 `fh_pos_breakout` 後 2026 切片首次扣費為正，樣本小、無 first-live | [report](studies/20260916_defensive_long_universe/REPORT.md) | [study code](../test/20260916_defensive_long_universe/) |
| `20260916_fh_pos_breakout_long_maker` | `RESEARCH_CLOSED_NEGATIVE`；maker Bid1 掛單在 breakout 事件有成交逆選擇（可成交子集比全事件差 40～60bp），模型 holdout IC 歸零，八個 variant 只有 top-list holdout 小樣本微正 | [report](studies/20260916_fh_pos_breakout_long_maker/REPORT.md) | [study code](../test/20260916_fh_pos_breakout_long_maker/) |
| `20260921_fh_neg_rejection_model_ablation` | `RESEARCH_DIAGNOSTIC_ONLY`；2026 static 全下優於 development-selected timing gate；L2-matched Elastic Net 與 Ridge 分數及交易近乎相同，無 holdout 改善 | [report](studies/20260921_fh_neg_rejection_model_ablation/REPORT.md) | [study code](../test/20260921_fh_neg_rejection_model_ablation/) |

## 新報告慣例

- Study id 使用 `<YYYYMMDD>_<topic>`，code 與 report 目錄同名。
- 主文件固定為 `report/studies/<study_id>/REPORT.md`。
- 報告連到 `experiments/<study_id>/` 的 manifests/artifacts；大型或可重建 artifacts 不搬進
  report tree。
- Report 必須標記 `canonical`、`scoped` 或 `archived`，並寫清楚 objective、日期窗、
  versions/fingerprints、primary estimand、uncertainty、限制與 reproduce 指令。
- 產生式 CSV/PNG 仍可寫在 `report/` 的 legacy output 目錄，但不進版控，也不能取代
  Markdown report。
