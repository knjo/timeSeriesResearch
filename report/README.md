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

## 新報告慣例

- Study id 使用 `<YYYYMMDD>_<topic>`，code 與 report 目錄同名。
- 主文件固定為 `report/studies/<study_id>/REPORT.md`。
- 報告連到 `test/<study_id>/` 的 manifests/artifacts；大型或可重建 artifacts 不搬進
  report tree。
- Report 必須標記 `canonical`、`scoped` 或 `archived`，並寫清楚 objective、日期窗、
  versions/fingerprints、primary estimand、uncertainty、限制與 reproduce 指令。
- 產生式 CSV/PNG 仍可寫在 `report/` 的 legacy output 目錄，但不進版控，也不能取代
  Markdown report。
