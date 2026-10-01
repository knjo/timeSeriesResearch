# intradaySignal experiments index

`experiments/<YYYYMMDD_topic>/` 放研究腳本與 study-local artifacts；正式結論統一由
[report index](../report/README.md) 進入（`report/studies/<同 id>/REPORT.md`）。`tests/` 是自動化
unit/integration tests，兩者用途不同。實驗不得寫入 `artifacts/`。

## Studies

| Study code | 研究範圍 | Report |
|---|---|---|
| [`20260819_leak_audit/`](20260819_leak_audit/) | notebook 復刻、future leakage 拔除、feature IC 與殘差漂移 | [legacy digest](../report/studies/20260819_20260821_legacy_digest/REPORT.md) |
| [`20260819_threshold_opt/`](20260819_threshold_opt/) | AB 分支、q sweep、三代 threshold 與 drawdown 比較 | [legacy experiment table](../report/studies/20260819_20260821_legacy_digest/EXPERIMENT_TABLE.md) |
| [`20260820_bucket_research/`](20260820_bucket_research/) | conditional baseline、tree/grid buckets、16-cell 與 q sweep | [legacy digest](../report/studies/20260819_20260821_legacy_digest/REPORT.md) |
| [`20260821_condition_explore/`](20260821_condition_explore/) | Overshoot、FillLots-at-low recovery、FillLots-at-high 與 depth condition 探索 | [legacy experiment table](../report/studies/20260819_20260821_legacy_digest/EXPERIMENT_TABLE.md) |
| [`20260821_depth_variants/`](20260821_depth_variants/) | depth pipeline、two-tier sizing 與 dual-book 比較 | [legacy experiment table](../report/studies/20260819_20260821_legacy_digest/EXPERIMENT_TABLE.md) |
| [`20260821_fh15_line/`](20260821_fh15_line/) | FH15 獨立 pipeline、overlap/twin split 與 FH2 redesign | [legacy digest](../report/studies/20260819_20260821_legacy_digest/REPORT.md) |
| [`20260821_long_line/`](20260821_long_line/) | FHL long pipeline、portfolio 與 refit | [legacy digest](../report/studies/20260819_20260821_legacy_digest/REPORT.md) |
| [`20260825_filllots_at_low_framework/`](20260825_filllots_at_low_framework/) | 26-candidate event shaping、shared/matched attribution 與 overlap audit | [framework validation](../report/studies/20260825_filllots_at_low_framework/REPORT.md) |
| [`20260826_filllots_at_low_state_bias/`](20260826_filllots_at_low_state_bias/) | 原始 sign state 的 long/short shared bias、HAC 與 distribution report | [state-bias report](../report/studies/20260826_filllots_at_low_state_bias/REPORT.md) |
| [`20260826_filllots_at_high_feature_bias/`](20260826_filllots_at_high_feature_bias/) | MD first-visit／large-fill 去重、tickFeature interactions、sparse promotion 與 dense attribution | [feature-bias report](../report/studies/20260826_filllots_at_high_feature_bias/REPORT.md) |
| [`20260826_filllots_at_high_signal/`](20260826_filllots_at_high_signal/) | `fh_neg_rejection` 空方的 causal rolling model、prior threshold、容量、成本、BackTest 與 negFill overlap | [tradingized signal report](../report/studies/20260826_filllots_at_high_signal/REPORT.md) |
| [`20260827_fh_neg_rejection_robustness/`](20260827_fh_neg_rejection_robustness/) | `fh_neg_rejection` D/G/B factorial 必要性、lineage、selection-bias 與 27-cell fixed-threshold plateau 診斷 | [robustness report](../report/studies/20260827_fh_neg_rejection_robustness/REPORT.md) |
| [`20260827_netfill_long_cover/`](20260827_netfill_long_cover/) | 六條 signed-FillLots taker-long standalone 與對 `fh_neg_rejection` 的固定 cover frontier | [long-cover report](../report/studies/20260827_netfill_long_cover/REPORT.md) |
| [`20260916_defensive_long_universe/`](20260916_defensive_long_universe/) | 抗跌族群清單當多方 stock-day universe：pool 層 shared bias、middle 對照、六條 long ledger 切分 | [defensive universe report](../report/studies/20260916_defensive_long_universe/REPORT.md) |
| [`20260916_fh_pos_breakout_long_maker/`](20260916_fh_pos_breakout_long_maker/) | `fh_pos_breakout` maker 多單：makerFill 100ms fill gate、causal 模型、16-cell 門檻、8 個 universe×gate variant、純多 BackTest | [maker-long report](../report/studies/20260916_fh_pos_breakout_long_maker/REPORT.md) |
| [`20260921_fh_neg_rejection_model_ablation/`](20260921_fh_neg_rejection_model_ablation/) | `fh_neg_rejection` static 全下、原 Ridge gate 與 L2-matched Elastic Net 的 frozen 2026 evaluation 對照 | [model ablation report](../report/studies/20260921_fh_neg_rejection_model_ablation/REPORT.md) |

## 儲存契約

- 方法：`experiments/<study_id>/*.py`，進版控。
- 結論：`report/studies/<study_id>/REPORT.md`，進版控。
- Frozen point baseline：`baseline/`；不得原地覆寫。
- 可重建 artifacts/cache：`experiments/<study_id>/artifacts/<event>/<version>/` 或 `/tmp`，
  不進版控，也不得指向正式 `data_index.parquet` 或 `data/`。
- 正式每日點位由根目錄 `build_index.py`／`data_loader.py` 維護，不屬於 study
  artifact。
- 所有 Python 執行使用 `uv run`；新增 study 時同步更新本索引與 report index。
