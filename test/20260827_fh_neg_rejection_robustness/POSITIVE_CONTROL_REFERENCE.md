# Positive-control reference

判定：canonical `FillLots_atLow < 0 × short` 結果可作為本研究 **G0 的歷史
pipeline-recovery component**，但不能單獨把完整 G0 判為通過，也不是
`fh_neg_rejection` 的獨立 OOS 證據。

可引用的理由：canonical 與目前 factorial run 的 label profile id/hash、eligible
pool id/version、完整 shared control 與 `shared_pool` estimand 全部一致。目前的
`run_factorial.py` 也直接沿用 canonical `run_sweep.py` 的 pool、label、beta/vol、
control evaluator，只替換 candidate registry/selector。2026-08-25 兩份既有輸出中，
11 個共同 pool/control 欄位逐值完全相等，包含 `eligible_pool_rows=2,600,076`、
raw/residual/z sums 與二次動差。

canonical recovery 本身是弱度但方向符合事前定義：2025-02-11 至 2026-08-25 共
377 sessions（376 active），short equal-day `delta_z=+0.023279`、HAC(5)
`p=.01923`、residual uplift `+5.745bp`，且 2025/2026 都為正。raw uplift 只有
`+1.246bp`，matched estimands 反號，因此它只證明研究管線能回收已知 shared-pool
distribution shift，不證明可執行 timing alpha。

主要限制是 raw files 在兩次 run 之間被刷新：2026-08-25 的共同 source paths 相同，
但 mtimes 全變，tickData size 也不同；舊 manifest 又沒有 raw content hash 或 runtime
code hash。因此 control moments 的完全一致支持語義 parity，仍不能聲稱兩次使用
byte-identical row-level snapshot。新 selector 自己的 source、reference lineage、key
uniqueness 與 no-outcome leakage 仍須由主研究另行通過，才可能完成 G0。

完整數值、artifact/code hashes 與限制在
[`positive_control_reference.json`](positive_control_reference.json)。
