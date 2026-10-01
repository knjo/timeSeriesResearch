# `fh_neg_rejection` robustness study

This directory owns the study-local implementation and artifacts for
`fh_neg_rejection_robustness/fixed_factorial_v1`. The frozen research contract
is [study_contract.json](study_contract.json). No result in this study may
overwrite the existing FillLots-at-high artifacts, the formal timeSeries
pipeline, or production strategy files.

## Current stage

The retrospective eight-cell factorial, lineage audit, secondary diagnostics,
27-cell threshold-neighborhood diagnostic, and final machine decision are now
materialized. The final status is `RESEARCH_SHADOW_ONLY`; deployment and
production promotion are not authorized. Read
[decision.json](artifacts/final_decision_v1/decision.json) for the fail-closed
gate result and the paired
[REPORT.md](../../report/studies/20260827_fh_neg_rejection_robustness/REPORT.md)
for the empirical record.

The separately contracted static execution snapshot is documented in
[STATIC_BACKTEST.md](../../report/studies/20260827_fh_neg_rejection_robustness/STATIC_BACKTEST.md).
It fixes one no-model-gate variant, reconstructs the conservative anchor and
current legal-tick label, applies the inherited static execution/capacity rules,
and saves exact BackTest plots under `static_backtest_retrospective_v1/`. It is
retrospective and does not alter the parent decision.

The fixed static BackTest also has separate selector-only extensions, one
versioned directory per requested end date (`static_backtest_extension_<END>_v1/`).
The latest is through the complete session `20260917`; see
[EXTENSION_20260917.md](../../report/studies/20260827_fh_neg_rejection_robustness/EXTENSION_20260917.md)
(earlier: `20260916`, `20260914`). Rerun `extend_static_backtest.py --end <END>
--output-dir artifacts/static_backtest_extension_<END>_v1` for a newer session; it
never revises the frozen baseline or the parent shadow-only decision.

Under the frozen leave-one-gate-out rule, all three D/G/B increments are
statistically positive. Their practical interpretations differ: depth (`D`) and
bearish vote (`B`) are materially supported, while the gap-from-high gate (`G`)
has only a marginal increment and no preregistered economic-materiality pass.
The separately frozen 27-cell neighborhood passes its retrospective diagnostic
plateau gates, but it is outcome-informed and cannot select or deploy a new
parameter. The effect weakens materially across chronological blocks despite
stable prevalence, and no full factorial-lineage-audited `Date > 20260827`
prospective first-live validation is available; these are explicit reasons to
remain shadow-only.

An independent methods review after three single-session smoke runs, but before
the full-period aggregate was available, found a non-circular block-bootstrap
endpoint bias and an incomplete calendar/deployment boundary. The transparent
repair is frozen in
[study_contract_amendment_1.json](study_contract_amendment_1.json). It does not
change the selector, eight cells, objective, alpha, block lengths, or evidence
dates. It requires circular blocks, an external status calendar, evidence-class
separation, joint-valid max-stat replicates, and a later threshold-plateau
contract before deployment.

A second transparent methods completion was frozen after daily materialization
had started but before any full-period aggregate or evaluator output was
inspected. [study_contract_amendment_2.json](study_contract_amendment_2.json)
fixes the already-required HAC calculation, raw/residual BP denominators, and
the one-sided temporal-block interval. It does not change a selector,
hypothesis, gate, or threshold, and it is not represented as untouched
preregistration for already visible smoke sessions.

The study is a fixed-parameter validation, not an optimizer:

- The source anchor remains fresh event, depth ratio `<= -8`, gap from high
  `>= 4` legal ticks, and bearish vote `>= 2`.
- The primary family contains only the eight binary combinations of the
  `D`epth, `G`ap, and `B`ear gates on a common fresh negative-state base.
- The full `D+G+B` cell is the confirmatory anchor. The other seven cells are
  factorial ablations used to test parsimony and interaction.
- A simpler cell that looks better is not promoted from this run. It requires a
  new selector version, a new contract, and a new prospective clock.
- Threshold-neighborhood output, if separately requested, is diagnostic and
  cannot select a v1 parameter.
- This eight-cell stage cannot authorize deployment by itself. A separately
  frozen joint threshold plateau is required after retained gates are known.

## Evidence classes

The date boundary is part of the frozen contract:

| Dates | Evidence class | Deployment use |
|---|---|---|
| `Date <= 20260825` | retrospective diagnostic | No |
| `Date == 20260826` | artifact-OOS, non-pristine | No |
| `Date == 20260827` | freeze-day excluded | No |
| `Date > 20260827` | prospective first-live | Eligible after all gates pass |

The 20260826 session is useful as an artifact-OOS spot check because it was not
inside the earlier artifact cutoff. It is not pristine: the condition and this
validation design were already informed by prior historical research. Only a
complete session after the contract freeze can begin prospective evidence.

## Frozen primary analysis

The sample unit is one unique `(Date, QuoteCode, ChannelSeq)` key. The selector
rebuilds `FillLots_atHigh` on the complete raw stream within each
`(QuoteCode, RecordHigh)` epoch, then applies the fresh-event sampler and D/G/B
membership.

The shared-pool daily estimand is:

```text
mean(short label_z | candidate cell, day)
- mean(short label_z | complete eligible label-valid pool, same day)
```

The period estimate gives each active session equal weight. A no-event session
has support zero and undefined condition effect; it is retained on the
requested calendar but is not imputed as an effect of zero. Pooled effects,
matched attribution, raw BP, residual BP, branch results, and overlap are
secondary diagnostics.

Before interpreting this family, the identical pool, short label, beta, vol,
and control pipeline must recover the preregistered `FillLots_atLow < 0`
positive control.

## Fresh-event lineage

The frozen sampler is the union of:

- first visit to a newly changed integer B1/A1 price pair within the current
  high epoch; and
- a current formal FillLots event at least as large as
  `max(20, strictly-prior same-side big_lots)`.

Each event must be tagged `md_only`, `large_fill_only`, or
`md_and_large_fill`. Branch results diagnose heterogeneous mechanisms; they are
not a branch-selection contest.

The current preMarket producer can sanitize an invalid depth or `big_lots`
reference to `1`. The v1 runner therefore has to reconstruct or prove the
strictly-prior reference support. A sanitized `1` alone is not valid lineage.
An invalid large-fill reference fails closed for that trigger and must remain
visible in the support report.

## Required inference and decisions

The runner must preserve the complete eight-cell trial ledger and apply a
session moving-block, centered-null max-stat audit across all eight daily
effect series. The primary block length is 10 requested sessions, with 5 and 20
as fixed sensitivities. Daily HAC is secondary.

The report must independently decide:

1. lineage and positive-control recovery;
2. support;
3. prospective shared-pool bias;
4. temporal stability;
5. D/G/B parsimony;
6. selection-bias disclosure; and
7. evidence-class eligibility.

Passing the condition gates only prepares a candidate-universe handoff. It does
not mean the raw condition must beat trading cost. Any downstream rolling model
must be versioned separately and may not rescue a selector that failed these
gates.

## Materialized local outputs

All generated files remain study-local under `artifacts/<run_version>/`. The
primary navigation points are:

```text
dgb_factorial_v1/                  # frozen eight-cell retrospective run
lineage_retrospective_v1/          # selector/source/reference support audits
inference_retrospective_v1/        # max-stat, HAC inputs, temporal and LOO evidence
secondary_diagnostics_v1/          # BP, monthly and attribution diagnostics
threshold_plateau_v1/              # frozen 27-cell materialization
threshold_plateau_evaluation_v1/   # diagnostic plateau decision and disclosures
final_decision_v1/                 # fail-closed decision.json and report_data.json
static_backtest_retrospective_v1/  # fixed no-model-gate ledger, summaries and plots
static_backtest_extension_<END>_v1/ # selector-only post-freeze extensions (latest END 20260917)
```

The paired report is
[REPORT.md](../../report/studies/20260827_fh_neg_rejection_robustness/REPORT.md).
