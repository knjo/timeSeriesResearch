# Historical trial-exposure inventory

`historical_trial_exposure.json` is a reproducible lower-bound inventory of
known outcome-inspected FillHigh/FH15 work from 20260821 and the
FillLots-at-high feature-bias/signal studies from 20260826.

It is not an estimate of every experiment that may have been tried. The
generator reads only checked-in code and JSON, verifies literal candidate or
grid definitions, and fingerprints each evidence file. Informal chat,
notebooks, removed code, and unrecorded experiments remain unknown; the
artifact never fills that gap with a guessed count.

## Rebuild

From the project root:

```bash
uv run python \
  src/research/timeSeries/test/20260827_fh_neg_rejection_robustness/\
build_historical_trial_exposure.py
```

The script is fail-closed when an expected source is missing, an AST literal can
no longer be reconstructed, or the 20260826 profile and study registries drift.
It also refuses to write outside its own study directory.

## Counting interpretation

The JSON keeps four kinds of exposure separate:

- condition selector configurations and their direction/label outcome tests;
- rolling model target/lookback components;
- threshold or model-gate variants selected by PnL/Sharpe; and
- execution, fee, portfolio, overlap, or post-hoc partition diagnostics.

Only the first kind is condition-level `delta_z` exposure. Even those historical
families are not automatically one max-stat family: the 20260821 and 20260826
studies used different selector grammars, event shaping, pool/report contracts,
and labels. The arithmetic selector/direction totals in the artifact are
disclosure context, not an exchangeable null distribution.

The current robustness study should therefore:

1. apply its blocked max-stat correction to its own frozen eight-cell D/G/B
   family;
2. disclose the historical families as material prior selection exposure;
3. avoid treating `fh_neg_rejection` as an ex-ante single historical test; and
4. rely on sessions after 20260827, not retrospective adjustment, for pristine
   prospective evidence.

Model, threshold, Sharpe, fee, and execution trials are intentionally excluded
from the current condition `delta_z` max-stat family. If a downstream strategy
family is later selected by net daily Sharpe, it needs its own DSR-like or
blocked family audit on a common calendar.

## Known reconstruction caveat

The `stage_fh15.py` docstring describes five models, while its executable
`runs` assignment contains two. The generator reports two executable model
configurations and records the discrepancy. It does not guess whether three
removed configurations were run elsewhere.
