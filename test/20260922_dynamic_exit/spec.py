"""Frozen candidate family for paired short-exit research; no entry changes."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    mode: int = 0
    span: float = 0.0
    ratio: float = 1.08
    k: int = 0
    scope: int = 0  # 0 all, 1 positions entered before activation, 2 new entries
    average: str = "vwap"
    baseline: str = "ref08"


def candidates() -> list[Candidate]:
    out = [Candidate("ref08", "reference"), Candidate("ref05", "reference", ratio=1.05),
           Candidate("ref06", "reference", ratio=1.06),
           Candidate("hold_close", "reference", mode=8)]
    for k in range(1, 7):
        for pct in (5, 6):
            out.append(Candidate(f"cascade_k{k}_to{pct:02d}", "cascade", ratio=1+pct/100, k=k))
    for k in (3, 5):
        for pct in (5, 6):
            for scope, label in ((1, "existing"), (2, "new")):
                out.append(Candidate(f"cascade_k{k}_to{pct:02d}_{label}", "cascade",
                                     ratio=1+pct/100, k=k, scope=scope))
    modes = {"entry": 1, "clock": 2, "rolling": 3, "near_sl": 4, "near_tp": 5, "near_both": 6}
    for span in (2.0, 2.5, 3.0):
        tag = str(span).replace(".", "p")
        baseline = f"vol_entry_{tag}"
        for label, mode in modes.items():
            out.append(Candidate(f"vol_{label}_{tag}", "volatility", mode=mode,
                                 span=span, baseline="ref08" if mode == 1 else baseline))
    out.append(Candidate("vol_ratchet_2p5", "volatility", mode=7, span=2.5, baseline="vol_entry_2p5"))
    for label, mode in (("rolling", 3), ("near_both", 6)):
        out.append(Candidate(f"vol_{label}_2p5_sma", "sensitivity", mode=mode, span=2.5,
                             average="minute_last_sma", baseline="vol_entry_2p5"))
    return out


def contract() -> dict:
    return {
        "version": "v4", "direction": "short", "entry": "unchanged signal ledger, BidPrice1, 1000 shares per row",
        "fees_bp": 19.3, "primary_execution": "observed AskPrice1 with positive AskLots1; wait for next ask if empty",
        "nonformal_phase": "retain timestamps of TrialMatch != 0 as non-executable book state; never carry a prior formal ask through a trial/auction phase",
        "slippage_sensitivity": "add 2 entry TickSize to every intraday exit; identical exit times",
        "touch": "valid trade FillPrice >= SL or <= TP, strictly after entry; SL priority on overlap",
        "reset_touch": "at scheduled update, evaluate current observable ask; fill at that ask, never at reset barrier",
        "close": "13:30; Close marks unresolved no-ask orders, explicitly flagged, not assumed executable",
        "hard_sl": "RefPrice * 1.08 for every volatility candidate",
        "hard_tp": "no additional absolute TP in supplied formula; near_tp/near_both retain original entry TP as floor",
        "minutes": "09:30 through 13:24, trailing [t-30min,t) formal trades; skip empty windows; no immediate fills during 13:25-13:30 closing auction",
        "vol": "prior full-session residual volatility multiplied by canonical remaining-to-13:30 clock",
        "cascade": "single permanent switch at K distinct held symbols actually stopped under ref08; no further stages",
        "cascade_identity": "all cascade policies equal ref08 before switch, so baseline first-stop times identify their causal switch",
        "reentry": "keep every supplied entry; already beyond new barrier exits on first later eligible event; no cap recycling",
        "near": "SL=min(entry SL, moving SL); TP=max(entry TP, moving TP); not nearest absolute distance",
        "ratchet": "add previous SL/TP to min/max; separate diagnostic from entry-bound comparison",
        "primary_estimand": "paired same-day net TWD difference on identical entries, with nominal-weighted and equal-day BP",
        "uncertainty": "5-session circular moving-block bootstrap, 2000 draws, paired mean intervals and family max-t adjusted p",
        "stability": "monthly, 2025/2026, pre/post 20260701; retrospective slices, not untouched OOS",
        "selection": "fixed family; no promoted winner, inspect neighboring k and spans; research diagnostic only",
        "candidates": [asdict(c) for c in candidates()],
    }
