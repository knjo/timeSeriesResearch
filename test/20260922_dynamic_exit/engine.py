"""Causal event replay. Raw ticks and scheduled resets are separate event types."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def bands(mode, entry, ref, entry_sigma, sigma, center, span, previous_sl, previous_tp):
    sl0 = min(entry * (1.0 + span * entry_sigma), ref * 1.08)
    tp0 = entry * (1.0 - span * entry_sigma)
    if mode == 1:
        return sl0, tp0
    if mode == 2:
        return min(entry * (1.0 + span * sigma), ref * 1.08), entry * (1.0 - span * sigma)
    sl = min(center * (1.0 + span * sigma), ref * 1.08)
    tp = center * (1.0 - span * sigma)
    if mode == 4:
        return min(sl0, sl), tp0
    if mode == 5:
        return sl0, max(tp0, tp)
    if mode == 6:
        return min(sl0, sl), max(tp0, tp)
    if mode == 7:
        return min(previous_sl, sl0, sl), max(previous_tp, tp0, tp)
    return sl, tp


@njit(cache=True)
def replay(t, kind, fill, ask, quote_time, exec_price, exec_time, center, clock,
           entries, prices, refs, vols, full_vols, closes, close_time,
           mode, span, ratio, activation, scope):
    """Return price/time/reason and audit flags for each original entry.

    kind 0 is a real trade; 1 a minute boundary; 2 a cross-stock cascade update.
    reason 0 close, 1 SL, 2 TP. Empty ask creates an irrevocable pending cover.
    """
    n = len(entries)
    out = np.zeros((n, 11), dtype=np.float64)
    for i in range(n):
        entry_t = entries[i]
        sl = refs[i] * 1.08
        tp = -np.inf
        eligible = scope == 0 or (scope == 1 and entry_t < activation) or (scope == 2 and entry_t >= activation)
        if mode == 0:
            if activation < 0 or (entry_t >= activation and eligible):
                sl = refs[i] * ratio
        elif mode != 8:
            sl, tp = bands(1, prices[i], refs[i], vols[i], vols[i], prices[i], span, sl, tp)
        out[i, 0] = closes[i]
        out[i, 1] = close_time
        out[i, 2] = 0
        out[i, 6] = sl
        out[i, 7] = tp
        out[i, 8] = -1
        if mode == 8:
            continue
        start = np.searchsorted(t, entry_t, side="right")
        updated = False
        for j in range(start, len(t)):
            if t[j] >= close_time:
                break
            reset = False
            if kind[j] == 2 and mode == 0 and eligible and activation >= 0 and t[j] == activation:
                sl = refs[i] * ratio
                updated = True
                reset = True
            elif kind[j] == 1 and mode >= 2 and mode <= 7 and np.isfinite(center[j]):
                sl, tp = bands(mode, prices[i], refs[i], vols[i], full_vols[i] * clock[j],
                               center[j], span, sl, tp)
                reset = True
                updated = True
            if kind[j] == 0:
                observed = fill[j]
            elif reset:
                observed = ask[j]
                # At a locked upper limit no ask exists; the last traded price is
                # still observable and can arm a pending cover, never a fake fill.
                if observed <= 0:
                    observed = fill[j]
            else:
                continue
            if observed <= 0:
                continue
            reason = 0
            if observed >= sl - 1e-10:
                reason = 1
            elif observed <= tp + 1e-10:
                reason = 2
            if reason:
                out[i, 2] = reason
                out[i, 3] = int(reset)
                out[i, 4] = int(exec_time[j] > t[j] or exec_time[j] < 0)
                out[i, 5] = int(exec_time[j] < 0 or exec_time[j] >= close_time)
                out[i, 6] = sl
                out[i, 7] = tp
                out[i, 8] = t[j]
                out[i, 9] = max(0, t[j] - quote_time[j]) / 1e6
                if exec_time[j] >= 0 and exec_time[j] < close_time:
                    out[i, 0] = exec_price[j]
                    out[i, 1] = exec_time[j]
                break
        out[i, 6] = sl
        out[i, 7] = tp
        out[i, 10] = int(updated)
    return out
