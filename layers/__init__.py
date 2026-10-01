"""The four pipeline layers; each module owns one section of :class:`SignalConfig`.

    envelope      → artifacts/envelope, store, params/<st>_condition, index/<st>_index
    point_filter  → artifacts/points/<st>_points
    day_selector  → artifacts/params/<st>_stock_days, <st>_day_scores
    model         → artifacts/model/<st>_selected

Modules are imported explicitly (``from ..layers import model``); this package
deliberately imports nothing so that reading a config never loads sklearn or the
rolling-model engine.
"""
