"""Registered signals and the on-disk layout shared by the layers and the CLI.

Add a signal by creating ``signals/<name>.py`` with one :class:`SignalConfig`
(its four sections: envelope + condition, point_filter, day_selector, model)
and listing it in ``REGISTRY``.  Run the layers with::

    uv run python src/research/intradaySignal/pipeline/run.py all --only <name>
"""

from __future__ import annotations

from pathlib import Path

from src.research.intradaySignal.framework.registry import SignalRegistry
from src.research.intradaySignal.signals.fh_neg import FH_NEG
from src.research.intradaySignal.signals.negfill import NEGFILL


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parents[2]
ARTIFACTS_DIR = MODULE_ROOT / "artifacts"

# artifact directory per layer output
DIR_NAMES = ("envelope", "store", "params", "index", "points", "model")

REGISTRY = SignalRegistry((NEGFILL, FH_NEG))


def resolve_dirs(output_root: str | Path | None) -> dict[str, Path]:
    """Return the artifact directories, optionally redirected (tests / dry runs)."""

    root = ARTIFACTS_DIR if output_root is None else Path(output_root)
    return {name: root / name for name in DIR_NAMES}


__all__ = ["ARTIFACTS_DIR", "DIR_NAMES", "MODULE_ROOT", "PROJECT_ROOT", "REGISTRY", "resolve_dirs"]
