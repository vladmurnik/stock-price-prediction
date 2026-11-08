"""Target clipping (T9): bound the huber target, never fabricate a return.

``objective="huber"`` has no built-in response clip, unlike quantile objectives. The
plan's edge cases require the model to extrapolate neither the training label nor
the live row, so ``y_raw`` is clipped to the training slice's
``[q_lo, q_hi]`` quantile band before it is handed to the booster.

The bounds are computed on the **training slice of the fold** (no test information),
stored in ``model_meta.json``, and reused verbatim by :func:`live_clip`.

Clipping is a label-space operation only: predictions are still returns and a row
whose every feature is NaN still scores NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["PI_BAND", "clip_bounds", "clip_targets", "stored_bounds"]


#: 0.5 % / 99.5 % band: a 2-session tail is almost always a data artefact or a
#: corporate action, and huber already down-weights the rest.
PI_BAND: tuple[float, float] = (0.005, 0.995)


def clip_bounds(
    y_train: pd.Series | np.ndarray, band: tuple[float, float] = PI_BAND
) -> tuple[float, float]:
    """Quantile bounds of a labelled training series (NaN dropped, not imputed)."""
    values = np.asarray(pd.to_numeric(pd.Series(y_train), errors="coerce"), dtype="float64")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("cannot clip on an empty label slice")
    lo, hi = (float(np.quantile(values, q)) for q in band)
    return lo, hi


def clip_targets(y: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """Clip a label vector to ``[lo, hi]``; NaN rows stay NaN (never zero-filled)."""
    lo, hi = bounds
    out = np.asarray(y, dtype="float64").copy()
    finite = np.isfinite(out)
    out[finite] = np.clip(out[finite], lo, hi)
    return out


def stored_bounds(meta: dict) -> tuple[float, float]:
    """Read ``target_clip_bounds`` from ``model_meta.json`` (raises if absent)."""
    bounds = meta.get("target_clip_bounds")
    if not bounds or len(bounds) != 2:
        raise ValueError("model_meta.json has no target_clip_bounds")
    lo, hi = (float(b) for b in bounds)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        raise ValueError(f"invalid target_clip_bounds: {bounds}")
    return lo, hi
