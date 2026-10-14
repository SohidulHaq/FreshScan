"""C3 consistency checks.

Feature vector z_t = (r_v, r_w, 1 - f, e):
    r_v  relative speed residual  |v_hat - v_bar| / (|v_bar| + v0)
    r_w  yaw-rate residual        |w_hat - w_bar|
    1-f  ICP misfit (seam)        1 - fitness
    e    ICP inlier RMSE (seam)

motion_features() is the version used for evaluation: LiDAR rates use the sensor's real
scan period (a constant measured on benign calibration drives, never per-frame timestamps),
both motion sources are averaged over the last k slots (causal), and the motion checks
abstain when the reference speed is below v_min (principle P3: too little motion to judge).
"""

from __future__ import annotations

import numpy as np

FEATURES = ("r_v", "r_w", "one_minus_fit", "rmse")
SEAM_FEATURES = ("one_minus_fit", "rmse")          # baseline B2
RESIDUAL_FEATURES = ("r_v", "r_w")                 # baseline B3


def residuals(v_hat, w_hat, v_bar, w_bar, v0: float = 1.0):
    v_hat, w_hat, v_bar, w_bar = map(np.asarray, (v_hat, w_hat, v_bar, w_bar))
    r_v = np.abs(v_hat - v_bar) / (np.abs(v_bar) + v0)
    r_w = np.abs(w_hat - w_bar)
    return r_v, r_w


def feature_table(v_hat, w_hat, v_bar, w_bar, fit, rmse, v0: float = 1.0) -> dict:
    r_v, r_w = residuals(v_hat, w_hat, v_bar, w_bar, v0)
    return {"r_v": r_v, "r_w": r_w,
            "one_minus_fit": 1.0 - np.asarray(fit, dtype=float),
            "rmse": np.asarray(rmse, dtype=float)}


def causal_mean(x, k: int) -> np.ndarray:
    """Mean of the last k values (fewer at the start of a stream)."""
    x = np.asarray(x, dtype=float)
    if k <= 1:
        return x
    c = np.cumsum(np.r_[0.0, x])
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - k, 0)
    return (c[idx] - c[lo]) / (idx - lo)


def motion_features(v_hat, w_hat, v_bar, w_bar, period_ratio: float = 1.0, k: int = 1,
                    v0: float = 1.0, v_min: float = 0.0, w0: float | None = None):
    """Speed and yaw-rate residuals with period correction, k-slot averaging and low-speed abstain.

    period_ratio = nominal period used by step 1 / real scan period.
    w0: if given, the yaw residual is relative, |w_hat - w_bar| / (|w_bar| + w0), like the speed residual.
    Returns r_v, r_w and a boolean array `checked` (False where the check abstains).
    """
    vh = causal_mean(np.asarray(v_hat, dtype=float) * period_ratio, k)
    wh = causal_mean(np.asarray(w_hat, dtype=float) * period_ratio, k)
    vb, wb = causal_mean(v_bar, k), causal_mean(w_bar, k)
    r_v = np.abs(vh - vb) / (np.abs(vb) + v0)
    r_w = np.abs(wh - wb)
    if w0 is not None:
        r_w = r_w / (np.abs(wb) + w0)
    checked = np.abs(vb) >= v_min
    r_v = np.where(checked, r_v, 0.0)
    r_w = np.where(checked, r_w, 0.0)
    return r_v, r_w, checked
