"""C5 decision and the B1 timestamp baseline.

All statistics are fit on benign calibration drives only.
"""

from __future__ import annotations

import numpy as np


def _clean(Z: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(Z, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)


class MaxRatioDetector:
    """a_t = max_j z_tj / q_j, where q_j is the benign `q`-th percentile of feature j."""

    def __init__(self, q: float = 99.0):
        self.q = q
        self.scale = None

    def fit(self, Z_benign: np.ndarray) -> "MaxRatioDetector":
        self.scale = np.maximum(np.percentile(_clean(Z_benign), self.q, axis=0), 1e-9)
        return self

    def score(self, Z: np.ndarray) -> np.ndarray:
        return np.max(_clean(Z) / self.scale, axis=1)


def threshold_at_fpr(benign_scores: np.ndarray, fpr: float = 0.01) -> float:
    """Alarm threshold eta that gives the target false-positive rate on benign frames."""
    return float(np.quantile(benign_scores, 1.0 - fpr))


def m_of_n(above: np.ndarray, m: int = 2, n: int = 3) -> np.ndarray:
    """Alarm at t if at least m of the last n frames (including t) are above the threshold."""
    above = np.asarray(above, dtype=int)
    c = np.convolve(above, np.ones(n, dtype=int))[: len(above)]
    return c >= m


def trust(score: np.ndarray) -> np.ndarray:
    """theta_t = exp(-a_t)."""
    return np.exp(-np.maximum(np.asarray(score, dtype=float), 0.0))


def timestamp_check_score(stamps: np.ndarray, period: float = 0.1) -> np.ndarray:
    """B1: deviation of the header-time step from the nominal period; large if out of order.

    Under replay the attacker keeps the expected timestamps, so this score is the same
    as on benign data.
    """
    d = np.diff(np.asarray(stamps, dtype=float), prepend=np.nan)
    score = np.abs(d - period)
    score[d <= 0] = 1e3            # non-increasing time: strongest alarm
    score[0] = 0.0
    return score
