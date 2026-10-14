"""Replay injection.

Inside a window [t_s, t_e], slot t delivers frame L_{t-R}. The header timestamp and the
IMU/GPS reference stay those of slot t. Outside windows, sigma(t) = t.

Labels
------
label = 1 for every slot inside a window and for the first fresh slot after it
(the exit seam), because that slot's registration still spans the last replayed frame.
seam  = 1 for the entry slot t_s and the exit slot t_e + 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayConfig:
    replay_s: float = 10.0     # replay age R [s]
    win_min_s: float = 2.0     # shortest window [s]
    win_max_s: float = 10.0    # longest window [s]
    gap_s: float = 10.0        # benign gap between windows [s]
    period_s: float = 0.1      # LiDAR period Delta [s]
    seed: int = 0


def windows(n: int, cfg: ReplayConfig) -> list:
    """Random, non-overlapping windows (start, end) inclusive, with start >= R."""
    rng = np.random.default_rng(cfg.seed)
    R = int(round(cfg.replay_s / cfg.period_s))
    wmin = int(round(cfg.win_min_s / cfg.period_s))
    wmax = int(round(cfg.win_max_s / cfg.period_s))
    gap = int(round(cfg.gap_s / cfg.period_s))
    out, t = [], max(R, gap)
    while True:
        start = t + int(rng.integers(0, gap + 1))
        length = int(rng.integers(wmin, wmax + 1))
        end = start + length - 1
        if end + 2 >= n:          # keep room for the exit seam
            break
        out.append((start, end))
        t = end + 1 + gap
    return out


def inject(n: int, cfg: ReplayConfig):
    """Return sigma, label, seam, R (in frames), and the list of windows."""
    R = int(round(cfg.replay_s / cfg.period_s))
    sigma = np.arange(n)
    label = np.zeros(n, dtype=int)
    seam = np.zeros(n, dtype=int)
    wins = windows(n, cfg)
    for s, e in wins:
        t = np.arange(s, e + 1)
        sigma[t] = t - R
        label[s:e + 2] = 1          # window + exit seam
        seam[s] = 1
        seam[e + 1] = 1
    return sigma, label, seam, R, wins
