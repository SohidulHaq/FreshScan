"""Step 1: run FreshScan components C1-C3 on benign and replayed streams and save per-frame CSVs.

For every calibration drive: the benign stream.
For every test drive: the benign stream and one replayed stream per (replay age R, seed).

Output: results/features/<drive>__<setting>.csv with one row per slot:
    slot, sigma, label, seam, tau, v_bar, w_bar, v_bar_src, w_bar_src,
    v_hat, w_hat, fit, rmse, r_v, r_w, one_minus_fit, prep_ms, icp_ms, check_ms

Usage (from the repository root):
    python scripts/01_compute_features.py                     # all drives in the config
    python scripts/01_compute_features.py --drives 2011_09_30_drive_0018
"""

from __future__ import annotations

import argparse
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

from common import load_config, out_dir  # noqa: E402  (adds the repository root to sys.path)
from freshscan.features import feature_table
from freshscan.kitti_loader import load_drive
from freshscan.registration import icp, prepare, select_backend
from freshscan.runinfo import write_run_info
from freshscan.replay import ReplayConfig, inject


class FrameCache:
    """Keeps the most recently prepared frames (C1) in memory."""

    def __init__(self, drive, fs: dict, size: int = 512):
        self.drive, self.fs, self.size = drive, fs, size
        self.data: OrderedDict = OrderedDict()
        self.prep_ms: dict = {}

    def get(self, i: int) -> np.ndarray:
        if i in self.data:
            self.data.move_to_end(i)
            return self.data[i]
        t0 = time.perf_counter()
        pts = prepare(self.drive.frame(i), self.fs["r_min"], self.fs["r_max"], self.fs["voxel"])
        self.prep_ms[i] = 1e3 * (time.perf_counter() - t0)
        self.data[i] = pts
        if len(self.data) > self.size:
            self.data.popitem(last=False)
        return pts


def run_stream(drive, sigma, fs, frames: FrameCache, pair_cache: dict):
    """C2 on consecutive delivered frames. Returns per-slot LiDAR motion, fit and timings."""
    n, period = len(sigma), fs["period_s"]
    v_hat, w_hat = np.zeros(n), np.zeros(n)
    fit, rmse = np.ones(n), np.zeros(n)
    prep_ms, icp_ms = np.zeros(n), np.zeros(n)
    T_prev, f_prev, prev_key = np.eye(4), 1.0, None
    for t in range(n):
        frames.get(int(sigma[t]))
        prep_ms[t] = frames.prep_ms.get(int(sigma[t]), 0.0)
        if t == 0:
            continue
        a, b = int(sigma[t - 1]), int(sigma[t])
        key = (prev_key, a, b)
        if key not in pair_cache:
            init = T_prev if f_prev >= fs["fit_reset"] else np.eye(4)   # P2: LiDAR-only init
            t0 = time.perf_counter()
            res = icp(frames.get(b), frames.get(a), init=init, d_max=fs["d_max"], iters=fs["icp_iters"])
            pair_cache[key] = (res.forward, res.yaw, res.fitness, res.rmse,
                               1e3 * (time.perf_counter() - t0), res.T)
        fwd, yaw, f, e, ms, T = pair_cache[key]
        v_hat[t], w_hat[t] = fwd / period, yaw / period
        fit[t], rmse[t], icp_ms[t] = f, e, ms
        T_prev, f_prev, prev_key = T, f, (a, b)
    return v_hat, w_hat, fit, rmse, prep_ms, icp_ms


def slot_mean(x):
    """Average of the samples at frames t-1 and t (frame 0 keeps its own sample)."""
    x = np.asarray(x, dtype=float)
    return np.r_[x[:1], 0.5 * (x[1:] + x[:-1])]


def save_stream(path, drive, sigma, label, seam, out, fs):
    v_hat, w_hat, fit, rmse, prep_ms, icp_ms = out
    n = len(sigma)
    # Reference motion for slot t = mean of the IMU/GPS samples at both ends of the interval
    # [t-1, t] that ICP measures.
    v_bar, w_bar = slot_mean(drive.speed[:n]), slot_mean(drive.yaw_rate[:n])
    t0 = time.perf_counter()
    feats = feature_table(v_hat, w_hat, v_bar, w_bar, fit, rmse, fs["v0"])
    check_ms = 1e3 * (time.perf_counter() - t0) / max(n, 1)
    df = pd.DataFrame({
        "slot": np.arange(n), "sigma": sigma, "label": label, "seam": seam,
        "tau": drive.stamps[:n],                       # header time of slot t (kept by the attacker)
        "v_bar": v_bar, "w_bar": w_bar,
        "v_bar_src": slot_mean(drive.speed)[sigma], "w_bar_src": slot_mean(drive.yaw_rate)[sigma],  # motion when the frame was captured
        "v_hat": v_hat, "w_hat": w_hat, "fit": fit, "rmse": rmse,
        "r_v": feats["r_v"], "r_w": feats["r_w"], "one_minus_fit": feats["one_minus_fit"],
        "prep_ms": prep_ms, "icp_ms": icp_ms, "check_ms": check_ms,
    })
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--drives", nargs="*", default=None, help="subset of drives to process")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fs, rp = cfg["freshscan"], cfg["replay"]
    feat_dir = out_dir(cfg, "features")
    backend = select_backend(fs.get("icp_backend", "auto"))
    print(f"ICP backend: {backend}")

    calib, test = cfg["calibration_drives"], cfg["test_drives"]
    todo = [d for d in calib + test if args.drives is None or d in args.drives]
    for name in todo:
        drive = load_drive(cfg["kitti_root"], name, fs["period_s"])
        n = len(drive)
        frames, pair_cache = FrameCache(drive, fs), {}
        streams = [("benign", np.arange(n), np.zeros(n, int), np.zeros(n, int))]
        if name in test:
            for R_s in rp["ages_s"]:
                for seed in rp["seeds"]:
                    rc = ReplayConfig(replay_s=R_s, win_min_s=rp["win_min_s"], win_max_s=rp["win_max_s"],
                                      gap_s=rp["gap_s"], period_s=fs["period_s"], seed=seed)
                    sigma, label, seam, R, wins = inject(n, rc)
                    if not wins:
                        print(f"[{name}] drive too short for R={R_s}s, skipped")
                        continue
                    streams.append((f"replay_R{R_s}_s{seed}", sigma, label, seam))
        for setting, sigma, label, seam in streams:
            path = os.path.join(feat_dir, f"{name}__{setting}.csv")
            if os.path.exists(path):
                print(f"[{name}] {setting:16s} exists, skipped"); continue
            t0 = time.time()
            out = run_stream(drive, sigma, fs, frames, pair_cache)
            save_stream(path, drive, sigma, label, seam, out, fs)
            write_run_info(cfg, name, backend)
            print(f"[{name}] {setting:16s} frames={n} attacked={int(label.sum())} "
                  f"({time.time() - t0:.0f}s) -> {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
