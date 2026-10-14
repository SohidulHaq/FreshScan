"""Calibration fit ("checkpoint") and the scoring rule of FreshScan, shared by steps 2 and 5.

FreshScan has no neural network and no training. Everything it learns comes from benign
(normal) driving, in one calibration *fold*:
    * the real LiDAR scan period (median header step; one constant, never per frame, P1),
    * the benign 99th percentile of every check (detector scales, C5),
    * the alarm threshold eta (target false-positive rate on the fold's benign frames),
    * the seam-alarm threshold (much stricter, default 0.1%).

Evaluation protocol (config key `protocol`):
    leave_one_drive_out  each test drive is scored with a fold calibrated on the benign streams
                         of all *other* drives (no attack data is ever used);
    fixed                one fold calibrated on `calibration_drives` only.
An extra fold "all" is calibrated on every benign drive; use it for new data.

Alarm rule (C5): alarm at slot t if
    (a) the FreshScan score is above eta in at least m of the last n slots, or
    (b) the seam score (ICP fit features only) is above the seam threshold in slot t.
Rule (b) lets one clearly broken registration, which marks the start or end of a replay,
raise an alarm at once instead of being filtered out by the persistence rule.

Files (results/checkpoint/):
    freshscan_checkpoint.json   settings, protocol, and per fold: drives, scan period, scales, thresholds
    calibration_scores.npz      per fold: benign calibration scores of every method (to set new thresholds)
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from .detector import MaxRatioDetector, m_of_n, threshold_at_fpr, timestamp_check_score, trust
from .features import FEATURES, RESIDUAL_FEATURES, SEAM_FEATURES, motion_features

VERSION = 2
METHODS = {"B2: Seam only": list(SEAM_FEATURES), "B3: Residuals only": list(RESIDUAL_FEATURES),
           "FreshScan": list(FEATURES)}
B1 = "B1: Timestamp check"


def _key(m: str) -> str:
    return m.split(":")[0].replace(" ", "_")


class Fold:
    """One calibration fit and the scoring rule that uses it."""

    def __init__(self, name, fs, cal_drives, period, scales, thr, cal_scores, percentile=99.0):
        self.name, self.fs, self.cal_drives = name, fs, list(cal_drives)
        self.period = float(period)
        self.dets = {}
        for m, sc in scales.items():
            d = MaxRatioDetector(percentile); d.scale = np.asarray(sc, dtype=float); self.dets[m] = d
        self.thr = {k: float(v) for k, v in thr.items()}
        self.cal_scores = cal_scores          # method -> benign calibration scores

    # --- fitting ---
    @classmethod
    def fit(cls, name, fs, cal_streams: dict) -> "Fold":
        """cal_streams: {drive: raw benign feature DataFrame (slot 0 already removed)}."""
        lp = fs.get("lidar_period_s", "auto")
        if lp == "auto":
            steps = np.concatenate([np.diff(df.tau.values) for df in cal_streams.values()])
            lp = float(np.median(steps[steps > 0]))
        fold = cls(name, fs, list(cal_streams), lp, {}, {}, {}, fs["percentile"])
        feats = [fold.features(df) for df in cal_streams.values()]
        for m, cols in METHODS.items():
            Z = np.vstack([f[cols].values for f in feats])
            fold.dets[m] = MaxRatioDetector(fs["percentile"]).fit(Z)
            fold.cal_scores[m] = fold.dets[m].score(Z)
            fold.thr[m] = threshold_at_fpr(fold.cal_scores[m], fs["target_fpr"])
        fold.cal_scores[B1] = np.concatenate([timestamp_check_score(df.tau.values, fold.period)
                                              for df in cal_streams.values()])
        fold.thr[B1] = threshold_at_fpr(fold.cal_scores[B1], fs["target_fpr"])
        sf = fs.get("seam_alarm_fpr")
        fold.thr["seam_alarm"] = threshold_at_fpr(fold.cal_scores["B2: Seam only"], sf) if sf else float("inf")
        return fold

    # --- scoring ---
    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        fs = self.fs
        df = df.copy()
        r_v, r_w, checked = motion_features(df.v_hat, df.w_hat, df.v_bar, df.w_bar,
                                            period_ratio=fs["period_s"] / self.period,
                                            k=int(fs.get("smooth_frames", 1)), v0=fs["v0"],
                                            v_min=fs.get("v_min", 0.0), w0=fs.get("w0"))
        df["r_v"], df["r_w"], df["checked"] = r_v, r_w, checked.astype(int)
        df["one_minus_fit"] = 1.0 - df["fit"].astype(float)
        return df

    def threshold(self, method: str, fpr: float | None = None) -> float:
        if fpr is None:
            return self.thr[method]
        return threshold_at_fpr(self.cal_scores[method], fpr)

    def score(self, df: pd.DataFrame, fpr: float | None = None, m: int | None = None,
              n: int | None = None, seam_fpr: float | None = None) -> pd.DataFrame:
        """Score one stream. Adds score::, thr::, alarm:: columns, `alarm` (FreshScan) and `trust`."""
        fs = self.fs
        m, n = m or fs["m"], n or fs["n"]
        df = self.features(df[df.slot > 0].reset_index(drop=True) if "slot" in df else df)
        for meth, cols in METHODS.items():
            df[f"score::{meth}"] = self.dets[meth].score(df[cols].values)
            df[f"thr::{meth}"] = self.threshold(meth, fpr)
        df[f"score::{B1}"] = timestamp_check_score(df.tau.values, self.period)
        df[f"thr::{B1}"] = self.threshold(B1, fpr)
        t_seam = self.thr["seam_alarm"] if seam_fpr is None else threshold_at_fpr(
            self.cal_scores["B2: Seam only"], seam_fpr)
        seam_alarm = df["score::B2: Seam only"].values > t_seam
        persist = {meth: m_of_n(df[f"score::{meth}"].values > df[f"thr::{meth}"].values, m, n)
                   for meth in METHODS}
        df["seam_alarm"] = seam_alarm.astype(int)
        df["alarm::B2: Seam only"] = seam_alarm.astype(int)                 # B2: seam rule only
        df["alarm::B3: Residuals only"] = persist["B3: Residuals only"].astype(int)
        df["alarm::FreshScan"] = (persist["FreshScan"] | seam_alarm).astype(int)
        df["alarm::B1: Timestamp check"] = m_of_n(df[f"score::{B1}"].values > df[f"thr::{B1}"].values,
                                                  m, n).astype(int)
        df["alarm"] = df["alarm::FreshScan"]
        df["trust"] = trust(df["score::FreshScan"].values)
        df["fold"] = self.name
        return df

    # --- (de)serialization ---
    def meta(self) -> dict:
        return {"calibration_drives": self.cal_drives, "lidar_period_s": self.period,
                "scales": {m: [float(x) for x in d.scale] for m, d in self.dets.items()},
                "thresholds": self.thr}


def save_checkpoint(path_dir: str, cfg: dict, folds: dict, protocol: str, run_info: dict | None = None) -> str:
    os.makedirs(path_dir, exist_ok=True)
    ck = {"freshscan_checkpoint_version": VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
          "protocol": protocol, "freshscan": cfg["freshscan"], "replay": cfg["replay"],
          "calibration_drives": cfg["calibration_drives"], "test_drives": cfg["test_drives"],
          "methods": METHODS, "folds": {name: f.meta() for name, f in folds.items()},
          "run_info": run_info or {}}
    with open(os.path.join(path_dir, "freshscan_checkpoint.json"), "w") as fh:
        json.dump(ck, fh, indent=2)
    arrays = {f"{name}__{_key(m)}": np.asarray(s) for name, f in folds.items() for m, s in f.cal_scores.items()}
    np.savez_compressed(os.path.join(path_dir, "calibration_scores.npz"), **arrays)
    return path_dir


class Checkpoint:
    def __init__(self, path_dir: str):
        with open(os.path.join(path_dir, "freshscan_checkpoint.json")) as fh:
            self.meta = json.load(fh)
        cal = np.load(os.path.join(path_dir, "calibration_scores.npz"))
        fs = self.meta["freshscan"]
        self.folds = {}
        for name, fm in self.meta["folds"].items():
            scores = {m: cal[f"{name}__{_key(m)}"] for m in list(METHODS) + [B1]}
            self.folds[name] = Fold(name, fs, fm["calibration_drives"], fm["lidar_period_s"],
                                    fm["scales"], fm["thresholds"], scores, fs["percentile"])

    def fold_for(self, drive: str) -> Fold:
        return self.folds.get(drive, self.folds.get("all"))


def load_checkpoint(path_dir: str) -> Checkpoint:
    return Checkpoint(path_dir)
