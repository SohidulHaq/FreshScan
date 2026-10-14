"""Step 5 (any time later): evaluate from the saved checkpoint, without the raw KITTI scans.

Uses results/checkpoint/ (written by 02_evaluate.py) and the per-frame feature CSVs of step 1.
Lets you try other operating points without recomputing anything expensive:
    * other false-positive rates (the threshold is re-set on the stored calibration scores)
    * other alarm rules (m of the last n frames) and seam-alarm rates
and reports detection, delay and false alarms per method, replay age and test drive.

Outputs (results/tables/):
    checkpoint_eval.csv            one row per (method, target FPR, replay age)
    checkpoint_eval_per_drive.csv  FreshScan per test drive and replay age

Usage (from the repository root):
    python scripts/05_evaluate_checkpoint.py                         # FPR 1%, 0.5%, 0.1%
    python scripts/05_evaluate_checkpoint.py --fpr 0.001 --m 3 --n 5
    python scripts/05_evaluate_checkpoint.py --features other_results/features   # new data, same fit
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import load_config, out_dir  # noqa: E402
from freshscan.checkpoint import B1, load_checkpoint

METHODS = [B1, "B2: Seam only", "B3: Residuals only", "FreshScan"]


def events(alarm: np.ndarray) -> int:
    a = np.asarray(alarm, dtype=int)
    return int(((a == 1) & (np.r_[0, a[:-1]] == 0)).sum())


def window_delays(label: np.ndarray, alarm: np.ndarray) -> list:
    starts = np.where(np.diff(np.r_[0, label]) == 1)[0]
    ends = np.where(np.diff(np.r_[label, 0]) == -1)[0]
    out = []
    for s, e in zip(starts, ends):
        hit = np.where(alarm[s:e + 1])[0]
        out.append(int(hit[0]) if len(hit) else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None, help="checkpoint folder (default: results/checkpoint)")
    ap.add_argument("--features", default=None, help="feature CSV folder (default: results/features)")
    ap.add_argument("--fpr", type=float, nargs="+", default=[0.01, 0.005, 0.001])
    ap.add_argument("--seam-fpr", type=float, default=None, help="seam-alarm rate (default: as calibrated)")
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ck = load_checkpoint(args.checkpoint or os.path.join(cfg["results_dir"], "checkpoint"))
    fs = ck.meta["freshscan"]
    m_, n_ = args.m or fs["m"], args.n or fs["n"]
    feat_dir = args.features or os.path.join(cfg["results_dir"], "features")
    test_drives = ck.meta["test_drives"]
    raw = {}
    for f in sorted(glob.glob(os.path.join(feat_dir, "*.csv"))):
        drive, setting = os.path.basename(f)[:-4].split("__")
        if drive in test_drives:
            raw[(drive, setting)] = pd.read_csv(f)
    if not raw:
        raise FileNotFoundError(f"No test-drive feature CSVs in {feat_dir}")
    print(f"Checkpoint from {ck.meta['created']} | protocol {ck.meta['protocol']} | {len(raw)} streams | "
          f"alarm rule {m_} of {n_} + seam alarm")

    rows, per_drive = [], []
    for fpr in args.fpr:
        streams = {k: ck.fold_for(k[0]).score(df, fpr=fpr, m=m_, n=n_, seam_fpr=args.seam_fpr)
                   for k, df in raw.items()}
        benign = [k for k in streams if k[1] == "benign"]
        ages = sorted({int(k[1].split("_")[1][1:]) for k in streams if k[1].startswith("replay")})
        period = np.mean([ck.fold_for(k[0]).period for k in benign])
        minutes = sum(len(streams[k]) for k in benign) * period / 60.0
        for meth in METHODS:
            col, tcol = f"score::{meth}", f"thr::{meth}"
            ben = np.concatenate([streams[k][col].values for k in benign])
            ben_hit = np.concatenate([streams[k][col].values > streams[k][tcol].values for k in benign])
            acol = f"alarm::{meth}"
            fa = sum(events(streams[k][acol].values) for k in benign)
            for R in ages:
                keys = [k for k in streams if k[1].startswith(f"replay_R{R}_")]
                att = np.concatenate([streams[k].loc[streams[k].label == 1, col].values for k in keys])
                hits = np.concatenate([(streams[k][col] > streams[k][tcol]).values[streams[k].label.values == 1]
                                       for k in keys])
                delays = []
                for k in keys:
                    wd = window_delays(streams[k].label.values, streams[k][acol].values.astype(bool))
                    delays += wd
                    if meth == "FreshScan":
                        lab = streams[k].label.values == 1
                        per_drive.append(dict(target_fpr=fpr, drive=k[0], replay_age_s=R,
                                              tpr=float(np.mean(streams[k][col].values[lab] > streams[k][tcol].values[lab])),
                                              windows_detected_pct=100.0 * np.mean([d is not None for d in wd])))
                hit = [d for d in delays if d is not None]
                y = np.r_[np.zeros(len(ben)), np.ones(len(att))]
                rows.append(dict(method=meth, target_fpr=fpr, replay_age_s=R,
                                 auroc=float(roc_auc_score(y, np.r_[ben, att])),
                                 benign_fpr_on_test=float(ben_hit.mean()), tpr=float(hits.mean()),
                                 windows_detected_pct=100.0 * len(hit) / max(len(delays), 1),
                                 median_delay_frames=float(np.median(hit)) if hit else float("nan"),
                                 p95_delay_frames=float(np.percentile(hit, 95)) if hit else float("nan"),
                                 false_alarm_events_per_min=fa / max(minutes, 1e-9)))
    tab = out_dir(cfg, "tables")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(tab, "checkpoint_eval.csv"), index=False)
    pd.DataFrame(per_drive).groupby(["target_fpr", "drive", "replay_age_s"], as_index=False)[
        ["tpr", "windows_detected_pct"]].mean().to_csv(os.path.join(tab, "checkpoint_eval_per_drive.csv"), index=False)
    show = df[df.method == "FreshScan"][["target_fpr", "replay_age_s", "auroc", "tpr", "benign_fpr_on_test",
                                          "windows_detected_pct", "median_delay_frames", "p95_delay_frames",
                                          "false_alarm_events_per_min"]]
    print(show.round(3).to_string(index=False))
    print(f"Written to {tab}/checkpoint_eval.csv and checkpoint_eval_per_drive.csv")


if __name__ == "__main__":
    main()
