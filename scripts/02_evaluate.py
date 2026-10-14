"""Step 2: calibrate FreshScan on benign drives, score every test stream, compute every result table.

Protocol (config `protocol`): leave_one_drive_out (default) calibrates each test drive on the benign
streams of all other drives; fixed calibrates once on `calibration_drives`. No attack data is used.
The calibration fits are saved as results/checkpoint/ (see freshscan/checkpoint.py).

Outputs (results/tables/):
    rq1_detection.csv    RQ1: AUROC and TPR@1% per method and replay age
    rq2_delay.csv        RQ2: detection delay, windows detected, false alarms per minute
    rq3_analysis.csv     RQ3: TPR@1% of FreshScan split by seam, speed difference, speed
    rq4_runtime.csv      RQ4: mean and p95 time per frame for each step
    dataset_summary.csv  numbers for the "Dataset" paragraph
    scores/<drive>__<setting>.csv  per-frame scores, alarms and trust (used for figures)

Usage (from the repository root):  python scripts/02_evaluate.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import load_config, out_dir  # noqa: E402
from freshscan.checkpoint import B1, METHODS, Fold, save_checkpoint


def load_streams(feat_dir: str) -> dict:
    out = {}
    for f in sorted(glob.glob(os.path.join(feat_dir, "*.csv"))):
        drive, setting = os.path.basename(f)[:-4].split("__")
        df = pd.read_csv(f)
        # slot 0 has no previous frame, so no LiDAR motion: leave it out of all statistics
        out[(drive, setting)] = df[df.slot > 0].reset_index(drop=True)
    if not out:
        raise FileNotFoundError(f"No feature CSVs in {feat_dir}. Run 01_compute_features.py first.")
    return out


def replay_age(setting: str):
    return int(setting.split("_")[1][1:]) if setting.startswith("replay") else None


def tpr_at(scores_att, thr):
    return float(np.mean(scores_att > thr)) if len(scores_att) else float("nan")


def auroc(ben, att):
    if len(ben) == 0 or len(att) == 0:
        return float("nan")
    y = np.r_[np.zeros(len(ben)), np.ones(len(att))]
    return float(roc_auc_score(y, np.r_[ben, att]))


def window_stats(df, alarm):
    """Detection delay (frames from window start to first alarm) for each window."""
    lab = df.label.values
    starts = np.where(np.diff(np.r_[0, lab]) == 1)[0]
    ends = np.where(np.diff(np.r_[lab, 0]) == -1)[0]
    delays = []
    for s, e in zip(starts, ends):
        hit = np.where(alarm[s:e + 1])[0]
        delays.append(int(hit[0]) if len(hit) else None)
    return delays


def events(alarm) -> int:
    a = np.asarray(alarm, dtype=int)
    return int(((a == 1) & (np.r_[0, a[:-1]] == 0)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    fs = cfg["freshscan"]
    protocol = cfg.get("protocol", "leave_one_drive_out")
    streams = load_streams(out_dir(cfg, "features"))
    tab_dir, score_dir = out_dir(cfg, "tables"), out_dir(cfg, "tables", "scores")

    benign = {k[0]: streams[k] for k in streams if k[1] == "benign"}
    test_drives = [d for d in cfg["test_drives"] if d in benign]
    calib_drives = [d for d in cfg["calibration_drives"] if d in benign]
    if not test_drives or not calib_drives:
        raise RuntimeError("Need benign streams of calibration drives and of test drives.")

    # ---- calibration folds (benign data only) ----
    folds = {}
    for td in test_drives:
        cal = [d for d in calib_drives + test_drives if d != td] if protocol == "leave_one_drive_out" \
            else calib_drives
        folds[td] = Fold.fit(td, fs, {d: benign[d] for d in cal})
    folds["all"] = Fold.fit("all", fs, dict(benign))          # for new data (not used in the tables)
    for td in test_drives:
        f = folds[td]
        print(f"[{td}] calibrated on {len(f.cal_drives)} drives, scan period {f.period * 1e3:.2f} ms, "
              f"eta {f.thr['FreshScan']:.3f}, seam-alarm threshold {f.thr['seam_alarm']:.3f}")

    # ---- score every test stream with its fold ----
    scored = {}
    t0 = time.perf_counter(); n_scored = 0
    for k in streams:
        if k[0] not in test_drives:
            continue
        df = folds[k[0]].score(streams[k])
        scored[k] = df
        n_scored += len(df)
        df.to_csv(os.path.join(score_dir, f"{k[0]}__{k[1]}.csv"), index=False)
    score_ms = 1e3 * (time.perf_counter() - t0) / max(n_scored, 1)

    test = list(scored)
    benign_test = [k for k in test if k[1] == "benign"]
    ages = sorted({replay_age(k[1]) for k in test if k[1].startswith("replay")})
    all_methods = [B1] + list(METHODS)
    period = float(np.mean([folds[d].period for d in test_drives]))
    minutes = sum(len(scored[k]) for k in benign_test) * period / 60.0

    def pooled(method, keys, attacked):
        sc, hit = [], []
        for k in keys:
            d = scored[k]
            sel = (d.label == 1).values if attacked else np.ones(len(d), bool)
            sc.append(d.loc[sel, f"score::{method}"].values)
            hit.append(d.loc[sel, f"score::{method}"].values > d.loc[sel, f"thr::{method}"].values)
        return np.concatenate(sc), np.concatenate(hit)

    # ---- RQ1: frame-level detection ----
    rq1 = []
    for m in all_methods:
        ben, ben_hit = pooled(m, benign_test, False)
        for age in ages:
            att, att_hit = pooled(m, [k for k in test if replay_age(k[1]) == age], True)
            rq1.append(dict(method=m, replay_age_s=age, auroc=auroc(ben, att), tpr_at_1pct=float(att_hit.mean()),
                            benign_fpr_on_test=float(ben_hit.mean())))
    pd.DataFrame(rq1).to_csv(os.path.join(tab_dir, "rq1_detection.csv"), index=False)

    # ---- RQ2: windows, delay, false alarm events ----
    rq2 = []
    for m in list(METHODS):
        fa = sum(events(scored[k][f"alarm::{m}"].values) for k in benign_test)
        for age in ages:
            delays = []
            for k in test:
                if replay_age(k[1]) == age:
                    delays += window_stats(scored[k], scored[k][f"alarm::{m}"].values.astype(bool))
            hit = [d for d in delays if d is not None]
            rq2.append(dict(method=m, replay_age_s=age,
                            median_delay_frames=float(np.median(hit)) if hit else float("nan"),
                            p95_delay_frames=float(np.percentile(hit, 95)) if hit else float("nan"),
                            windows_detected_pct=100.0 * len(hit) / max(len(delays), 1),
                            n_windows=len(delays),
                            false_alarms_per_min=fa / max(minutes, 1e-9)))
    pd.DataFrame(rq2).to_csv(os.path.join(tab_dir, "rq2_delay.csv"), index=False)

    # ---- RQ3 (FreshScan, one replay age): frame-level TPR split as in the detectability analysis ----
    age = cfg["rq3"]["replay_age_s"]
    att = pd.concat([scored[k][scored[k].label == 1] for k in test if replay_age(k[1]) == age])
    hit = att["score::FreshScan"].values > att["thr::FreshScan"].values
    dv = np.abs(att.v_bar.values - att.v_bar_src.values)
    sp = np.abs(att.v_bar.values)
    e1, e2 = cfg["rq3"]["dv_bins"]; s1, s2 = cfg["rq3"]["speed_bins"]
    splits = [
        ("Position in window", "Seam (entry and exit)", att.seam.values == 1),
        ("Position in window", "Inside", att.seam.values == 0),
        ("|v(t)-v(t-R)|", f"<{e1} m/s", dv < e1),
        ("|v(t)-v(t-R)|", f"{e1}-{e2} m/s", (dv >= e1) & (dv <= e2)),
        ("|v(t)-v(t-R)|", f">{e2} m/s", dv > e2),
        ("Current speed", f"<{s1} m/s", sp < s1),
        ("Current speed", f"{s1}-{s2} m/s", (sp >= s1) & (sp <= s2)),
        ("Current speed", f">{s2} m/s", sp > s2),
    ]
    rq3 = [dict(split=a, bin=b, share_of_attacked_pct=100.0 * sel.mean(),
                tpr_at_1pct=float(hit[sel].mean()) if sel.any() else float("nan")) for a, b, sel in splits]
    pd.DataFrame(rq3).to_csv(os.path.join(tab_dir, "rq3_analysis.csv"), index=False)

    # ---- RQ4: run time ----
    allf = pd.concat([streams[k] for k in test])
    prep = allf.prep_ms[allf.prep_ms > 0]; icpm = allf.icp_ms[allf.icp_ms > 0]
    check = allf.check_ms + score_ms
    rq4 = [dict(step="Preparation (C1)", mean_ms=prep.mean(), p95_ms=prep.quantile(0.95)),
           dict(step="ICP (C2)", mean_ms=icpm.mean(), p95_ms=icpm.quantile(0.95)),
           dict(step="Checks and decision (C3, C5)", mean_ms=check.mean(), p95_ms=check.quantile(0.95))]
    tot = prep.mean() + icpm.mean() + check.mean()
    rq4.append(dict(step="Total", mean_ms=tot, p95_ms=prep.quantile(0.95) + icpm.quantile(0.95) + check.quantile(0.95)))
    pd.DataFrame(rq4).to_csv(os.path.join(tab_dir, "rq4_runtime.csv"), index=False)

    # ---- dataset summary ----
    def frames(keys): return int(sum(len(streams[k]) for k in keys))
    cal_frames = [sum(len(benign[d]) for d in folds[td].cal_drives) for td in test_drives]
    _, ben_hit = pooled("FreshScan", benign_test, False)
    summary = dict(protocol=protocol, calibration_drives=len(calib_drives),
                   drives_per_fold=int(np.mean([len(folds[td].cal_drives) for td in test_drives])),
                   calibration_frames_per_fold_mean=int(np.mean(cal_frames)),
                   test_drives=len(benign_test), test_frames=frames(benign_test),
                   test_minutes=round(minutes, 1), seeds=len(cfg["replay"]["seeds"]),
                   threshold_eta_mean=float(np.mean([folds[d].thr["FreshScan"] for d in test_drives])),
                   seam_threshold_mean=float(np.mean([folds[d].thr["seam_alarm"] for d in test_drives])),
                   lidar_period_ms=round(period * 1e3, 2),
                   benign_test_checked_pct=round(100.0 * np.mean(np.concatenate(
                       [scored[k].checked.values for k in benign_test])), 1),
                   benign_test_fpr_pct=round(100.0 * float(ben_hit.mean()), 2))
    pd.DataFrame([summary]).to_csv(os.path.join(tab_dir, "dataset_summary.csv"), index=False)
    with open(os.path.join(tab_dir, "thresholds.json"), "w") as fh:
        json.dump({name: f.thr for name, f in folds.items()}, fh, indent=2)
    print(f"Tables written to {tab_dir}")

    run_info = {}
    for f in sorted(glob.glob(os.path.join(cfg["results_dir"], "run_info", "*.json"))):
        run_info[os.path.basename(f)[:-5]] = json.load(open(f))
    ck = save_checkpoint(out_dir(cfg, "checkpoint"), cfg, folds, protocol, run_info)
    print(f"Checkpoint written to {ck}")


if __name__ == "__main__":
    main()
