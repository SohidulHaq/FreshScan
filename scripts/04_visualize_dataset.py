"""Step 4: dataset and behaviour figures.

Two modes:
  1. --save-samples DRIVE ...   (needs the raw KITTI scans; the notebook calls it right after step 1
     of each drive, before the scans are deleted) saves a few scans of each drive to
     results/dataset_samples/<drive>.npz so the bird's-eye-view figure can be drawn later.
  2. default: makes all figures from the feature CSVs, scores and checkpoint (no raw scans needed):
       fig_ds_overview          speed and yaw rate of every drive over time (normal-only vs test)
       fig_ds_distributions     speed / yaw-rate histograms, share of time below the abstain speed
       fig_ds_trajectories      driven path of every drive (integrated from IMU/GPS speed and yaw rate)
       fig_ds_motion_agreement  LiDAR (ICP) motion vs IMU/GPS motion on benign frames
       fig_ds_features          benign vs replayed distributions of every feature and the final score
       fig_ds_replay_example    one replay stream: delivered frame index, speeds, score and alarms
       fig_ds_bev               bird's-eye view: a fresh scan pair vs a replayed (10 s old) pair
     and tables/dataset_drives.csv (one row per drive: frames, minutes, km, speeds, abstain share, ICP fit).

Figures go to results/figures/dataset/ as PDF and PNG.

Usage (from the repository root):
    python scripts/04_visualize_dataset.py
    python scripts/04_visualize_dataset.py --save-samples 2011_09_30_drive_0018
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import load_config, out_dir  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "lines.linewidth": 0.9,
                     "legend.fontsize": 7, "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": 0.3})


SCALE = 1.0          # --hires: larger text for high-resolution PNGs
DPI = 180


def fsz(x):
    return x * SCALE


def save(fig, d, name, layout=True):
    if layout:
        fig.tight_layout()
    for ext in (("png",) if DPI > 200 else ("pdf", "png")):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), dpi=DPI)
    plt.close(fig)
    print("  wrote", name)


# ----------------------------------------------------------------------------------------------
def save_samples(cfg, drives):
    from freshscan.kitti_loader import load_drive
    from freshscan.registration import prepare
    fs = cfg["freshscan"]
    d = out_dir(cfg, "dataset_samples")
    lag = int(round(10.0 / fs["period_s"]))                 # "replayed" scan: 10 s earlier
    for name in drives:
        drv = load_drive(cfg["kitti_root"], name, fs["period_s"])
        sp = np.asarray(drv.speed)
        cand = np.where((sp > 5.0) & (np.arange(len(sp)) > lag + 1))[0]
        i = int(cand[len(cand) // 2]) if len(cand) else len(sp) // 2
        i = max(i, lag + 1)
        np.savez_compressed(os.path.join(d, f"{name}.npz"),
                            index=i, lag=lag, speed=sp[i], speed_old=sp[i - lag],
                            bev=prepare(drv.frame(i), fs["r_min"], fs["r_max"], 0.2).astype(np.float32),
                            cur=prepare(drv.frame(i), fs["r_min"], fs["r_max"], fs["voxel"]).astype(np.float32),
                            prev=prepare(drv.frame(i - 1), fs["r_min"], fs["r_max"], fs["voxel"]).astype(np.float32),
                            old=prepare(drv.frame(i - lag), fs["r_min"], fs["r_max"], fs["voxel"]).astype(np.float32))
        print(f"[{name}] saved sample scans (frame {i}, speed {sp[i]:.1f} m/s)")


# ----------------------------------------------------------------------------------------------
def load_benign(cfg):
    out = {}
    for f in sorted(glob.glob(os.path.join(cfg["results_dir"], "features", "*__benign.csv"))):
        name = os.path.basename(f).split("__")[0]
        df = pd.read_csv(f)
        out[name] = df[df.slot > 0].reset_index(drop=True)
    return out


def role(cfg, name):
    return "normal-only" if name in cfg["calibration_drives"] else "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--save-samples", nargs="+", default=None, metavar="DRIVE")
    ap.add_argument("--hires", action="store_true",
                    help="high-resolution PNGs (800 dpi, larger text)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    global SCALE, DPI
    if args.hires:
        SCALE, DPI = 1.55, 800
        plt.rcParams.update({"font.size": fsz(8), "legend.fontsize": fsz(7), "xtick.labelsize": fsz(7),
                             "ytick.labelsize": fsz(7), "axes.titlesize": fsz(8), "axes.labelsize": fsz(8)})
    if args.save_samples:
        save_samples(cfg, args.save_samples)
        return

    fs = cfg["freshscan"]
    figd = out_dir(cfg, "figures", "hires_png" if args.hires else "dataset")
    ck_path = os.path.join(cfg["results_dir"], "checkpoint", "freshscan_checkpoint.json")
    ck = json.load(open(ck_path)) if os.path.exists(ck_path) else None
    period = ck["folds"]["all"]["lidar_period_s"] if ck and "folds" in ck else fs["period_s"]
    ratio = fs["period_s"] / period
    v_min = fs.get("v_min", 0.0)
    benign = load_benign(cfg)
    if not benign:
        raise FileNotFoundError("No benign feature CSVs. Run step 1 first.")
    names = list(benign)
    print(f"Dataset figures for {len(names)} drives -> {figd}")

    # ---- table: one row per drive ----
    rows = []
    for n in names:
        d = benign[n]
        rows.append(dict(drive=n, role=role(cfg, n), frames=len(d) + 1,
                         minutes=round((len(d) + 1) * period / 60, 2),
                         distance_km=round(float(np.sum(np.abs(d.v_bar)) * period / 1000), 2),
                         mean_speed_mps=round(float(d.v_bar.mean()), 2), max_speed_mps=round(float(d.v_bar.max()), 2),
                         below_vmin_pct=round(100 * float(np.mean(np.abs(d.v_bar) < v_min)), 1),
                         mean_abs_yaw_deg_s=round(float(np.degrees(np.abs(d.w_bar)).mean()), 2),
                         icp_fit_median=round(float(d.fit.median()), 3), icp_fit_min=round(float(d.fit.min()), 3)))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out_dir(cfg, "tables"), "dataset_drives.csv"), index=False)
    print(tab.to_string(index=False))

    # ---- 1. overview ----
    fig, axes = plt.subplots(len(names), 1, figsize=(7.2, (1.05 if SCALE > 1 else 1.25) * len(names) + 0.4), sharex=False, squeeze=False)
    for ax, n in zip(axes[:, 0], names):
        d = benign[n]; t = d.slot.values * period
        ax.plot(t, d.v_bar, color="black", label="speed [m/s]")
        ax2 = ax.twinx(); ax2.plot(t, np.degrees(d.w_bar), color="tab:blue", lw=0.6, label="yaw rate [deg/s]")
        ax2.set_ylabel("deg/s", fontsize=fsz(7)); ax2.grid(False)
        ax.axhline(v_min, color="tab:red", ls=":", lw=0.7)
        ax.set_ylabel("m/s"); ax.set_title(f"{n} ({role(cfg, n)})", fontsize=fsz(8), loc="left")
    axes[-1, 0].set_xlabel("time [s]")
    save(fig, figd, "fig_ds_overview")

    # ---- 2. distributions ----
    cal = pd.concat([benign[n] for n in names if role(cfg, n) == "normal-only"] or [pd.DataFrame(columns=["v_bar", "w_bar"])])
    tst = pd.concat([benign[n] for n in names if role(cfg, n) == "test"] or [pd.DataFrame(columns=["v_bar", "w_bar"])])
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6 if SCALE > 1 else 2.2))
    bins = np.linspace(-1, max(30, float(pd.concat([cal.v_bar, tst.v_bar]).max()) + 1), 40)
    for df, lab, st in [(cal, "normal-only", "step"), (tst, "test", "stepfilled")]:
        if len(df):
            ax[0].hist(df.v_bar, bins=bins, density=True, histtype=st, alpha=0.5 if st == "stepfilled" else 1, label=lab)
            ax[1].hist(np.degrees(np.abs(df.w_bar)), bins=40, density=True, histtype=st,
                       alpha=0.5 if st == "stepfilled" else 1, label=lab, log=True)
    ax[0].axvline(v_min, color="tab:red", ls=":"); ax[0].set_xlabel("speed [m/s]")
    (ax[1] if SCALE > 1 else ax[0]).legend(loc="upper right")
    ax[1].set_xlabel("|yaw rate| [deg/s]")
    ax[2].barh(tab.drive.str.replace("2011_", "").str.replace("_drive_", " / "), tab.below_vmin_pct,
               color=["0.3" if r == "normal-only" else "0.7" for r in tab.role])
    ax[2].set_xlabel(f"% time below {v_min:g} m/s"); ax[2].tick_params(axis="y", labelsize=fsz(6))
    save(fig, figd, "fig_ds_distributions")

    # ---- 3. trajectories (dead reckoning from IMU/GPS speed and yaw rate) ----
    if SCALE > 1:
        ncol = (len(names) + 1) // 2
        fig, axes = plt.subplots(2, ncol, figsize=(7.2, 4.4), squeeze=False)
        for extra in axes.ravel()[len(names):]:
            extra.axis("off")
        axes = axes.ravel()[None, :len(names)]
    else:
        fig, axes = plt.subplots(1, len(names), figsize=(1.6 * len(names), 2.0), squeeze=False)
    for ax, n in zip(axes[0], names):
        d = benign[n]
        yaw = np.cumsum(d.w_bar.values) * period
        x = np.cumsum(d.v_bar.values * np.cos(yaw)) * period
        y = np.cumsum(d.v_bar.values * np.sin(yaw)) * period
        ax.plot(x, y, color="black" if role(cfg, n) == "normal-only" else "tab:blue")
        ax.plot(x[:1], y[:1], "o", ms=3, color="tab:green")
        ax.set_aspect("equal", "datalim")
        if SCALE > 1:
            km = float(np.sum(np.abs(d.v_bar)) * period / 1000)
            ax.set_title(f"{n[:10].replace('2011_', '')} / {n[-4:]}\n{role(cfg, n)}, {km:.1f} km", fontsize=fsz(6))
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.set_title(n.replace("2011_", "").replace("_drive_", "\n"), fontsize=fsz(6))
            ax.tick_params(labelsize=fsz(5))
    save(fig, figd, "fig_ds_trajectories")

    # ---- 4. LiDAR vs IMU/GPS motion agreement (benign) ----
    allb = pd.concat(benign.values())
    vh, wh = allb.v_hat * ratio, allb.w_hat * ratio
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.6)) if SCALE > 1 else plt.subplots(1, 4, figsize=(7.2, 2.0))
    ax = np.ravel(ax)
    ax[0].hexbin(allb.v_bar, vh, gridsize=50, bins="log", cmap="Greys", mincnt=1)
    lim = [min(allb.v_bar.min(), vh.min()), max(allb.v_bar.max(), vh.max())]
    ax[0].plot(lim, lim, "r--", lw=0.6); ax[0].set_xlabel("IMU/GPS speed [m/s]"); ax[0].set_ylabel("LiDAR speed [m/s]")
    ax[1].hist(vh - allb.v_bar, bins=80, color="0.4"); ax[1].set_xlabel("speed difference [m/s]"); ax[1].set_ylabel("frames")
    ax[2].hexbin(np.degrees(allb.w_bar), np.degrees(wh), gridsize=50, bins="log", cmap="Greys", mincnt=1)
    lw = np.degrees([allb.w_bar.min(), allb.w_bar.max()])
    ax[2].plot(lw, lw, "r--", lw=0.6); ax[2].set_xlabel("IMU yaw rate [deg/s]"); ax[2].set_ylabel("LiDAR yaw rate [deg/s]")
    ax[3].hist(np.degrees(wh - allb.w_bar), bins=80, color="0.4"); ax[3].set_xlabel("yaw-rate difference [deg/s]"); ax[3].set_ylabel("frames")
    if SCALE > 1:
        ax[3].set_xlim(-6, 6)
    r_v = np.corrcoef(allb.v_bar, vh)[0, 1]; r_w = np.corrcoef(allb.w_bar, wh)[0, 1]
    ax[0].set_title(f"corr {r_v:.3f}", fontsize=fsz(7)); ax[2].set_title(f"corr {r_w:.3f}", fontsize=fsz(7))
    save(fig, figd, "fig_ds_motion_agreement")

    # ---- 5. feature distributions: benign vs replay (uses the scored streams of step 2) ----
    sc_dir = os.path.join(cfg["results_dir"], "tables", "scores")
    thr_path = os.path.join(cfg["results_dir"], "tables", "thresholds.json")
    scored = {os.path.basename(f)[:-4]: pd.read_csv(f) for f in sorted(glob.glob(os.path.join(sc_dir, "*.csv")))}
    if scored:
        thr = float(np.median(pd.concat(scored.values()).groupby("fold")["thr::FreshScan"].first()))  # median over folds
        R = cfg["rq3"]["replay_age_s"]
        ben = pd.concat([v for k, v in scored.items() if k.endswith("__benign")])
        att = pd.concat([v[v.label == 1] for k, v in scored.items() if f"__replay_R{R}_" in k])
        feats = [("r_v", "speed residual"), ("r_w", "yaw-rate residual"),
                 ("one_minus_fit", "1 - ICP fitness"), ("rmse", "ICP inlier RMSE [m]"),
                 ("score::FreshScan", "score" if SCALE > 1 else "FreshScan score")]
        fig, ax = plt.subplots(1, len(feats), figsize=(7.2, 2.5 if SCALE > 1 else 1.9))
        for a, (c, lab) in zip(ax, feats):
            hi = np.nanpercentile(np.r_[ben[c], att[c]], 99.5)
            b = np.linspace(0, hi if hi > 0 else 1, 40)
            a.hist(np.clip(ben[c], 0, b[-1]), bins=b, density=True, histtype="stepfilled", color="0.75", label="benign")
            a.hist(np.clip(att[c], 0, b[-1]), bins=b, density=True, histtype="step", color="black", label=f"replay {R} s")
            a.set_yscale("log"); a.set_xlabel(lab, fontsize=fsz(7)); a.tick_params(labelsize=fsz(6))
            if c == "score::FreshScan":
                a.axvline(thr, color="tab:red", ls="--", lw=0.8)
        if SCALE > 1:
            h, l = ax[0].get_legend_handles_labels()
            fig.tight_layout(rect=(0, 0, 1, 0.84))
            fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
        else:
            ax[0].legend(fontsize=fsz(6))
        save(fig, figd, "fig_ds_features", layout=SCALE == 1)

        # ---- 6. replay example ----
        key = next((k for k in scored if f"__replay_R{R}_s0" in k), None)
        if key:
            d = scored[key]; t = d.slot.values * period
            fig, ax = plt.subplots(3, 1, figsize=(7.2, 5.6 if SCALE > 1 else 4.2), sharex=True)
            for a in ax:
                lab = d.label.values
                for s0, e0 in zip(np.where(np.diff(np.r_[0, lab]) == 1)[0], np.where(np.diff(np.r_[lab, 0]) == -1)[0]):
                    a.axvspan(t[s0], t[e0], color="0.88", lw=0)
            ax[0].plot(t, d.slot, color="0.6", ls="--", label="expected (fresh)")
            ax[0].plot(t, d.sigma, color="black", label="delivered frame")
            ax[0].set_ylabel("frame index"); ax[0].legend(loc="upper left")
            ax[1].plot(t, d.v_bar, color="black", label="IMU/GPS speed now")
            ax[1].plot(t, d.v_bar_src, color="tab:blue", lw=0.7, label="speed when delivered frame was captured")
            ax[1].set_ylabel("m/s"); ax[1].legend(loc="upper left")
            ax[2].plot(t, d["score::FreshScan"], color="black", label="FreshScan score")
            thr_d = float(d["thr::FreshScan"].iloc[0])
            ax[2].axhline(thr_d, color="tab:red", ls="--", lw=0.8, label="threshold")
            al = d.alarm.values == 1
            ax[2].plot(t[al], np.full(al.sum(), thr_d * 1.05), "|", color="tab:red", ms=6, label="alarm")
            sa = d.seam_alarm.values == 1
            ax[2].plot(t[sa], np.full(sa.sum(), thr_d * 2.0), "v", color="tab:blue", ms=4, label="seam alarm")
            ax[2].set_yscale("log"); ax[2].set_ylabel("score"); ax[2].set_xlabel("time [s]")
            if SCALE > 1:
                for a, nc in zip(ax, (2, 2, 4)):
                    a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=nc, frameon=False, borderaxespad=0.1,
                             columnspacing=1.0, handlelength=1.6)
            else:
                ax[2].legend(loc="upper left")
                ax[0].set_title(key.replace("__", "  "), fontsize=fsz(8), loc="left")
            save(fig, figd, "fig_ds_replay_example")

    # ---- 7. bird's-eye view from saved samples ----
    samples = sorted(glob.glob(os.path.join(cfg["results_dir"], "dataset_samples", "*.npz")))
    if samples:
        z = np.load(samples[0]); name = os.path.basename(samples[0])[:-4]
        from scipy.spatial import cKDTree
        def overlap(a, b):
            d, _ = cKDTree(b).query(a, distance_upper_bound=fs_cfg["d_max"])
            return float(np.isfinite(d).mean())
        fs_cfg = cfg["freshscan"]
        print(f"  BEV {name} frame {int(z['index'])} at {float(z['speed']):.1f} m/s: points of the current scan with a "
              f"neighbor within {fs_cfg['d_max']:g} m: previous scan {100 * overlap(z['cur'], z['prev']):.1f}%, "
              f"scan from 10 s earlier {100 * overlap(z['cur'], z['old']):.1f}%")
        fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.9 if SCALE > 1 else 2.6))
        b = z["bev"]
        ax[0].scatter(b[:, 0], b[:, 1], s=0.05, c=b[:, 2], cmap="viridis", vmin=-2.5, vmax=2.0)
        ax[0].set_title("(a) one scan (color: height)" if SCALE > 1 else f"{name}\nframe {int(z['index'])}, {float(z['speed']):.1f} m/s", fontsize=fsz(7))
        ax[1].scatter(z["prev"][:, 0], z["prev"][:, 1], s=0.08, color="0.6", label="previous scan")
        ax[1].scatter(z["cur"][:, 0], z["cur"][:, 1], s=0.08, color="black", label="current scan")
        ax[1].set_title("(b) previous + current scan" if SCALE > 1 else "fresh: previous + current scan", fontsize=fsz(7))
        ax[2].scatter(z["old"][:, 0], z["old"][:, 1], s=0.08, color="tab:red", label="10 s old scan")
        ax[2].scatter(z["cur"][:, 0], z["cur"][:, 1], s=0.08, color="black", label="current scan")
        ax[2].set_title("(c) 10 s old + current scan" if SCALE > 1 else "replay seam: 10 s old + current scan", fontsize=fsz(7))
        for a in ax:
            a.set_aspect("equal"); a.set_xlim(-40, 40); a.set_ylim(-40, 40); a.tick_params(labelsize=fsz(6))
            if SCALE > 1:
                a.set_xlabel("x [m]")
        if SCALE > 1:
            ax[0].set_ylabel("y [m]")
        save(fig, figd, "fig_ds_bev")
    else:
        print("  (no dataset_samples/*.npz yet: the bird's-eye-view figure needs --save-samples during step 1)")


if __name__ == "__main__":
    main()
