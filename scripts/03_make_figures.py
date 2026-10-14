"""Step 3: result and dataset figures (colour, vector PDF, 12.2 cm wide).

    results/figures/fig_roc.pdf            ROC curves for replay with R = 10 s (RQ1)
    results/figures/fig_features.pdf       each check on normal vs replayed frames (RQ1)
    results/figures/fig_replay_drive.pdf   one whole test drive under replay: every window and alarm (RQ2)
    results/figures/fig_timeseries.pdf     zoom on one replay window: speeds, ICP fitness, trust (RQ3)
    results/figures/fig_motion.pdf         LiDAR (ICP) motion vs IMU/GPS motion on normal driving (premise)
    results/figures/fig_dataset.pdf        speed / yaw-rate distributions and driven paths of all drives (setup)
    results/figures/fig_bev.pdf            bird's-eye view: fresh scan pair vs a replay seam (framework)
    results/tables/dataset_drives.csv      one row per drive (setup table)
    results/tables/summary_stats.json        other numbers (correlations, errors, seam overlap)

Colours: categorical slots in a fixed order (validated for colour-vision deficiency), and every series
also has its own line style or marker so the figures stay readable in print.

Usage (from the repository root):  python scripts/03_make_figures.py [--drive DRIVE]
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from common import load_config, out_dir  # noqa: E402

# --- style ---
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # categorical slots 1-4
CRIT = "#d03b3b"                                                          # status: alarm
INK, INK2, MUTED, SHADE = "#0b0b0b", "#52514e", "#8a8985", "#ececea"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
TEXTW = 4.8                                                               # figure width [in] (12.2 cm)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "legend.fontsize": 6.8, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": "#e3e2de",
    "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.2, "pdf.fonttype": 42,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})
METHODS = [  # entity -> (colour, line style, width); fixed for every figure
    ("FreshScan", BLUE, "-", 1.6),
    ("B3: Residuals only", ORANGE, "-.", 1.2),
    ("B2: Seam only", AQUA, "--", 1.2),
    ("B1: Timestamp check", YELLOW, ":", 1.4),
]


def save(fig, fig_dir, name):
    fig.savefig(os.path.join(fig_dir, f"{name}.pdf"))
    fig.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=200)
    plt.close(fig)
    print("  wrote", name)


def shade_windows(ax, t, lab):
    starts = np.where(np.diff(np.r_[0, lab]) == 1)[0]
    ends = np.where(np.diff(np.r_[lab, 0]) == -1)[0]
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[e], color=SHADE, lw=0, zorder=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--drive", default=None, help="test drive for the replay figures")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fs = cfg["freshscan"]
    res = cfg["results_dir"]
    score_dir = os.path.join(res, "tables", "scores")
    fig_dir, tab_dir = out_dir(cfg, "figures"), out_dir(cfg, "tables")
    age = cfg["rq3"]["replay_age_s"]
    files = sorted(glob.glob(os.path.join(score_dir, "*.csv")))
    if not files:
        raise FileNotFoundError("Run 02_evaluate.py first.")
    dfs = {os.path.basename(f)[:-4]: pd.read_csv(f) for f in files}
    ck_path = os.path.join(res, "checkpoint", "freshscan_checkpoint.json")
    ck = json.load(open(ck_path)) if os.path.exists(ck_path) else {"folds": {}}
    folds = ck.get("folds", {})
    period_all = folds.get("all", {}).get("lidar_period_s", fs["period_s"])
    stats = {}

    def period_of(df):
        return folds.get(str(df.fold.iloc[0]), {}).get("lidar_period_s", period_all) if "fold" in df else period_all

    # --- Fig: ROC (RQ1) ---
    ben = pd.concat([d for k, d in dfs.items() if k.endswith("__benign")])
    att = pd.concat([d[d.label == 1] for k, d in dfs.items() if f"__replay_R{age}_" in k])
    fig, ax = plt.subplots(figsize=(2.35, 2.05))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=(0, (2, 2)))
    for m, c, ls, lw in METHODS[::-1]:
        col = f"score::{m}"
        fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(ben)), np.ones(len(att))], np.r_[ben[col], att[col]])
        ax.plot(fpr, tpr, color=c, ls=ls, lw=lw, label=m.replace("FreshScan", "FreshScan (ours)"))
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="lower right", handlelength=2.4, frameon=True, framealpha=0.95, edgecolor="#c9c8c3", facecolor="white")
    fig.tight_layout(pad=0.3)
    save(fig, fig_dir, "fig_roc")

    # --- Fig: feature distributions, normal vs replay (RQ1) ---
    att_in = att[att.seam == 0]
    att_seam = att[att.seam == 1]
    panels = [("r_v", "Speed residual $r^v_t$"), ("r_w", "Yaw-rate residual $r^\\omega_t$"),
              ("one_minus_fit", "ICP misfit $1-f_t$"), ("score::FreshScan", "FreshScan score $a_t$")]
    fig, axs = plt.subplots(1, 4, figsize=(TEXTW, 1.75))
    for ax, (c, lab) in zip(axs, panels):
        logx = c.startswith("score")
        allv = np.r_[ben[c], att[c]]
        if logx:
            b = np.logspace(np.log10(max(np.percentile(allv, 0.5), 1e-3)), np.log10(np.percentile(allv, 99.9)), 45)
        else:
            b = np.linspace(0, np.percentile(allv, 99.5), 45)
        clip = lambda v: np.clip(v, b[0], b[-1])  # noqa: E731
        ax.hist(clip(ben[c]), bins=b, density=True, color=BLUE, alpha=0.35, lw=0, label="normal")
        ax.hist(clip(ben[c]), bins=b, density=True, histtype="step", color=BLUE, lw=0.9)
        ax.hist(clip(att_in[c]), bins=b, density=True, histtype="step", color=ORANGE, lw=1.1, label="replayed, inside")
        ax.hist(clip(att_seam[c]), bins=b, density=True, histtype="step", color=INK, lw=0.8, ls=(0, (2, 1.2)),
                label="replayed, seam")
        ax.set_yscale("log"); ax.set_xlabel(lab); ax.tick_params(axis="y", labelsize=6)
        if logx:
            ax.set_xscale("log")
            thr = float(np.median(pd.concat(dfs.values())["thr::FreshScan"]))
            ax.axvline(thr, color=CRIT, lw=0.9, ls="--")
    axs[0].set_ylabel("Density (log)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, ["normal frames", "replayed frames inside a window", "replayed frames at a seam"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=2.0)
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.88))
    save(fig, fig_dir, "fig_features")

    # --- Fig: whole drive under replay (RQ2) ---
    keys = sorted(k for k in dfs if f"__replay_R{age}_s0" in k)
    if args.drive:
        keys = [k for k in keys if k.startswith(args.drive)] or keys
    df = dfs[keys[0]]
    p = period_of(df)
    t = df.slot.values * p
    lab = df.label.values
    thr = float(df["thr::FreshScan"].iloc[0])
    fig, axs = plt.subplots(3, 1, figsize=(TEXTW, 3.55), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1.1, 1.25]})
    for ax in axs:
        shade_windows(ax, t, lab)
    axs[0].plot(t, df.slot, color=MUTED, lw=0.9, ls=(0, (3, 2)), label="expected frame (fresh)")
    axs[0].plot(t, df.sigma, color=BLUE, lw=1.0, label="delivered frame")
    axs[0].set_ylabel("Frame index")
    axs[1].plot(t, df.v_bar, color=INK, lw=0.9, label="IMU/GPS speed now")
    axs[1].plot(t, df.v_bar_src, color=ORANGE, lw=0.9, ls="--", label="speed when the delivered frame was captured")
    axs[1].set_ylabel("Speed (m/s)")
    axs[2].plot(t, df["score::FreshScan"], color=BLUE, lw=0.8, label="FreshScan score")
    axs[2].axhline(thr, color=INK2, lw=0.8, ls="--", label="alarm threshold")
    al = df.alarm.values == 1
    sa = df.seam_alarm.values == 1
    ymax = np.nanmax(df["score::FreshScan"]) * 3
    axs[2].plot(t[al], np.full(al.sum(), ymax * 0.55), "|", color=CRIT, ms=5, mew=1.0, label="alarm")
    axs[2].plot(t[sa], np.full(sa.sum(), ymax), "v", color=CRIT, ms=3.2, mfc="white", mew=0.8, label="seam alarm")
    axs[2].set_yscale("log"); axs[2].set_ylabel("Score (log)"); axs[2].set_xlabel("Time (s)")

    axs[2].set_ylim(np.nanmin(df["score::FreshScan"]) / 2.5, ymax * 2.2)
    axs[2].set_xlim(t[0], t[-1])
    for ax in axs:
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=4, borderaxespad=0.1, columnspacing=1.2)
    fig.tight_layout(pad=0.3, h_pad=1.5)
    save(fig, fig_dir, "fig_replay_drive")
    nw = int((np.diff(np.r_[0, lab]) == 1).sum())
    stats["replay_drive"] = {"stream": keys[0], "minutes": round(len(df) * p / 60, 2), "windows": nw}

    # --- Fig: zoom on one replay window (RQ3) ---
    starts = np.where(np.diff(np.r_[0, lab]) == 1)[0]
    ends = np.where(np.diff(np.r_[lab, 0]) == -1)[0]
    s0, e0 = starts[0], ends[0]
    lo, hi = max(0, s0 - 150), min(len(df), e0 + 150)
    tt = t[lo:hi]
    ratio = fs["period_s"] / p
    fig, axs = plt.subplots(3, 1, figsize=(TEXTW * 0.82, 2.55), sharex=True)
    for ax in axs:
        ax.axvspan(t[s0], t[e0], color=SHADE, lw=0, zorder=0)
    axs[0].plot(tt, df.v_bar.values[lo:hi], color=INK, lw=1.0, label=r"IMU/GPS speed $\bar v_t$")
    axs[0].plot(tt, df.v_hat.values[lo:hi] * ratio, color=BLUE, lw=1.0, ls="--", label=r"LiDAR speed $\hat v_t$")
    axs[0].set_ylabel("Speed (m/s)")
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=2, borderaxespad=0)
    axs[1].plot(tt, df.fit.values[lo:hi], color=AQUA, lw=1.0)
    axs[1].set_ylabel(r"ICP fitness $f_t$")
    axs[2].plot(tt, df.trust.values[lo:hi], color=BLUE, lw=1.0, label=r"trust $\theta_t$")
    axs[2].axhline(np.exp(-thr), color=INK2, lw=0.8, ls="--", label="alarm threshold")
    a2 = df.alarm.values[lo:hi] == 1
    axs[2].plot(tt[a2], np.full(a2.sum(), 0.02), "|", color=CRIT, ms=4, mew=1.0, label="alarm")
    axs[2].set_ylabel(r"Trust $\theta_t$"); axs[2].set_xlabel("Time (s)")
    axs[2].legend(loc="lower right", fontsize=6, ncol=1, borderaxespad=0.2, frameon=True, framealpha=0.95, edgecolor="#c9c8c3")
    axs[2].set_xlim(tt[0], tt[-1])
    fig.tight_layout(pad=0.3, h_pad=0.2)
    save(fig, fig_dir, "fig_timeseries")

    # --- normal (benign) streams of every drive, for the next figures ---
    feats = {}
    for f in sorted(glob.glob(os.path.join(res, "features", "*__benign.csv"))):
        name = os.path.basename(f).split("__")[0]
        d = pd.read_csv(f)
        feats[name] = d[d.slot > 0].reset_index(drop=True)
    if feats:
        role = lambda n: "test" if n in cfg["test_drives"] else "normal only"  # noqa: E731
        v_min = fs.get("v_min", 0.0)
        rows = []
        for n, d in feats.items():
            rows.append(dict(drive=n, role=role(n), frames=len(d) + 1, minutes=round((len(d) + 1) * period_all / 60, 2),
                             distance_km=round(float(np.sum(np.abs(d.v_bar)) * period_all / 1000), 2),
                             mean_speed_mps=round(float(d.v_bar.mean()), 1), max_speed_mps=round(float(d.v_bar.max()), 1),
                             below_vmin_pct=round(100 * float(np.mean(np.abs(d.v_bar) < v_min)), 1),
                             mean_abs_yaw_deg_s=round(float(np.degrees(np.abs(d.w_bar)).mean()), 1),
                             icp_fit_median=round(float(d.fit.median()), 3)))
        pd.DataFrame(rows).to_csv(os.path.join(tab_dir, "dataset_drives.csv"), index=False)

        # --- Fig: motion agreement (premise) ---
        allb = pd.concat(feats.values())
        vh, wh = allb.v_hat * fs["period_s"] / period_all, allb.w_hat * fs["period_s"] / period_all
        moving = np.abs(allb.v_bar) >= v_min
        ev, ew = (vh - allb.v_bar)[moving], np.degrees(wh - allb.w_bar)[moving]
        stats["motion"] = {
            "frames": int(len(allb)), "frames_moving": int(moving.sum()),
            "corr_speed": round(float(np.corrcoef(allb.v_bar, vh)[0, 1]), 4),
            "corr_yaw": round(float(np.corrcoef(allb.w_bar, wh)[0, 1]), 4),
            "speed_err_median_abs_mps": round(float(np.median(np.abs(ev))), 3),
            "speed_err_p95_abs_mps": round(float(np.percentile(np.abs(ev), 95)), 3),
            "yaw_err_median_abs_deg_s": round(float(np.median(np.abs(ew))), 3),
            "yaw_err_p95_abs_deg_s": round(float(np.percentile(np.abs(ew), 95)), 3)}
        fig, axs = plt.subplots(1, 2, figsize=(TEXTW * 0.92, 2.25), layout="constrained")
        for ax, x, y, lab, unit, cc in [
                (axs[0], allb.v_bar, vh, "speed", "m/s", stats["motion"]["corr_speed"]),
                (axs[1], np.degrees(allb.w_bar), np.degrees(wh), "yaw rate", "deg/s", stats["motion"]["corr_yaw"])]:
            hb = ax.hexbin(x, y, gridsize=55, bins="log", cmap=BLUES, mincnt=1, linewidths=0)
            lim = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lim, lim, color=INK2, lw=0.7, ls=(0, (3, 2)))
            ax.set_xlabel(f"IMU/GPS {lab} ({unit})"); ax.set_ylabel(f"LiDAR (ICP) {lab} ({unit})")
            ax.text(0.04, 0.95, f"correlation {cc:.3f}", transform=ax.transAxes, va="top", fontsize=6.8)
            ax.set_aspect("equal", adjustable="box")
        cb = fig.colorbar(hb, ax=axs, fraction=0.04, pad=0.01, shrink=0.9)
        cb.set_label("Frames (log)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
        cb.outline.set_linewidth(0.4)
        save(fig, fig_dir, "fig_motion")

        # --- Fig: dataset (speed / yaw distributions, driven paths) ---
        names = sorted(feats, key=lambda n: (role(n) != "test", n))
        fig = plt.figure(figsize=(TEXTW, 3.05))
        top = fig.add_gridspec(1, 2, left=0.10, right=0.985, top=0.895, bottom=0.585, wspace=0.32)
        bot = fig.add_gridspec(1, len(names), left=0.02, right=0.985, top=0.36, bottom=0.085, wspace=0.12)
        ax1, ax2 = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
        groups = [("test drives (replay injected)", [n for n in names if role(n) == "test"], ORANGE, "--"),
                  ("normal-only drives", [n for n in names if role(n) != "test"], BLUE, "-")]
        vb = np.linspace(0, 28, 57)
        wb = np.linspace(0, 40, 41)
        for lab_, ns, c, ls in groups:
            v = np.concatenate([feats[n].v_bar.values for n in ns])
            w = np.degrees(np.abs(np.concatenate([feats[n].w_bar.values for n in ns])))
            ax1.hist(np.clip(v, 0, 28), bins=vb, density=True, histtype="step", color=c, ls=ls, lw=1.2, label=lab_)
            ax2.hist(np.clip(w, 0, 40), bins=wb, density=True, histtype="step", color=c, ls=ls, lw=1.2, label=lab_)
        ax1.axvline(v_min, color=INK2, lw=0.7, ls=":")
        ym = ax1.get_ylim()[1]
        ax1.set_ylim(0, ym * 1.28)
        ax1.annotate("motion checks abstain below 2 m/s", xy=(v_min, ym * 1.17), xytext=(4.5, ym * 1.17),
                     fontsize=6, color=INK2, va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
        ax1.set_xlabel("IMU/GPS speed (m/s)"); ax1.set_ylabel("Density")
        ax2.set_xlabel("|Yaw rate| (deg/s)"); ax2.set_yscale("log"); ax2.set_ylabel("Density (log)")
        for ax, cap in [(ax1, "(a) Speed"), (ax2, "(b) Turning")]:
            ax.text(0.5, -0.43, cap, transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
        h, l = ax1.get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.0))
        fig.text(0.5, 0.045, "(c) Driven paths (integrated from IMU/GPS; dot = start)", ha="center", va="top", fontsize=7.5)
        for i, n in enumerate(names):
            ax = fig.add_subplot(bot[0, i])
            d = feats[n]
            yaw = np.cumsum(d.w_bar.values) * period_all
            x = np.cumsum(d.v_bar.values * np.cos(yaw)) * period_all
            y = np.cumsum(d.v_bar.values * np.sin(yaw)) * period_all
            c = ORANGE if role(n) == "test" else BLUE
            ax.plot(x, y, color=c, lw=0.9, ls="-")
            ax.plot(x[:1], y[:1], "o", ms=2.6, color=INK)
            ax.set_aspect("equal", "datalim")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color("#c9c8c3"); sp.set_linewidth(0.5)
            short = n.replace("2011_", "").replace("_drive_", "/")
            ax.set_title(f"{short}, {np.sum(np.abs(d.v_bar)) * period_all / 1000:.1f} km", fontsize=5.6, pad=2)
        save(fig, fig_dir, "fig_dataset")

    # --- Fig: bird's-eye view of a replay seam (framework) ---
    samples = sorted(glob.glob(os.path.join(res, "dataset_samples", "*.npz")))
    if samples:
        from scipy.spatial import cKDTree

        def overlap(cur, other, d=None):
            dist, _ = cKDTree(other).query(cur, distance_upper_bound=d or fs["d_max"])
            return float(np.isfinite(dist).mean())
        scored_s = []
        for f in samples:
            z = np.load(f)
            scored_s.append((overlap(z["cur"], z["prev"]) - overlap(z["cur"], z["old"]), f))
        pick = [f for f in samples if args.drive and os.path.basename(f).startswith(args.drive)] or [max(scored_s)[1]]
        z = np.load(pick[0]); name = os.path.basename(pick[0])[:-4]
        of, oo = overlap(z["cur"], z["prev"]), overlap(z["cur"], z["old"])
        fig, axs = plt.subplots(1, 3, figsize=(TEXTW, 1.95))
        b = z["bev"]
        o = np.argsort(b[:, 2])
        axs[0].scatter(b[o, 0], b[o, 1], s=0.05, c=b[o, 2], cmap=BLUES, vmin=-3.5, vmax=1.0, lw=0, rasterized=True)
        axs[0].set_title("(a) One scan (color: height)")
        for ax, other, lab_, ttl, ov in [(axs[1], z["prev"], "previous scan", "(b) Fresh pair", of),
                                          (axs[2], z["old"], "scan from 10 s earlier", "(c) Replay seam", oo)]:
            ax.scatter(z["cur"][:, 0], z["cur"][:, 1], s=0.15, color=BLUE, lw=0, rasterized=True, label="current scan")
            ax.scatter(other[:, 0], other[:, 1], s=0.15, color=ORANGE, lw=0, rasterized=True, label=lab_)
            ax.set_title(ttl)
            ax.text(0.03, 0.03, f"{100 * ov:.0f}% of points have\na match within {fs['d_max']:g} m", transform=ax.transAxes,
                    fontsize=6, va="bottom", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#c9c8c3", lw=0.4, alpha=0.92))
            ax.legend(loc="upper left", markerscale=10, handletextpad=0.1, borderaxespad=0.15, fontsize=5.8,
                      frameon=True, framealpha=0.92, edgecolor="#c9c8c3")
        for ax in axs:
            ax.set_aspect("equal"); ax.set_xlim(-35, 35); ax.set_ylim(-35, 35)
            ax.set_xticks([-30, 0, 30]); ax.set_yticks([-30, 0, 30]); ax.grid(False)
            ax.set_xlabel("x (m), driving direction")
        axs[0].set_ylabel("y (m)")
        fig.tight_layout(pad=0.3, w_pad=0.4)
        save(fig, fig_dir, "fig_bev")
        stats["bev"] = {"drive": name, "frame": int(z["index"]), "speed_mps": round(float(z["speed"]), 1),
                        "overlap_fresh_pct": round(100 * of, 1), "overlap_10s_pct": round(100 * oo, 1)}

    with open(os.path.join(tab_dir, "summary_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))

    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
