# FreshScan

Code for *FreshScan: Detecting Replayed LiDAR Frames in Autonomous Vehicles Through
Ego-Motion Consistency*.

FreshScan checks whether a LiDAR frame is fresh by comparing the motion seen between two
consecutive scans (point-to-point ICP) with the motion reported by the IMU/GPS. A replayed
frame shows motion from another moment, so the two stop agreeing. The monitor needs no
camera, no attack data and no trusted timestamps.

This repository has the monitor (components C1, C2, C3 and C5), the replay-injection
benchmark on KITTI raw, the calibration checkpoint from our full run, and the scripts that
produce all result tables and figures. The time-shift search (C4) and the other attack
types (delay, freeze, drop, reorder) are part of the framework but are not evaluated yet, so
they are not included here.

## Repository layout

```
freshscan/          the method
  kitti_loader.py     reads KITTI raw synced drives (Velodyne scans, OXTS motion, timestamps)
  replay.py           replay injection: frame t-R is delivered at slot t inside random windows
  registration.py     C1 crop and voxel grid, C2 point-to-point ICP (Open3D GPU/CPU or NumPy)
  features.py         C3 speed and yaw-rate residuals, seam features (ICP misfit, RMSE)
  detector.py         C5 max-ratio score, threshold, 2-of-3 rule, trust; B1 timestamp check
  checkpoint.py       saves and loads the calibration, scores new feature files
  runinfo.py          records the CPU/GPU used for each drive
scripts/
  00_download_kitti.sh       downloads the drives listed in the config
  01_compute_features.py     runs C1-C3 on normal and replayed streams
  02_evaluate.py             calibration, scoring, RQ1-RQ4 tables, checkpoint
  03_make_figures.py         result and dataset figures (vector PDF)
  04_visualize_dataset.py    extra dataset plots; saves scan samples for the bird's-eye view
  05_evaluate_checkpoint.py  re-evaluates from the checkpoint at other thresholds
configs/            default.yaml: drives, replay settings and FreshScan parameters
notebooks/          FreshScan_Colab.ipynb: the whole experiment on Google Colab
results/            tables, figures, checkpoint and hardware info of our full run
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Our full run used Python 3.13 on Colab; the code was also tested with Python 3.11. Open3D is optional; without
it ICP falls back to NumPy/SciPy, which gives the same result but is much slower.

## Running on Google Colab

The easiest way to reproduce the results is `notebooks/FreshScan_Colab.ipynb`. Upload it to
Colab, select a GPU runtime and choose *Runtime > Run all*. The notebook contains the code,
downloads one KITTI drive at a time, saves everything to Google Drive and resumes if Colab
disconnects.

## Running locally

1. Register at <https://www.cvlibs.net/datasets/kitti/raw_data.php> and accept the license.
2. Download the seven drives in `configs/default.yaml` with `bash scripts/00_download_kitti.sh`.
   If the script cannot reach the server, download the *synced+rectified data* zips by hand and
   unpack them so that you get `data/kitti_raw/2011_09_30/2011_09_30_drive_0018_sync/{velodyne_points,oxts}/`.
3. Run the pipeline:

```bash
python scripts/01_compute_features.py   # slowest step: one ICP per frame and stream
python scripts/02_evaluate.py
python scripts/03_make_figures.py
```

`01_compute_features.py` skips outputs that already exist, so it can be stopped and restarted,
and `--drives` lets you split the drives across machines.

## Settings

All settings are in `configs/default.yaml`:

| Setting | Value |
|---|---|
| Range crop / voxel / ICP match distance | 3-60 m / 0.5 m / 1 m |
| Scan period | measured on normal driving (median header step), 103.7 ms on KITTI |
| Averaging window | 3 frames |
| Speed and yaw-rate offsets | 3 m/s, 0.05 rad/s |
| Motion checks skipped below | 2 m/s |
| Alarm | score above the 1% threshold in 2 of the last 3 frames, or seam score above the 0.1% threshold |
| Trust | exp(-score) |
| Replay ages / window length / gap | 5, 10, 30 s / 2-10 s / at least 10 s |
| Seeds | 0, 1, 2 |
| Calibration | leave-one-drive-out over 7 drives, normal driving only |

## Checkpoint and later evaluation

FreshScan has no trained model. Its "checkpoint" is the calibration: the measured scan period,
the 99th percentile of each check on normal driving, and the alarm thresholds for each
leave-one-drive-out fold. Ours is in `results/checkpoint/`.

To try other operating points without running ICP again:

```bash
python scripts/05_evaluate_checkpoint.py --fpr 0.01 0.002 0.001 --m 2 --n 3
```

This needs the per-frame feature files from step 1 (`results/features/`, about 30 MB). They are
not in the repository; run step 1 or the Colab notebook to create them. To change the motion
checks themselves (averaging, offsets, low-speed cutoff), edit the config and rerun
`02_evaluate.py`.

## Our results

`results/tables/` has the numbers from our full run:

| File | Content |
|---|---|
| `rq1_detection.csv` | AUROC and TPR at 1% false positives per method and replay age |
| `rq2_delay.csv` | detection delay, windows detected, false alarms per minute |
| `rq3_analysis.csv` | detection split by seam/inside, speed difference and current speed |
| `rq4_runtime.csv` | time per frame for preparation, ICP and the checks |
| `checkpoint_eval.csv` | other operating points (0.5%, 0.2%, 0.1%) |
| `summary_stats.json` | LiDAR vs. IMU/GPS agreement and seam overlap |
| `dataset_drives.csv` | frames, minutes, distance and speeds of each drive |

## Definitions

- **Replay:** inside a window, slot t delivers frame t-R; the timestamp and the IMU/GPS data stay those of slot t.
- **Attacked frames:** every slot in a window plus the first fresh slot after it, whose registration still spans the last replayed frame.
- **Seam:** the first slot of a window and the first fresh slot after it.
- **False alarms per minute:** runs of consecutive alarmed frames on normal test driving.
- **Detection delay:** frames from the first replayed frame to the first alarm.
- **B1 / B2 / B3:** timestamp check / seam features only / motion residuals only.

## Data

The experiments use KITTI raw (Geiger et al., IJRR 2013), released under CC BY-NC-SA 3.0. This
repository does not contain KITTI data; `results/` only has statistics derived from it.
