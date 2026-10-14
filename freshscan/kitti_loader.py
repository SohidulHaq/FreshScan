"""Reader for KITTI raw *synced* drives.

Expected layout (as unpacked from the KITTI raw data zips):

    <root>/<date>/<date>_drive_<id>_sync/
        velodyne_points/data/0000000000.bin   float32 x, y, z, reflectance
        velodyne_points/timestamps.txt        one timestamp per frame (optional)
        oxts/data/0000000000.txt              30 values per line

OXTS fields used (0-based index in each line, see the KITTI raw devkit readme):
    8  vf : forward velocity, parallel to the earth surface [m/s]
    22 wu : angular rate around the upward axis [rad/s]
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np

OXTS_VF = 8
OXTS_WU = 22


@dataclass
class Drive:
    name: str
    velo_files: list
    speed: np.ndarray        # reference forward speed per frame [m/s]
    yaw_rate: np.ndarray     # reference yaw rate per frame [rad/s]
    stamps: np.ndarray       # LiDAR header time per frame [s], relative to frame 0

    def __len__(self) -> int:
        return len(self.velo_files)

    def frame(self, i: int) -> np.ndarray:
        """Frame i as an (N, 3) array of x, y, z points (Velodyne axes: x forward, y left, z up)."""
        pts = np.fromfile(self.velo_files[i], dtype=np.float32).reshape(-1, 4)
        return pts[:, :3].astype(np.float64)


def _read_stamps(path: str, n: int, period: float) -> np.ndarray:
    if not os.path.exists(path):
        return np.arange(n) * period
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # format: 2011-09-26 13:02:25.964389445  (nanoseconds are cut to microseconds)
            day, clock = line.split(" ")
            hms, frac = clock.split(".")
            t = datetime.strptime(f"{day} {hms}", "%Y-%m-%d %H:%M:%S").timestamp() + float("0." + frac)
            out.append(t)
    out = np.asarray(out[:n])
    return out - out[0]


def load_drive(root: str, name: str, period: float = 0.1) -> Drive:
    """`name` is e.g. '2011_09_30_drive_0018'."""
    date = name[:10]
    base = os.path.join(root, date, f"{name}_sync")
    velo = sorted(glob.glob(os.path.join(base, "velodyne_points", "data", "*.bin")))
    oxts = sorted(glob.glob(os.path.join(base, "oxts", "data", "*.txt")))
    if not velo or len(velo) != len(oxts):
        raise FileNotFoundError(f"No matching Velodyne/OXTS files under {base} "
                                f"({len(velo)} velodyne, {len(oxts)} oxts)")
    vals = np.array([np.loadtxt(f) for f in oxts])
    stamps = _read_stamps(os.path.join(base, "velodyne_points", "timestamps.txt"), len(velo), period)
    return Drive(name=name, velo_files=velo, speed=vals[:, OXTS_VF], yaw_rate=vals[:, OXTS_WU],
                 stamps=stamps)
