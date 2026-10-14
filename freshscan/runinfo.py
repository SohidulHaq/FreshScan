"""Records the hardware and software used for each drive (needed for the run-time numbers, RQ4)."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else "none"
    except Exception:
        return "none"


def collect(backend: str) -> dict:
    info = {"icp_backend": backend, "cpu": _cpu_model(), "cpu_count": os.cpu_count(),
            "gpu": _gpu_name(), "python": platform.python_version(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import open3d
        info["open3d"] = open3d.__version__
    except Exception:
        info["open3d"] = "not installed"
    import numpy
    info["numpy"] = numpy.__version__
    return info


def write_run_info(cfg: dict, drive: str, backend: str) -> None:
    d = os.path.join(cfg["results_dir"], "run_info")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{drive}.json")
    if not os.path.exists(path):
        with open(path, "w") as fh:
            json.dump(collect(backend), fh, indent=2)
