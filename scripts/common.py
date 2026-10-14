"""Shared helpers for the experiment scripts."""

from __future__ import annotations

import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)


def load_config(path: str | None = None) -> dict:
    path = path or os.path.join(CODE, "configs", "default.yaml")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for key in ("kitti_root", "results_dir"):
        if not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(CODE, cfg[key])
    return cfg


def out_dir(cfg: dict, *parts: str) -> str:
    d = os.path.join(cfg["results_dir"], *parts)
    os.makedirs(d, exist_ok=True)
    return d
