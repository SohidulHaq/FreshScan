"""LiDAR ego-motion by point-to-point ICP (component C2).

The transform maps the *current* frame P_t onto the *previous* frame P_{t-1}.
Because static points move opposite to the vehicle, this transform equals the
vehicle motion between the two frames, expressed in the frame of P_{t-1}
(KITTI Velodyne axes: x forward, y left, z up).

Backends (same algorithm and result, chosen with select_backend()):
    cuda    Open3D tensor ICP on the GPU (Open3D wheels on PyPI are built with CUDA)
    open3d  Open3D ICP on the CPU
    numpy   small NumPy/SciPy ICP, runs anywhere (slowest)
"auto" picks cuda if a GPU is visible to Open3D, then open3d, then numpy.

Principle P2: the initial guess comes only from the previous LiDAR estimate,
never from the IMU/GPS reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # optional fast backend
    import open3d as o3d  # type: ignore
    HAVE_O3D = True
except Exception:  # pragma: no cover - depends on the environment
    HAVE_O3D = False

from scipy.spatial import cKDTree


@dataclass
class IcpResult:
    T: np.ndarray        # 4x4 vehicle motion (P_t -> P_{t-1})
    fitness: float       # share of points in P_t with a match closer than d_max
    rmse: float          # inlier RMSE [m]

    @property
    def forward(self) -> float:
        return float(self.T[0, 3])

    @property
    def yaw(self) -> float:
        return float(np.arctan2(self.T[1, 0], self.T[0, 0]))


def prepare(points: np.ndarray, r_min: float = 3.0, r_max: float = 60.0,
            voxel: float = 0.5) -> np.ndarray:
    """Crop to [r_min, r_max] (horizontal range) and voxel down-sample."""
    r = np.linalg.norm(points[:, :2], axis=1)
    pts = points[(r >= r_min) & (r <= r_max)]
    if voxel <= 0 or len(pts) == 0:
        return pts
    # One integer key per voxel; np.unique on a 1-D key is several times faster than on rows.
    k = np.floor(pts / voxel).astype(np.int64)
    k -= k.min(axis=0)
    dims = k.max(axis=0) + 1
    key = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    _, idx = np.unique(key, return_index=True)
    return pts[np.sort(idx)]


def _best_fit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares rigid transform mapping a onto b (Kabsch)."""
    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cb - R @ ca
    return T


def _icp_numpy(src: np.ndarray, dst: np.ndarray, init: np.ndarray,
               d_max: float, iters: int, tol: float) -> IcpResult:
    tree = cKDTree(dst)
    T = init.copy()
    prev_err = np.inf
    for _ in range(iters):
        moved = src @ T[:3, :3].T + T[:3, 3]
        dist, j = tree.query(moved, distance_upper_bound=d_max)
        ok = np.isfinite(dist)
        if ok.sum() < 10:
            break
        T = _best_fit(src[ok], dst[j[ok]])
        err = float(np.sqrt(np.mean(dist[ok] ** 2)))
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    moved = src @ T[:3, :3].T + T[:3, 3]
    dist, _ = tree.query(moved, distance_upper_bound=d_max)
    ok = np.isfinite(dist)
    fitness = float(ok.mean()) if len(ok) else 0.0
    rmse = float(np.sqrt(np.mean(dist[ok] ** 2))) if ok.any() else float("inf")
    return IcpResult(T=T, fitness=fitness, rmse=rmse)


BACKEND = "open3d" if HAVE_O3D else "numpy"
_CUDA = None


def cuda_available() -> bool:
    try:
        return bool(HAVE_O3D and o3d.core.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


def select_backend(pref: str = "auto") -> str:
    """Choose the ICP backend ('auto', 'cuda', 'open3d' or 'numpy') and return the one in use."""
    global BACKEND, _CUDA
    pref = (pref or "auto").lower()
    if pref in ("auto", "cuda") and cuda_available():
        BACKEND, _CUDA = "cuda", o3d.core.Device("CUDA:0")
    elif pref == "numpy" or not HAVE_O3D:
        BACKEND = "numpy"
    else:
        if pref == "cuda":
            print("[FreshScan] CUDA requested but not available to Open3D; using the CPU.")
        BACKEND = "open3d"
    return BACKEND


def _icp_cuda(cur, prev, init, d_max, iters) -> IcpResult:
    reg_t = o3d.t.pipelines.registration
    s = o3d.t.geometry.PointCloud(o3d.core.Tensor(np.ascontiguousarray(cur, dtype=np.float32), device=_CUDA))
    d = o3d.t.geometry.PointCloud(o3d.core.Tensor(np.ascontiguousarray(prev, dtype=np.float32), device=_CUDA))
    res = reg_t.icp(s, d, d_max, o3d.core.Tensor(np.asarray(init, dtype=np.float64)),
                    reg_t.TransformationEstimationPointToPoint(),
                    reg_t.ICPConvergenceCriteria(max_iteration=iters))
    return IcpResult(T=res.transformation.cpu().numpy().astype(np.float64),
                     fitness=float(res.fitness), rmse=float(res.inlier_rmse))


def icp(cur: np.ndarray, prev: np.ndarray, init: np.ndarray | None = None,
        d_max: float = 1.0, iters: int = 50, tol: float = 1e-5) -> IcpResult:
    """Register the current frame `cur` onto the previous frame `prev`."""
    init = np.eye(4) if init is None else init
    if BACKEND == "cuda":
        return _icp_cuda(cur, prev, init, d_max, iters)
    if BACKEND == "open3d":
        s = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cur))
        d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(prev))
        reg = o3d.pipelines.registration.registration_icp(
            s, d, d_max, init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters),
        )
        return IcpResult(T=np.asarray(reg.transformation), fitness=float(reg.fitness),
                         rmse=float(reg.inlier_rmse))
    return _icp_numpy(cur, prev, init, d_max, iters, tol)
