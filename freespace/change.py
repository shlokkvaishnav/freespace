"""Visibility-aware change detection.

For every surface point of scan X we ask each depth frame of scan Y one question:
"Did Y's sensor ray reach this point, stop at it, or stop before it?"

    residual r = depth_Y(pixel) - z(point)
    r >  tol  -> Y saw *through* the point's position: free-space evidence (point is gone)
    |r| <= tol -> Y saw a surface at the point: hit evidence (point is still there)
    r < -tol  -> something in front: occluded, no evidence
    outside frustum / no depth / grazing / depth edge -> no evidence

Points with too few observations stay UNKNOWN. A naive cloud-to-cloud distance
cannot make this distinction and marks every unseen region as changed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial import cKDTree

from .scan import Scan

UNCHANGED, CHANGED, UNKNOWN = 0, 1, 2
STATE_NAMES = {UNCHANGED: "unchanged", CHANGED: "changed", UNKNOWN: "unknown"}


@dataclass
class Evidence:
    n_hit: np.ndarray   # (N,) frames that confirm the surface
    n_free: np.ndarray  # (N,) frames that see through it
    state: np.ndarray   # (N,) UNCHANGED / CHANGED / UNKNOWN

    @property
    def n_obs(self) -> np.ndarray:
        return self.n_hit + self.n_free

    @property
    def p_changed(self) -> np.ndarray:
        """Posterior-like score with a weak Beta(1,1) prior."""
        return (self.n_free + 1) / (self.n_obs + 2)


def depth_edges(depth: np.ndarray, thresh: float) -> np.ndarray:
    """True where the depth map is unreliable: invalid pixels and discontinuities."""
    d = depth.copy()
    invalid = d <= 0
    d[invalid] = np.nan
    hi = maximum_filter(np.nan_to_num(d, nan=0.0), size=3)
    lo = minimum_filter(np.nan_to_num(d, nan=1e3), size=3)
    return invalid | (hi - lo > thresh)


def accumulate(points: np.ndarray, normals: np.ndarray, other: Scan, cfg) -> tuple[np.ndarray, np.ndarray]:
    c = cfg.change
    cos_max = np.cos(np.radians(c.max_incidence))
    n_hit = np.zeros(len(points), np.int32)
    n_free = np.zeros(len(points), np.int32)

    for f in other.frames:
        bad = depth_edges(f.depth, c.edge_thresh)
        h, w = f.depth.shape
        T = f.T_cw
        pc = points @ T[:3, :3].T + T[:3, 3]
        z = pc[:, 2]
        front = z > 0.1
        K = f.K_depth
        u = np.round(K[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]).astype(np.int64)
        v = np.round(K[1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]).astype(np.int64)
        inb = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = np.flatnonzero(inb)
        if idx.size == 0:
            continue
        ui, vi = u[idx], v[idx]
        ok = ~bad[vi, ui]

        # incidence: angle between surface normal and the ray to the camera
        cam = f.T_wc[:3, 3]
        ray = cam - points[idx]
        cosang = np.einsum("ij,ij->i", normals[idx], ray) / (np.linalg.norm(ray, axis=1) + 1e-9)
        ok &= np.abs(cosang) >= cos_max

        idx, ui, vi = idx[ok], ui[ok], vi[ok]
        r = f.depth[vi, ui] - z[idx]
        tol = c.depth_tol_base + c.depth_tol_rel * z[idx]
        n_free[idx[r > tol]] += 1
        n_hit[idx[np.abs(r) <= tol]] += 1
    return n_hit, n_free


def classify(n_hit: np.ndarray, n_free: np.ndarray, cfg) -> np.ndarray:
    c = cfg.change
    n_obs = n_hit + n_free
    state = np.full(len(n_hit), UNCHANGED, np.int8)
    state[n_obs < c.min_obs] = UNKNOWN
    changed = (n_obs >= c.min_obs) & (n_free >= c.free_ratio * n_obs)
    state[changed] = CHANGED
    return state


def detect(points: np.ndarray, normals: np.ndarray, other: Scan, cfg) -> Evidence:
    n_hit, n_free = accumulate(points, normals, other, cfg)
    state = classify(n_hit, n_free, cfg)
    state = smooth_states(points, state, radius=cfg.recon.points_voxel * 2.0)
    return Evidence(n_hit, n_free, state)


def smooth_states(points: np.ndarray, state: np.ndarray, radius: float) -> np.ndarray:
    """Majority vote over a small neighbourhood among *observed* points.

    Removes salt-and-pepper CHANGED points on static surfaces (pose jitter, LiDAR noise)
    and fills small holes inside a changed object. UNKNOWN points stay UNKNOWN.
    """
    obs = state != UNKNOWN
    if obs.sum() == 0:
        return state
    tree = cKDTree(points[obs])
    neigh = tree.query_ball_point(points[obs], r=radius)
    ch = (state[obs] == CHANGED).astype(np.float32)
    vote = np.array([ch[n].mean() for n in neigh])
    out = state.copy()
    out[np.flatnonzero(obs)] = np.where(vote > 0.5, CHANGED, UNCHANGED)
    return out


def naive_diff(points: np.ndarray, other_points: np.ndarray, thresh: float = 0.05) -> np.ndarray:
    """Baseline for the ablation: a point is 'changed' if no point of the other cloud is near it."""
    d, _ = cKDTree(other_points).query(points)
    return np.where(d > thresh, CHANGED, UNCHANGED).astype(np.int8)
