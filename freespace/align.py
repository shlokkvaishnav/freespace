"""Put scan A into scan B's world frame.

Two ARKit sessions share gravity (+Y up) but not heading or origin, so the
unknown transform has 4 DoF: yaw about the up axis + translation.

1. Image retrieval: CLIP image embeddings pick the A/B frame pairs that look alike.
2. SIFT keypoint matches (LightGlue optional) lifted to 3D with LiDAR depth.
3. 2-point RANSAC over yaw + translation. Changed objects are outliers here.
4. Robust point-to-plane ICP (Tukey loss) on the fused surfaces to refine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from . import features
from .recon import Surface
from .scan import Scan, backproject, transform_points


@dataclass
class Alignment:
    T_ba: np.ndarray          # maps A-world points into B-world
    n_corr: int
    n_inliers: int
    icp_fitness: float
    icp_rmse: float
    method: str

    def summary(self) -> str:
        yaw = np.degrees(np.arctan2(self.T_ba[0, 2], self.T_ba[0, 0]))
        return (f"{self.method}: {self.n_inliers}/{self.n_corr} inliers, yaw {yaw:.1f} deg, "
                f"t {np.round(self.T_ba[:3, 3], 3).tolist()}, ICP fitness {self.icp_fitness:.2f} "
                f"rmse {self.icp_rmse * 100:.1f} cm")


def yaw_matrix(theta: float, up: np.ndarray) -> np.ndarray:
    """Rotation by theta about unit axis `up` (Rodrigues)."""
    k = up / np.linalg.norm(up)
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * Kx @ Kx


def fit_yaw(pa: np.ndarray, pb: np.ndarray, up: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Least-squares yaw + translation with pb ~ R pa + t (closed form)."""
    w = np.ones(len(pa)) if w is None else w
    ca, cb = np.average(pa, 0, w), np.average(pb, 0, w)
    a, b = pa - ca, pb - cb
    a -= np.outer(a @ up, up)
    b -= np.outer(b @ up, up)
    s = np.sum(w * np.einsum("ij,ij->i", np.cross(a, b), np.broadcast_to(up, a.shape)))
    c = np.sum(w * np.einsum("ij,ij->i", a, b))
    R = yaw_matrix(np.arctan2(s, c), up)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cb - R @ ca
    return T


def ransac_yaw(pa: np.ndarray, pb: np.ndarray, up: np.ndarray, iters: int, thresh: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    best, best_in = None, np.zeros(len(pa), bool)
    if len(pa) < 2:
        return np.eye(4), best_in
    for _ in range(iters):
        i, j = rng.choice(len(pa), 2, replace=False)
        da, db = pa[j] - pa[i], pb[j] - pb[i]
        if np.linalg.norm(da - (da @ up) * up) < 0.1:
            continue
        T = fit_yaw(pa[[i, j]], pb[[i, j]], up)
        inl = np.linalg.norm(transform_points(T, pa) - pb, axis=1) < thresh
        if inl.sum() > best_in.sum():
            best, best_in = T, inl
    if best is None:
        return np.eye(4), best_in
    for _ in range(3):  # refit on inliers, recompute inliers
        best = fit_yaw(pa[best_in], pb[best_in], up)
        best_in = np.linalg.norm(transform_points(best, pa) - pb, axis=1) < thresh
    return best, best_in


def retrieval_pairs(A: Scan, B: Scan, k: int, stride_target: int = 80) -> list[tuple[int, int]]:
    sa = max(1, len(A.frames) // stride_target)
    sb = max(1, len(B.frames) // stride_target)
    ia, ib = list(range(0, len(A.frames), sa)), list(range(0, len(B.frames), sb))
    ea = features.embed([A.frames[i].rgb for i in ia])
    eb = features.embed([B.frames[i].rgb for i in ib])
    S = ea @ eb.T
    pairs, used_a = [], set()
    for flat in np.argsort(-S, axis=None):
        a, b = np.unravel_index(flat, S.shape)
        if a in used_a:  # spread pairs over A's trajectory
            continue
        used_a.add(a)
        pairs.append((ia[a], ib[b]))
        if len(pairs) == k:
            break
    return pairs


def lift(frame, uv_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RGB pixel coords -> world points via the depth map. Returns (points, valid)."""
    h, w = frame.depth.shape
    H, W = frame.rgb.shape[:2]
    uv = uv_rgb * np.array([w / W, h / H])
    p = backproject(frame.depth, frame.K_depth, uv)
    return transform_points(frame.T_wc, p), p[:, 2] > 0.05


def image_correspondences(A: Scan, B: Scan, cfg) -> tuple[np.ndarray, np.ndarray, str]:
    matcher = features.make_matcher(cfg.align.matcher)
    PA, PB = [], []
    for ia, ib in retrieval_pairs(A, B, cfg.align.retrieval_pairs):
        fa, fb = A.frames[ia], B.frames[ib]
        u0, u1 = matcher(fa.rgb, fb.rgb)
        if len(u0) < 8:
            continue
        pa, va = lift(fa, u0)
        pb, vb = lift(fb, u1)
        ok = va & vb
        PA.append(pa[ok])
        PB.append(pb[ok])
    if not PA:
        return np.zeros((0, 3)), np.zeros((0, 3)), matcher.name
    return np.concatenate(PA), np.concatenate(PB), matcher.name


def fpfh_init(sa: Surface, sb: Surface, up: np.ndarray, voxel: float = 0.05) -> np.ndarray:
    """Geometry-only fallback: FPFH correspondences + the same yaw RANSAC."""
    def prep(s):
        p = s.pcd().voxel_down_sample(voxel)
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
        f = o3d.pipelines.registration.compute_fpfh_feature(
            p, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
        return np.asarray(p.points), f.data.T

    pa, fa = prep(sa)
    pb, fb = prep(sb)
    from scipy.spatial import cKDTree

    _, j = cKDTree(fb).query(fa)
    _, i = cKDTree(fa).query(fb)
    mutual = i[j] == np.arange(len(fa))
    T, _ = ransac_yaw(pa[mutual], pb[j[mutual]], up, 20000, voxel * 1.5)
    return T


def refine_icp(sa: Surface, sb: Surface, T0: np.ndarray, dist: float):
    loss = o3d.pipelines.registration.TukeyLoss(k=dist)
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    res = T0
    for d in (dist * 2, dist, dist / 2):
        r = o3d.pipelines.registration.registration_icp(sa.pcd(), sb.pcd(), d, res, est)
        res = r.transformation
    return np.asarray(res), r.fitness, r.inlier_rmse


def align(A: Scan, B: Scan, sa: Surface, sb: Surface, cfg) -> Alignment:
    ac = cfg.align
    up = B.up / np.linalg.norm(B.up)
    pa, pb, mname = image_correspondences(A, B, cfg)
    T, inl = ransac_yaw(pa, pb, up, ac.ransac_iters, ac.ransac_thresh)
    method = f"{mname}+ransac4dof"
    if inl.sum() < 30:
        T = fpfh_init(sa, sb, up)
        method = "fpfh+ransac4dof"
    T, fit, rmse = refine_icp(sa, sb, T, ac.icp_dist)
    return Alignment(T, len(pa), int(inl.sum()), fit, rmse, method + "+icp")
