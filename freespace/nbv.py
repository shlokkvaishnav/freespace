"""Where to look next: camera spots that would turn UNKNOWN into a real answer.

The 'before' surface that the second walk never saw is UNKNOWN: Freespace will not call it
removed. A robot can fix that by looking. This module proposes up to 3 spots:

1. Unknown points at robot-relevant heights (0 - 1.8 m), thinned to a 10 cm grid, grouped in regions.
2. Candidate spots: free floor cells of the 'after' map (away from obstacles), camera at 1.4 m,
   looking at the centre of one unknown region.
3. For each candidate: count the unknown points inside its view (60 deg, 0.5 - 4 m) that a ray cast
   against both fused meshes actually reaches (walls and furniture block the view).
4. Greedy choice: the spot that sees the most points not yet covered, three times.
   Points on the room shell (large planes: floor, walls, ceiling) count less than points on objects:
   "is the lamp still there?" matters more to a robot than "is the wall still there?".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation
from sklearn.cluster import DBSCAN

from . import costmap
from .change import UNKNOWN


@dataclass
class Viewpoint:
    position: np.ndarray   # camera centre (shared frame)
    target: np.ndarray     # point it looks at
    covers: int            # unknown points it would newly see
    fraction: float        # ... as a share of all relevant unknown points
    seen: np.ndarray | None = None  # those unknown points (for tests and plots; not saved)


def _scene(meshes) -> o3d.t.geometry.RaycastingScene:
    sc = o3d.t.geometry.RaycastingScene()
    for m in meshes:
        if m is not None and len(m.triangles):
            sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    return sc


def _visible(scene, cam: np.ndarray, target: np.ndarray, pts: np.ndarray, hfov: float, rng: tuple) -> np.ndarray:
    d = pts - cam
    dist = np.linalg.norm(d, axis=1)
    fwd = (target - cam) / (np.linalg.norm(target - cam) + 1e-9)
    cos = (d @ fwd) / (dist + 1e-9)
    ok = (dist > rng[0]) & (dist < rng[1]) & (cos > np.cos(np.radians(hfov / 2)))
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return ok
    rays = np.concatenate([np.broadcast_to(cam, (idx.size, 3)), d[idx] / dist[idx, None]], 1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    ok[idx] = hit >= dist[idx] - 0.05  # nothing in front of the point
    return ok


def room_shell(points: np.ndarray, n_planes: int = 6, min_share: float = 0.04, dist: float = 0.03) -> np.ndarray:
    """True for points on the large planes of the room (floor, walls, ceiling), found by RANSAC plane fits."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    shell = np.zeros(len(points), bool)
    rest = np.arange(len(points))
    for _ in range(n_planes):
        if len(rest) < 100:
            break
        sub = pcd.select_by_index(rest.tolist())
        _, inl = sub.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=500)
        if len(inl) < min_share * len(points):
            break
        shell[rest[inl]] = True
        rest = np.delete(rest, inl)
    return shell


def suggest(res, cfg, n_views: int = 3) -> list[Viewpoint]:
    nc = cfg.nbv
    up = res.scan_b.up / np.linalg.norm(res.scan_b.up)
    sa = res.surf_a
    h = sa.points @ up - res.floor_h
    unknown = (res.ev_a.state == UNKNOWN) & (h > 0.02) & (h < nc.max_height)
    if unknown.sum() < nc.min_points:
        return []
    shell = room_shell(sa.points)
    pts, on_shell = sa.points[unknown], shell[unknown]
    _, keep = np.unique(np.floor(pts / nc.voxel).astype(int), axis=0, return_index=True)
    pts, on_shell = pts[keep], on_shell[keep]
    weight = np.where(on_shell, nc.shell_weight, 1.0)

    # objects and room shell are grouped separately, so an object in a corner gets its own view target
    regions = []
    for part in (pts[~on_shell], pts[on_shell]):
        if len(part) < 5:
            continue
        lab = DBSCAN(eps=nc.voxel * 2.5, min_samples=3).fit_predict(part)
        regions += [part[lab == l].mean(0) for l in set(lab) - {-1} if (lab == l).sum() >= 5]
    if not regions:
        return []

    g = res.grid
    free = (g.after == costmap.FREE) & ~binary_dilation(g.after == costmap.OCC, iterations=int(nc.clearance / g.res))
    ys, xs = np.nonzero(free)
    step = max(1, int(nc.spacing / g.res))
    sel = (ys % step == 0) & (xs % step == 0)
    flat = np.stack([xs[sel], ys[sel]], 1) * g.res + g.origin + g.res / 2
    cams = flat @ g.basis + (res.floor_h + nc.camera_height) * up

    scene = _scene([res.surf_a.mesh, res.surf_b.mesh])
    vis = []  # (cam, target, mask)
    for cam in cams:
        for tgt in regions:
            m = _visible(scene, cam, tgt, pts, nc.hfov, (0.5, nc.max_range))
            if m.sum() >= 5:
                vis.append((cam, tgt, m))
    covered = np.zeros(len(pts), bool)
    out = []
    for _ in range(n_views):
        if not vis:
            break
        gains = [weight[m & ~covered].sum() for _, _, m in vis]
        k = int(np.argmax(gains))
        if gains[k] < nc.min_points * nc.shell_weight:
            break
        cam, tgt, m = vis.pop(k)
        new = m & ~covered
        out.append(Viewpoint(cam, tgt, int(new.sum()), new.sum() / len(pts), pts[new]))
        covered |= m
    return out
