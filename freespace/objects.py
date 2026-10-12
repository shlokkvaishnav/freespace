"""Group changed surface points into objects, pick good views, pair removed+added into moves."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from . import features
from .align import yaw_matrix
from .change import CHANGED, UNCHANGED
from .scan import Scan


@dataclass
class View:
    frame: int               # index into scan.frames
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in rgb pixels
    visible: int             # cluster points visible in this frame


@dataclass
class Blob:
    side: str                # "before" (removed surface) or "after" (added surface)
    points: np.ndarray       # object points (grown from the changed points, see grow_object)
    colors: np.ndarray
    changed: np.ndarray | None = None  # the CHANGED surface points that started this object
    views: list[View] = field(default_factory=list)
    embedding: np.ndarray | None = None

    @property
    def centroid(self) -> np.ndarray:
        return self.points.mean(0)

    def extent(self, up: np.ndarray) -> np.ndarray:
        """(height, long side, short side) of a gravity-aligned box."""
        h = self.points @ up
        flat = self.points - np.outer(h, up)
        c = flat - flat.mean(0)
        _, s, Vt = np.linalg.svd(c, full_matrices=False)
        proj = c @ Vt[:2].T
        span = proj.max(0) - proj.min(0)
        return np.array([h.max() - h.min(), span.max(), span.min()])


@dataclass
class Change:
    id: int
    status: str              # moved | removed | added
    before: Blob | None
    after: Blob | None
    T_move: np.ndarray | None = None  # before -> after, for moves
    label: str = "object"
    attributes: str = ""
    description: str = ""
    label_score: float = 0.0
    facts: dict = field(default_factory=dict)
    flat: bool = False       # found by colour only (too thin for the depth test)
    motion: str = ""         # moves only: move | slide | turn_in_place | hinge
    yaw_alternatives: list[float] = field(default_factory=list)  # other turns that fit the shape equally well
    fit_error: float = 0.0   # median distance (m) after the best rigid fit

    @property
    def blob(self) -> Blob:
        return self.after if self.after is not None else self.before

    @property
    def distance(self) -> float:
        """How far the object centre moved, from the fitted rigid transform (not from partial centroids)."""
        if self.status != "moved":
            return 0.0
        c = self.before.centroid
        if self.T_move is None:
            return float(np.linalg.norm(self.after.centroid - c))
        return float(np.linalg.norm(self.T_move[:3, :3] @ c + self.T_move[:3, 3] - c))

    def yaw(self, up: np.ndarray) -> float:
        if self.T_move is None:
            return 0.0
        return _yaw_of(self.T_move, up)

    def confidence(self) -> float:
        return float(min(1.0, sum(v.visible for v in self.blob.views) / 400))


def cluster(points: np.ndarray, colors: np.ndarray, state: np.ndarray, side: str, cfg) -> list[Blob]:
    oc = cfg.objects
    idx = np.flatnonzero(state == CHANGED)
    if idx.size < oc.cluster_min_points:
        return []
    lab = DBSCAN(eps=oc.cluster_eps, min_samples=8).fit_predict(points[idx])
    blobs = []
    for l in set(lab) - {-1}:
        m = idx[lab == l]
        if m.size < oc.cluster_min_points:
            continue
        span = points[m].max(0) - points[m].min(0)
        if span.max() < oc.min_extent:
            continue
        blobs.append(Blob(side, points[m], colors[m]))
    return blobs


def _flat(p: np.ndarray, up: np.ndarray) -> np.ndarray:
    return p - np.outer(p @ up, up)


def support_height(points: np.ndarray, normals: np.ndarray, obj_pts: np.ndarray, up: np.ndarray, floor_h: float,
                   margin: float = 0.08) -> float:
    """Height of the highest horizontal surface under the object (floor, table top, shelf).
    The footprint is widened by `margin`: right under the object the support is often hidden by the object."""
    h_all, h_obj = points @ up, obj_pts @ up
    lo, hi = _flat(obj_pts, up).min(0) - margin, _flat(obj_pts, up).max(0) + margin
    fp = _flat(points, up)
    inside = np.all((fp >= lo) & (fp <= hi), axis=1)
    horizontal = np.abs(normals @ up) > 0.9
    # if only a flat top face is known, that face (and the rest of the top) must not count as its own support
    top_only = np.ptp(h_obj) < 0.03
    limit = h_obj.min() - 0.015 if top_only else h_obj.min() + 0.01
    below = inside & horizontal & (h_all < limit) & (h_all > floor_h - 0.05)
    return float(np.percentile(h_all[below], 95)) if below.sum() >= 10 else floor_h


def grow_object(blob: Blob, surf, up: np.ndarray, floor_h: float, cfg, state: np.ndarray | None = None) -> Blob:
    """Grow the CHANGED points into the whole object in their own scan.

    1. Cut away the support surface under the object (floor / table top), so objects on a
       shared surface do not join.
    2. Keep surface points that connect (DBSCAN) to the changed points; grow the box and repeat.
    This recovers the parts the ray test cannot call changed (the bottom, which touches the support)
    and the rest of an object that moved only a little.
    Points the other scan *confirmed* (UNCHANGED) are allowed only near the support or near the changed
    points, so growth does not run into a static neighbour (a table next to a door).
    """
    oc = cfg.objects
    seed = blob.points
    sup = support_height(surf.points, surf.normals, seed, up, floor_h)
    h_all = surf.points @ up
    above = h_all > sup + oc.support_gap
    if state is not None:
        near_seed = cKDTree(seed).query(surf.points, distance_upper_bound=oc.grow_margin)[0] < np.inf
        above &= (state != UNCHANGED) | (h_all < sup + oc.bottom_band) | near_seed
    fp = _flat(surf.points, up)
    kept = seed
    keep_idx = None
    for _ in range(oc.grow_iters):
        lo = _flat(kept, up).min(0) - oc.grow_margin
        hi = _flat(kept, up).max(0) + oc.grow_margin
        top = (kept @ up).max() + oc.grow_margin
        sel = np.flatnonzero(above & (h_all < top) & np.all((fp >= lo) & (fp <= hi), axis=1))
        if sel.size < 10:
            break
        # tighter link than the change clustering: objects a few cm apart must not join
        lab = DBSCAN(eps=oc.grow_eps, min_samples=5).fit_predict(surf.points[sel])
        d, j = cKDTree(surf.points[sel]).query(seed)
        hit = set(lab[j[d < oc.grow_eps]]) - {-1}
        new_idx = sel[np.isin(lab, list(hit))]
        if new_idx.size == 0 or (keep_idx is not None and new_idx.size <= keep_idx.size):
            break
        span = np.ptp(_flat(surf.points[new_idx], up), axis=0).max()
        if span > oc.max_object_size:  # grew into a wall or a large piece of furniture: stop
            break
        keep_idx = new_idx
        kept = surf.points[keep_idx]
    if keep_idx is None:
        return Blob(blob.side, seed, blob.colors, changed=seed)
    return Blob(blob.side, surf.points[keep_idx], surf.colors[keep_idx], changed=seed)


def merge_same_object(blobs: list[Blob], voxel: float) -> list[Blob]:
    """Clusters that grew into the same object become one."""
    keys = [set(map(tuple, np.floor(b.points / voxel).astype(int))) for b in blobs]
    out, used = [], set()
    for i, b in enumerate(blobs):
        if i in used:
            continue
        pts, cols, ch, k = [b.points], [b.colors], [b.changed], set(keys[i])
        for j in range(i + 1, len(blobs)):
            if j in used:
                continue
            if len(k & keys[j]) > 0.3 * min(len(k), len(keys[j])):
                used.add(j)
                k |= keys[j]
                pts.append(blobs[j].points); cols.append(blobs[j].colors); ch.append(blobs[j].changed)
        P, C = np.concatenate(pts), np.concatenate(cols)
        _, uniq = np.unique(np.round(P / (voxel / 2)).astype(int), axis=0, return_index=True)
        out.append(Blob(b.side, P[uniq], C[uniq], changed=np.concatenate(ch)))
    return out


def height_change(a: Blob, b: Blob, up: np.ndarray) -> float:
    """Relative change of the height along gravity. A rigid object keeps its height (unless it tips over).
    Height is used, not the horizontal sides: those depend on the outline and on what each walk saw."""
    ha, hb = a.extent(up)[0], b.extent(up)[0]
    return float(abs(ha - hb) / max(ha, hb, 0.05))


def footprint_iou(a: Blob, b: Blob, up: np.ndarray) -> float:
    """Overlap of the two horizontal bounding boxes (in a fixed plane basis)."""
    u = np.cross(up, [1.0, 0, 0]); u = u if np.linalg.norm(u) > 0.1 else np.cross(up, [0, 0, 1.0]); u /= np.linalg.norm(u)
    B = np.stack([u, np.cross(up, u)], 1)
    pa, pb = a.points @ B, b.points @ B
    lo, hi = np.maximum(pa.min(0), pb.min(0)), np.minimum(pa.max(0), pb.max(0))
    inter = np.prod(np.clip(hi - lo, 0, None))
    area = lambda p: np.prod(p.max(0) - p.min(0))
    return float(inter / max(area(pa) + area(pb) - inter, 1e-6))


def find_views(blob: Blob, scan: Scan, cfg, n: int = 3) -> list[View]:
    """Frames where most of the blob is directly visible (depth-consistent), spread in time."""
    c = cfg.change
    cands = []
    for i, f in enumerate(scan.frames):
        T = f.T_cw
        pc = blob.points @ T[:3, :3].T + T[:3, 3]
        z = pc[:, 2]
        ok = z > 0.1
        if ok.sum() == 0:
            continue
        h, w = f.depth.shape
        K = f.K_depth
        u = K[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
        v = K[1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
        ok &= (u >= 0) & (u < w - 0.5) & (v >= 0) & (v < h - 0.5)
        ui, vi = np.round(u[ok]).astype(int).clip(0, w - 1), np.round(v[ok]).astype(int).clip(0, h - 1)
        d = f.depth[vi, ui]
        vis = np.abs(d - z[ok]) < c.depth_tol_base + c.depth_tol_rel * z[ok]
        nvis = int(vis.sum())
        if nvis < 10:
            continue
        H, W = f.rgb.shape[:2]
        su, sv = W / w, H / h
        uu, vv = u[ok][vis] * su, v[ok][vis] * sv
        x0, x1, y0, y1 = uu.min(), uu.max(), vv.min(), vv.max()
        pad = 0.2 * max(x1 - x0, y1 - y0) + 12
        box = (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(W, x1 + pad)), int(min(H, y1 + pad)))
        # prefer views where the blob is whole (not cut by the border) and near the centre
        full = nvis / len(blob.points)
        centred = 1 - np.hypot((x0 + x1) / 2 / W - 0.5, (y0 + y1) / 2 / H - 0.5)
        cands.append((nvis * (0.5 + full) * centred, i, box, nvis))
    cands.sort(reverse=True)
    views: list[View] = []
    for _, i, box, nvis in cands:
        if all(abs(i - v.frame) > max(3, len(scan.frames) // 20) for v in views):
            views.append(View(i, box, nvis))
        if len(views) == n:
            break
    return views


def crops(blob: Blob, scan: Scan) -> list[np.ndarray]:
    out = []
    for v in blob.views:
        x0, y0, x1, y1 = v.box
        out.append(scan.frames[v.frame].rgb[y0:y1, x0:x1])
    return out


def size_cost(a: Blob, b: Blob, up: np.ndarray) -> float:
    ea, eb = a.extent(up), b.extent(up)
    return float(np.mean(np.abs(ea - eb) / np.maximum(np.maximum(ea, eb), 0.05)))


def colour_cost(a: Blob, b: Blob) -> float:
    return float(np.linalg.norm(np.median(a.colors, 0) - np.median(b.colors, 0)))


def _yaw_of(T: np.ndarray, up: np.ndarray) -> float:
    a = np.cross(up, [1.0, 0, 0])
    a = a if np.linalg.norm(a) > 0.1 else np.cross(up, [0, 0, 1.0])
    a /= np.linalg.norm(a)
    b = T[:3, :3] @ a
    return float(np.degrees(np.arctan2(np.cross(a, b) @ up, a @ b)))


def _colour_err(a: Blob, b: Blob, T: np.ndarray) -> float:
    moved = a.points @ T[:3, :3].T + T[:3, 3]
    d, j = cKDTree(b.points).query(moved)
    m = d < 0.03
    return float(np.median(np.linalg.norm(a.colors[m] - b.colors[j[m]], axis=1))) if m.sum() > 10 else np.inf


def estimate_move(a: Blob, b: Blob, up: np.ndarray) -> tuple[np.ndarray, float, list[float]]:
    """Rigid before->after transform. Returns (T, shape error, other turns that fit as well).

    ICP from 12 yaw seeds. Every distinct turn whose shape error is close to the best is a candidate
    (a square box fits at 0/90/180/270). The surface texture then decides; if it cannot, the other
    turns are returned so the report can say the turn is ambiguous."""
    cands = _icp_candidates(a, b, up)
    best_err = cands[0][1]
    close = [c for c in cands if c[1] <= 1.6 * best_err + 0.005]  # partial views make the true fit a bit worse
    distinct = []
    for T, e in close:
        y = _yaw_of(T, up)
        if all(abs((y - _yaw_of(T2, up) + 180) % 360 - 180) > 15 for T2, _ in distinct):
            distinct.append((T, e))
    if len(distinct) == 1:
        return distinct[0][0], distinct[0][1], []
    col = [_colour_err(a, b, T) for T, _ in distinct]
    order = np.argsort(col)
    if col[order[0]] < 0.75 * col[order[1]]:  # texture clearly prefers one turn
        T, e = distinct[order[0]]
        return T, e, []
    # texture cannot decide: prefer the smallest turn (objects are usually moved without turning)
    k = int(np.argmin([abs(_yaw_of(T2, up)) for T2, _ in distinct]))
    T, e = distinct[k]
    return T, e, [_yaw_of(T2, up) for i, (T2, _) in enumerate(distinct) if i != k]


def _icp_candidates(a: Blob, b: Blob, up: np.ndarray) -> list[tuple[np.ndarray, float]]:
    pa = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a.points))
    pb = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(b.points))
    out = []
    for yaw in np.radians(np.arange(0, 360, 30)):
        T0 = np.eye(4)
        T0[:3, :3] = yaw_matrix(yaw, up)
        T0[:3, 3] = b.centroid - T0[:3, :3] @ a.centroid
        r = o3d.pipelines.registration.registration_icp(
            pa, pb, 0.05, T0, o3d.pipelines.registration.TransformationEstimationPointToPoint())
        # symmetric error so partial views do not win by overlapping a small part
        moved = np.asarray(o3d.geometry.PointCloud(pa).transform(r.transformation).points)
        d1 = np.asarray(pb.compute_point_cloud_distance(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(moved))))
        d2 = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(moved)).compute_point_cloud_distance(pb))
        out.append((np.asarray(r.transformation), float(np.median(d1) + np.median(d2))))
    return sorted(out, key=lambda c: c[1])


def motion_type(ch: "Change", up: np.ndarray) -> str:
    """Class-agnostic kind of rigid motion, from where the rotation axis is:
    inside the object -> turn_in_place; at its edge -> hinge (a door); no turn, short -> slide."""
    yaw = abs(ch.yaw(up))
    if yaw < 5:
        return "slide" if ch.distance < 0.6 else "move"
    if yaw < 15:
        return "move"
    R, t = ch.T_move[:3, :3], ch.T_move[:3, 3]
    # pivot c on the rotation axis: (I - R) c = t, solved in the horizontal plane
    a = np.cross(up, [1.0, 0, 0]); a = a if np.linalg.norm(a) > 0.1 else np.cross(up, [0, 0, 1.0]); a /= np.linalg.norm(a)
    B = np.stack([a, np.cross(up, a)], 1)                   # 3x2 plane basis
    M = B.T @ (np.eye(3) - R) @ B
    c = B @ np.linalg.lstsq(M, B.T @ t, rcond=None)[0]
    flat = _flat(ch.before.points, up)
    ctr = flat.mean(0)
    radius = np.percentile(np.linalg.norm(flat - ctr, axis=1), 95)
    rel = np.linalg.norm(_flat(c[None], up)[0] - ctr) / max(radius, 1e-3)
    if rel < 0.35:
        return "turn_in_place"
    if rel < 1.2:
        return "hinge"
    return "move"


def build_changes(removed: list[Blob], added: list[Blob], scan_a: Scan, scan_b: Scan, surf_a, surf_b,
                  floor_h: float, min_shift: float, cfg, state_a=None, state_b=None) -> list[Change]:
    up = scan_b.up / np.linalg.norm(scan_b.up)
    vox = cfg.recon.points_voxel
    removed = merge_same_object([grow_object(b, surf_a, up, floor_h, cfg, state_a) for b in removed], vox)
    added = merge_same_object([grow_object(b, surf_b, up, floor_h, cfg, state_b) for b in added], vox)
    for b in removed:
        b.views = find_views(b, scan_a, cfg)
    for b in added:
        b.views = find_views(b, scan_b, cfg)
    for blobs, scan in ((removed, scan_a), (added, scan_b)):
        for b in blobs:
            c = crops(b, scan)
            b.embedding = features.embed(c).mean(0) if c else None

    oc = cfg.objects
    pairs = []
    if removed and added:
        C = np.full((len(removed), len(added)), 9.0)
        for i, r in enumerate(removed):
            for j, a in enumerate(added):
                if r.embedding is None or a.embedding is None:
                    continue
                app = 1 - float(r.embedding @ a.embedding) / (
                    np.linalg.norm(r.embedding) * np.linalg.norm(a.embedding))
                col, size = colour_cost(r, a), size_cost(r, a, up)
                # gates: a move keeps its colour and its size; only then compare the total
                if col > oc.max_colour_diff or size > oc.max_size_diff:
                    continue
                C[i, j] = app + size + col
        ri, ai = linear_sum_assignment(C)
        pairs = [(i, j) for i, j in zip(ri, ai) if C[i, j] < oc.match_max_cost]

    changes: list[Change] = []
    used_r, used_a = {i for i, _ in pairs}, {j for _, j in pairs}
    for i, j in pairs:
        T, err, alts = estimate_move(removed[i], added[j], up)
        ch = Change(0, "moved", removed[i], added[j], T, yaw_alternatives=alts, fit_error=err)
        if err > oc.max_fit_error or height_change(removed[i], added[j], up) > oc.max_height_change:
            # colour matches, but no rigid motion explains the shape (or its height changed by more than half)
            ch.status, ch.T_move, ch.yaw_alternatives = "modified", None, []
        elif ch.distance < min_shift and abs(ch.yaw(up)) < oc.min_turn:
            continue  # same object, same place: noise from pose or depth, not a change
        else:
            ch.motion = motion_type(ch, up)
        changes.append(ch)
    # a removed and an added region in the same place and colour that are not a rigid move: the shape changed
    left_r = [i for i in range(len(removed)) if i not in used_r]
    left_a = [j for j in range(len(added)) if j not in used_a]
    for i in list(left_r):
        for j in list(left_a):
            if (footprint_iou(removed[i], added[j], up) > oc.modified_overlap
                    and colour_cost(removed[i], added[j]) < oc.max_colour_diff):
                changes.append(Change(0, "modified", removed[i], added[j]))
                left_r.remove(i)
                left_a.remove(j)
                break
    changes += [Change(0, "removed", removed[i], None) for i in left_r]
    changes += [Change(0, "added", None, added[j]) for j in left_a]
    changes.sort(key=lambda c: -len(c.blob.points))
    for k, c in enumerate(changes):
        c.id = k + 1
    return changes
