"""End-to-end: two recordings in, change list + 3D views out."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console

from . import appearance, change, costmap, describe, nbv, objects
from .align import Alignment, align
from .io import load_stray
from .recon import Surface, fuse
from .scan import Scan

con = Console()


@dataclass
class Result:
    scan_a: Scan             # already in the shared (B) frame
    scan_b: Scan
    surf_a: Surface
    surf_b: Surface
    ev_a: change.Evidence    # on surf_a points: CHANGED = removed
    ev_b: change.Evidence    # on surf_b points: CHANGED = added
    naive_a: np.ndarray
    naive_b: np.ndarray
    alignment: Alignment
    changes: list[objects.Change]
    floor_h: float
    grid: costmap.Grid
    paths: tuple | None
    views: list = None       # where to look next (nbv.Viewpoint)


class Timer:
    def __init__(self, what: str):
        self.what = what

    def __enter__(self):
        self.t = time.time()
        con.print(f"[bold cyan]>[/] {self.what} ...")

    def __exit__(self, *a):
        con.print(f"  done in {time.time() - self.t:.1f}s")


def run(before: Path, after: Path, cfg, vlm: str = "local", cache_dir: Path | None = None,
        T_init: np.ndarray | None = None, baseline: str = "freespace") -> Result:
    """baseline='naive' replaces the ray test by the plain cloud-to-cloud distance (for the comparison table);
    everything after it (objects, moves, words) stays the same."""
    with Timer("load recordings"):
        A, B = load_stray(before, cfg), load_stray(after, cfg)
        con.print(f"  before: {len(A.frames)} keyframes, after: {len(B.frames)} keyframes")
    with Timer("fuse each recording (TSDF)"):
        sa, sb = fuse(A, cfg), fuse(B, cfg)
        con.print(f"  surface points: before {len(sa.points)}, after {len(sb.points)}")
    with Timer("align before -> after"):
        al = align(A, B, sa, sb, cfg) if T_init is None else Alignment(T_init, 0, 0, 1, 0, "given")
        con.print(f"  {al.summary()}")
    A, sa = A.transformed(al.T_ba), sa.transformed(al.T_ba)

    with Timer("visibility-aware change test"):
        ev_a = change.detect(sa.points, sa.normals, B, cfg)
        ev_b = change.detect(sb.points, sb.normals, A, cfg)
        for name, ev in (("before", ev_a), ("after", ev_b)):
            n = np.bincount(ev.state, minlength=3)
            con.print(f"  {name}: unchanged {n[0]}, changed {n[1]}, unknown {n[2]}")
    naive_a = change.naive_diff(sa.points, sb.points)
    naive_b = change.naive_diff(sb.points, sa.points)
    if baseline == "naive":
        ev_a = change.Evidence(ev_a.n_hit, ev_a.n_free, naive_a.copy(), ev_a.hit_res)
        ev_b = change.Evidence(ev_b.n_hit, ev_b.n_free, naive_b.copy(), ev_b.hit_res)
        con.print("  [yellow]baseline: naive distance diff instead of the ray test[/]")

    up = B.up / np.linalg.norm(B.up)
    floor_h = float(np.percentile(np.concatenate([sa.points, sb.points]) @ up, 2))
    with Timer("group into objects and match moves"):
        removed = objects.cluster(sa.points, sa.colors, ev_a.state, "before", cfg)
        added = objects.cluster(sb.points, sb.colors, ev_b.state, "after", cfg)
        min_shift = max(cfg.objects.min_shift, 3 * al.icp_rmse)
        changes = objects.build_changes(removed, added, A, B, sa, sb, floor_h, min_shift, cfg,
                                        ev_a.state, ev_b.state)

    if cfg.appearance.enabled:
        with Timer("colour check where both scans saw the surface"):
            near = [b.points for c in changes for b in (c.before, c.after) if b is not None]
            flat = appearance.flat_changes(sa, ev_a, sb, up, near, cfg)
            for c in flat:
                for blob, scan in ((c.before, A), (c.after, B)):
                    if blob is not None:
                        blob.views = objects.find_views(blob, scan, cfg)
            changes += flat
            for k, c in enumerate(changes):
                c.id = k + 1
            con.print(f"  {len(flat)} flat / surface changes")

    with Timer(f"describe changes ({vlm})"):
        describe.describe_all(changes, A, B, floor_h, sb, cfg, vlm=vlm, cache_dir=cache_dir)
        add_evidence(changes, sa, ev_a, sb, ev_b)
        for c in changes:
            con.print(f"  #{c.id} [bold]{c.status}[/] {c.label}: {c.description}")

    grid = costmap.build(sa.points, sb.points, up, floor_h)
    paths = costmap.demo_paths(grid)
    res = Result(A, B, sa, sb, ev_a, ev_b, naive_a, naive_b, al, changes, floor_h, grid, paths)
    with Timer("where to look next"):
        res.views = nbv.suggest(res, cfg)
        for k, v in enumerate(res.views, 1):
            con.print(f"  view {k}: covers {v.fraction:.0%} of the unseen surface")
    return res


def add_evidence(changes, sa, ev_a, sb, ev_b) -> None:
    """Ray counts behind each change, so the report can say *why* (e.g. '9 of 10 views saw through it')."""
    from scipy.spatial import cKDTree

    for c in changes:
        if c.flat:
            c.facts["evidence"] = "colour only (too thin for the depth test)"
            continue
        side, surf, ev = (c.before, sa, ev_a) if c.before is not None else (c.after, sb, ev_b)
        _, idx = cKDTree(surf.points).query(side.changed if side.changed is not None else side.points)
        n_free, n_obs = int(np.median(ev.n_free[idx])), int(np.median(ev.n_obs[idx]))
        if side is c.before:
            c.facts["evidence"] = f"median {n_free} of {n_obs} views of the second walk saw through the old surface"
        else:
            c.facts["evidence"] = f"median {n_free} of {n_obs} views of the first walk saw empty space where it is now"


def change_record(c: objects.Change, up: np.ndarray) -> dict:
    b = c.blob
    rec = {
        "id": c.id,
        "status": c.status,
        "label": c.label,
        "attributes": c.attributes,
        "description": c.description,
        "confidence": round(c.confidence(), 2),
        "facts": c.facts,
    }
    for side in ("before", "after"):
        blob = getattr(c, side)
        if blob is not None:
            lo, hi = blob.points.min(0), blob.points.max(0)
            rec[side] = {"center": blob.centroid.round(3).tolist(),
                         "bbox_min": lo.round(3).tolist(), "bbox_max": hi.round(3).tolist(),
                         "n_points": int(len(blob.points))}
    if c.T_move is not None:
        rec["transform_before_to_after"] = np.round(c.T_move, 4).tolist()
    return rec


def save_crops(res: Result, out: Path) -> None:
    """One image per change and side (crops/<id>_before.jpg, crops/<id>_after.jpg), for review."""
    import cv2

    (out / "crops").mkdir(parents=True, exist_ok=True)
    for c in res.changes:
        for side, scan in (("before", res.scan_a), ("after", res.scan_b)):
            blob = getattr(c, side)
            if blob is None:
                continue
            imgs = objects.crops(blob, scan)
            if imgs and imgs[0].size:
                cv2.imwrite(str(out / "crops" / f"{c.id}_{side}.jpg"), cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR))


def save(res: Result, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    save_crops(res, out)
    up = res.scan_b.up / np.linalg.norm(res.scan_b.up)
    summary = {
        "frame": "after recording (ARKit world of the second session)",
        "alignment": {"T_after_before": np.round(res.alignment.T_ba, 5).tolist(),
                      "method": res.alignment.method, "inliers": res.alignment.n_inliers,
                      "icp_fitness": round(res.alignment.icp_fitness, 3),
                      "icp_rmse_m": round(res.alignment.icp_rmse, 4)},
        "coverage": coverage_stats(res),
        "changes": [change_record(c, up) for c in res.changes],
        "look_next": [{"camera": v.position.round(3).tolist(), "look_at": v.target.round(3).tolist(),
                       "unseen_points": v.covers, "share_of_unseen": round(v.fraction, 3)} for v in (res.views or [])],
    }
    if res.paths:
        s, t, pb, pa = res.paths
        summary["navigation"] = {"path_before_m": round(costmap.path_length(pb, res.grid.res), 2),
                                 "path_after_m": round(costmap.path_length(pa, res.grid.res), 2)}
    (out / "changes.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        out / "clouds.npz",
        a_pts=res.surf_a.points.astype(np.float32), a_col=res.surf_a.colors.astype(np.float32),
        a_state=res.ev_a.state, a_naive=res.naive_a,
        b_pts=res.surf_b.points.astype(np.float32), b_col=res.surf_b.colors.astype(np.float32),
        b_state=res.ev_b.state, b_naive=res.naive_b,
    )
    return summary


def coverage_stats(res: Result) -> dict:
    def frac(s, k):
        return round(float((s == k).mean()), 3)

    return {
        "before_surface_unknown_in_after": frac(res.ev_a.state, change.UNKNOWN),
        "after_surface_unknown_in_before": frac(res.ev_b.state, change.UNKNOWN),
        "naive_changed_before": frac(res.naive_a, change.CHANGED),
        "visibility_changed_before": frac(res.ev_a.state, change.CHANGED),
        "naive_changed_after": frac(res.naive_b, change.CHANGED),
        "visibility_changed_after": frac(res.ev_b.state, change.CHANGED),
    }
