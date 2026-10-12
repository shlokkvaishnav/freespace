"""Rerun recording of the result: both scans, change layers, objects, cameras, costmap."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr

from . import costmap
from .change import CHANGED, UNKNOWN

RED, GREEN, BLUE, GREY = (230, 60, 50), (40, 200, 90), (60, 120, 255), (150, 150, 160)
PURPLE = (170, 90, 220)
STATUS_COLOUR = {"removed": RED, "added": GREEN, "moved": BLUE, "surface_changed": PURPLE}


def grid_image(g: np.ndarray) -> np.ndarray:
    img = np.zeros((*g.shape, 3), np.uint8)
    img[g == costmap.FREE] = (235, 235, 235)
    img[g == costmap.OCC] = (40, 40, 40)
    img[g == costmap.UNK] = (170, 190, 210)
    return img


def log_rerun(res, path: Path) -> None:
    rec = rr.RecordingStream("freespace")
    rec.save(str(path))
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    sa, sb = res.surf_a, res.surf_b
    rec.log("world/before/surface", rr.Points3D(sa.points, colors=sa.colors, radii=0.008), static=True)
    rec.log("world/after/surface", rr.Points3D(sb.points, colors=sb.colors, radii=0.008), static=True)

    m = res.ev_a.state == CHANGED
    rec.log("world/changes/removed_surface", rr.Points3D(sa.points[m], colors=RED, radii=0.012), static=True)
    m = res.ev_b.state == CHANGED
    rec.log("world/changes/added_surface", rr.Points3D(sb.points[m], colors=GREEN, radii=0.012), static=True)
    m = res.ev_a.state == UNKNOWN
    rec.log("world/changes/unknown_before_not_seen_after", rr.Points3D(sa.points[m], colors=GREY, radii=0.01), static=True)
    m = res.naive_a == CHANGED
    rec.log("world/ablation/naive_removed", rr.Points3D(sa.points[m], colors=(255, 140, 0), radii=0.01), static=True)

    for c in res.changes:
        col = STATUS_COLOUR[c.status]
        for side in ("before", "after"):
            b = getattr(c, side)
            if b is None:
                continue
            lo, hi = b.points.min(0), b.points.max(0)
            tag = f"#{c.id} {c.label} ({c.status}{', was here' if c.status == 'moved' and side == 'before' else ''})"
            rec.log(f"world/objects/{c.id}/{side}",
                    rr.Boxes3D(centers=[(lo + hi) / 2], half_sizes=[(hi - lo) / 2], labels=[tag], colors=[col]),
                    static=True)
        if c.status == "moved":
            rec.log(f"world/objects/{c.id}/arrow",
                    rr.Arrows3D(origins=[c.before.centroid], vectors=[c.after.centroid - c.before.centroid],
                                colors=[col], labels=[f"{c.distance:.2f} m"]), static=True)
        rec.log(f"world/objects/{c.id}/description", rr.TextDocument(c.description), static=True)

    for name, scan, col in (("before", res.scan_a, (255, 180, 0)), ("after", res.scan_b, (0, 200, 255))):
        traj = np.stack([f.T_wc[:3, 3] for f in scan.frames])
        rec.log(f"world/cameras/{name}", rr.LineStrips3D([traj], colors=[col]), static=True)

    rec.log("map/before", rr.Image(grid_image(res.grid.before)), static=True)
    rec.log("map/after", rr.Image(grid_image(res.grid.after)), static=True)
    if res.paths:
        s, t, pb, pa = res.paths
        if pb:
            rec.log("map/before/path", rr.LineStrips2D([np.array(pb)[:, ::-1]], colors=[(255, 140, 0)]), static=True)
        if pa:
            rec.log("map/after/path", rr.LineStrips2D([np.array(pa)[:, ::-1]], colors=[(0, 160, 255)]), static=True)
