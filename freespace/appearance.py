"""Colour evidence for changes the depth test cannot see (a book or a sheet of paper on a desk).

Same rule as the ray test: compare only where *both* scans saw the same surface
(points the depth test calls UNCHANGED). Each fused TSDF point already carries a colour
averaged over every frame that saw it, so the check is simple:

1. Skip points near edges and creases (their fused colour mixes two surfaces).
   Pair each 'before' point with the nearest 'after' point (within a few cm, same surface direction).
2. Remove the global exposure difference: one gain per colour channel (robust median).
3. Compare chromaticity (hue, not brightness). Flag pairs whose colour differs by more than max(min_diff, median + k * MAD).
4. Neighbourhood vote, then cluster. The height difference decides the kind:
   new surface higher -> flat object added, lower -> flat object removed,
   same height -> 'surface changed', but only if the colour name changes (e.g. white -> blue).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from .change import CHANGED, UNCHANGED, UNKNOWN, smooth_states
from .describe import colour_name
from .objects import Blob, Change


def near_edge(points: np.ndarray, normals: np.ndarray, radius: float, min_dot: float = 0.8) -> np.ndarray:
    """True near creases and object edges, where a fused colour mixes two surfaces and depends on the viewpoint."""
    tree = cKDTree(points)
    out = np.zeros(len(points), bool)
    for i, nb in enumerate(tree.query_ball_point(points, r=radius)):
        out[i] = np.abs(normals[nb] @ normals[i]).min() < min_dot
    return out


def colour_flags(sa, ev_a, sb, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (flag over 'before' points, index of the paired 'after' point or -1, colour difference)."""
    ac = cfg.appearance
    n = len(sa.points)
    pair = np.full(n, -1)
    diff = np.zeros(n)
    both = np.flatnonzero((ev_a.state == UNCHANGED) & ~near_edge(sa.points, sa.normals, ac.edge_radius))
    if both.size == 0:
        return np.zeros(n, bool), pair, diff
    d, j = cKDTree(sb.points).query(sa.points[both])
    # same surface only: close *and* facing the same way (else a floor point pairs with a table side)
    ok = (d < ac.match_radius) & (np.abs(np.einsum("ij,ij->i", sa.normals[both], sb.normals[j])) > ac.min_normal_dot)
    both, j = both[ok], j[ok]
    ca, cb = sa.colors[both], sb.colors[j]
    gain = np.median(cb, 0) / np.maximum(np.median(ca, 0), 1e-3)
    # chromaticity (r, g, b divided by their sum): a new object changes the hue,
    # while shading, exposure and texture blending mostly change the brightness
    chroma = lambda c: c / (c.sum(1, keepdims=True) + 0.05)
    dd = np.linalg.norm(chroma(np.clip(ca * gain, 0, 1)) - chroma(cb), axis=1)
    mad = np.median(np.abs(dd - np.median(dd))) + 1e-6
    thr = max(ac.min_diff, np.median(dd) + ac.k_mad * mad)
    pair[both], diff[both] = j, dd

    state = np.full(n, UNKNOWN, np.int8)
    state[both] = np.where(dd > thr, CHANGED, UNCHANGED)
    state = smooth_states(sa.points, state, radius=cfg.recon.points_voxel * 2.0)
    return state == CHANGED, pair, diff


def flat_changes(sa, ev_a, sb, up: np.ndarray, near: list[np.ndarray], cfg) -> list[Change]:
    """Changes found only by colour. `near` = point sets of changes already found by geometry (skipped)."""
    ac = cfg.appearance
    flag, pair, _ = colour_flags(sa, ev_a, sb, cfg)
    if near:
        known = np.concatenate(near)
        d, _ = cKDTree(known).query(sa.points)
        flag &= d > ac.skip_near
    idx = np.flatnonzero(flag)
    if idx.size < ac.min_points:
        return []
    lab = DBSCAN(eps=cfg.objects.cluster_eps, min_samples=5).fit_predict(sa.points[idx])
    out = []
    for l in set(lab) - {-1}:
        m = idx[lab == l]
        if m.size < ac.min_points or np.ptp(sa.points[m], axis=0).max() < cfg.objects.min_extent:
            continue
        mb = pair[m]
        before = Blob("before", sa.points[m], sa.colors[m], changed=sa.points[m])
        after = Blob("after", sb.points[mb], sb.colors[mb], changed=sb.points[mb])
        dh = float(np.median(sb.points[mb] @ up - sa.points[m] @ up))
        if dh > ac.flat_height:
            out.append(Change(0, "added", None, after, flat=True))
        elif dh < -ac.flat_height:
            out.append(Change(0, "removed", before, None, flat=True))
        elif colour_name(np.median(before.colors, 0)) != colour_name(np.median(after.colors, 0)):
            # no height cue: the weakest evidence, so the colour must change name (white -> blue), not just shade
            out.append(Change(0, "surface_changed", before, after, flat=True))
    return out
