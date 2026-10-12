"""2D navigation view of the change: occupancy grids before/after and a path that must re-plan.

Cells are 5 cm. A cell is OCCUPIED if surface points sit in the robot's body height
band, FREE if a scan saw floor there and nothing in the band, else UNKNOWN.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

FREE, OCC, UNK = 0, 1, 2


@dataclass
class Grid:
    origin: np.ndarray   # (2,) min corner in plane coords
    res: float
    basis: np.ndarray    # (2, 3) plane axes (world)
    before: np.ndarray   # (H, W) FREE/OCC/UNK
    after: np.ndarray


def plane_basis(up: np.ndarray) -> np.ndarray:
    a = np.cross(up, [1.0, 0, 0])
    if np.linalg.norm(a) < 0.1:
        a = np.cross(up, [0, 0, 1.0])
    a /= np.linalg.norm(a)
    return np.stack([a, np.cross(up, a)])


def occupancy(points: np.ndarray, up: np.ndarray, floor_h: float, basis: np.ndarray,
              origin: np.ndarray, shape: tuple[int, int], res: float, band=(0.08, 1.6)) -> np.ndarray:
    h = points @ up - floor_h
    uv = ((points @ basis.T - origin) / res).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < shape[0])
    g = np.full(shape, UNK, np.int8)
    floor = ok & (h < 0.05)
    g[uv[floor, 1], uv[floor, 0]] = FREE
    body = ok & (h > band[0]) & (h < band[1])
    g[uv[body, 1], uv[body, 0]] = OCC
    return g


def build(points_a: np.ndarray, points_b: np.ndarray, up: np.ndarray, floor_h: float, res: float = 0.05) -> Grid:
    basis = plane_basis(up)
    allp = np.concatenate([points_a, points_b]) @ basis.T
    origin = allp.min(0) - 0.1
    W, H = ((allp.max(0) + 0.1 - origin) / res).astype(int) + 1
    a = occupancy(points_a, up, floor_h, basis, origin, (H, W), res)
    b = occupancy(points_b, up, floor_h, basis, origin, (H, W), res)
    return Grid(origin, res, basis, a, b)


def astar(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int], inflate: int = 5) -> list[tuple[int, int]]:
    """8-connected A* on FREE cells after inflating obstacles by the robot radius (cells)."""
    blocked = binary_dilation(grid == OCC, iterations=inflate) | (grid == UNK)
    H, W = grid.shape
    if blocked[start] or blocked[goal]:
        return []
    moves = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    g = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in prev:
                cur = prev[cur]
                path.append(cur)
            return path[::-1]
        for dy, dx, c in moves:
            n = (cur[0] + dy, cur[1] + dx)
            if not (0 <= n[0] < H and 0 <= n[1] < W) or blocked[n]:
                continue
            ng = g[cur] + c
            if ng < g.get(n, np.inf):
                g[n], prev[n] = ng, cur
                heapq.heappush(pq, (ng + np.hypot(n[0] - goal[0], n[1] - goal[1]), n))
    return []


def demo_paths(grid: Grid, inflate: int = 5):
    """Pick two far-apart cells free in both grids; plan before and after."""
    both = (grid.before == FREE) & (grid.after == FREE)
    safe = both & ~binary_dilation((grid.before == OCC) | (grid.after == OCC), iterations=inflate)
    ys, xs = np.nonzero(safe)
    if len(ys) < 2:
        return None
    pts = np.stack([ys, xs], 1)
    i = np.argmin(pts[:, 0] + pts[:, 1])
    j = np.argmax(pts[:, 0] + pts[:, 1])
    s, t = tuple(pts[i]), tuple(pts[j])
    return s, t, astar(grid.before, s, t, inflate), astar(grid.after, s, t, inflate)


def path_length(path, res: float) -> float:
    if len(path) < 2:
        return float("nan")
    p = np.array(path, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() * res)
