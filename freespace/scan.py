"""Core data types shared by all stages."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    idx: int                 # frame number in the source recording
    depth: np.ndarray        # (h, w) float32 metres, 0 = invalid
    K_depth: np.ndarray      # (3, 3) intrinsics at depth resolution
    rgb: np.ndarray          # (H, W, 3) uint8, RGB order
    K_rgb: np.ndarray        # (3, 3) intrinsics at rgb resolution
    T_wc: np.ndarray         # (4, 4) camera-to-world, OpenCV camera axes (x right, y down, z forward)

    @property
    def T_cw(self) -> np.ndarray:
        return invert(self.T_wc)


@dataclass
class Scan:
    name: str
    frames: list[Frame]
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))

    def transformed(self, T: np.ndarray) -> "Scan":
        """Same scan with all poses moved by world transform T (new_world <- old_world)."""
        frames = [Frame(f.idx, f.depth, f.K_depth, f.rgb, f.K_rgb, T @ f.T_wc) for f in self.frames]
        return Scan(self.name, frames, T[:3, :3] @ self.up)


def invert(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


def backproject(depth: np.ndarray, K: np.ndarray, uv: np.ndarray | None = None) -> np.ndarray:
    """Pixels -> camera-frame 3D points. With uv=None, uses every pixel (row-major)."""
    if uv is None:
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u, v, z = u.ravel(), v.ravel(), depth.ravel()
    else:
        u, v = uv[:, 0], uv[:, 1]
        z = depth[np.clip(v.round().astype(int), 0, depth.shape[0] - 1),
                  np.clip(u.round().astype(int), 0, depth.shape[1] - 1)]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], -1)


def project(p_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = np.maximum(p_cam[:, 2], 1e-6)
    return np.stack([K[0, 0] * p_cam[:, 0] / z + K[0, 2], K[1, 1] * p_cam[:, 1] / z + K[1, 2]], -1)


def scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    K = K.copy()
    K[0] *= sx
    K[1] *= sy
    return K
