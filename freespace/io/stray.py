"""Loader for Stray Scanner (iOS, LiDAR) recordings.

Folder layout written by the app:
    rgb.mp4               colour video (1920x1440)
    depth/000000.png      16-bit depth in millimetres (256x192)
    confidence/000000.png LiDAR confidence 0/1/2
    odometry.csv          timestamp, frame, x, y, z, qx, qy, qz, qw  (camera-to-world)
    camera_matrix.csv     3x3 intrinsics at rgb resolution
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..scan import Frame, Scan, scale_K

# ARKit cameras look down -Z with +Y up (OpenGL). Right-multiply to get OpenCV axes.
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def read_poses(folder: Path, convention: str = "opencv") -> tuple[np.ndarray, np.ndarray]:
    odo = np.loadtxt(folder / "odometry.csv", delimiter=",", skiprows=1)
    idx = odo[:, 1].astype(int)
    T = np.tile(np.eye(4), (len(odo), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(odo[:, 5:9]).as_matrix()
    T[:, :3, 3] = odo[:, 2:5]
    if convention == "opengl":
        T = T @ GL_TO_CV
    return idx, T


def select_keyframes(T: np.ndarray, min_trans: float, min_rot_deg: float, max_frames: int) -> list[int]:
    keep = [0]
    for i in range(1, len(T)):
        A, B = T[keep[-1]], T[i]
        dt = np.linalg.norm(A[:3, 3] - B[:3, 3])
        cos = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2
        dr = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        if dt > min_trans or dr > min_rot_deg:
            keep.append(i)
    if len(keep) > max_frames:
        keep = [keep[int(j)] for j in np.linspace(0, len(keep) - 1, max_frames)]
    return keep


def load_stray(folder: str | Path, cfg) -> Scan:
    folder = Path(folder)
    io = cfg.io
    frame_ids, T = read_poses(folder, io.get("pose_convention", "opencv"))
    keep = select_keyframes(T, io.keyframe_trans, io.keyframe_rot, io.max_frames)
    wanted = {int(frame_ids[k]): k for k in keep}

    K_full = np.loadtxt(folder / "camera_matrix.csv", delimiter=",")
    frames: list[Frame] = []
    cap = cv2.VideoCapture(str(folder / "rgb.mp4"))
    i = 0
    while wanted:
        ok, bgr = cap.read()
        if not ok:
            break
        if i in wanted:
            k = wanted.pop(i)
            frames.append(_make_frame(folder, i, bgr, K_full, T[k], io))
        i += 1
    cap.release()
    frames.sort(key=lambda f: f.idx)
    if not frames:
        raise RuntimeError(f"no frames loaded from {folder}")
    return Scan(folder.name, frames, np.asarray(io.up_axis, float))


def _make_frame(folder: Path, i: int, bgr: np.ndarray, K_full: np.ndarray, T_wc: np.ndarray, io) -> Frame:
    H0, W0 = bgr.shape[:2]
    depth = cv2.imread(str(folder / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    conf_path = folder / "confidence" / f"{i:06d}.png"
    if conf_path.exists():
        conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
        depth[conf < io.min_confidence] = 0.0
    depth[depth > io.max_depth] = 0.0
    h, w = depth.shape

    W = io.rgb_width
    Hs = round(H0 * W / W0)
    rgb = cv2.cvtColor(cv2.resize(bgr, (W, Hs), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return Frame(
        idx=i,
        depth=depth,
        K_depth=scale_K(K_full, w / W0, h / H0),
        rgb=rgb,
        K_rgb=scale_K(K_full, W / W0, Hs / H0),
        T_wc=T_wc,
    )
