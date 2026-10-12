"""Synthetic room pair in Stray Scanner format, with ground truth.

Used for tests and to run the full pipeline without a phone. The two scans
differ by: one moved+rotated object, one removed object, one added object, and
one object that scan B never sees (must come out UNKNOWN, not "removed").
Scan B is written in its own world frame (random yaw + offset), like a second
ARKit session.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

ROOM = np.array([5.0, 2.6, 4.0])  # x (width), y (height, up), z (depth)
RGB_W, RGB_H, FX = 960, 720, 780.0
D_W, D_H = 256, 192


@dataclass
class Obj:
    name: str
    size: tuple[float, float, float]          # x, y, z extents
    pos: tuple[float, float]                  # x, z of the footprint centre (rests on floor or `base`)
    color: tuple[int, int, int]
    yaw: float = 0.0
    base: float = 0.0                         # height of the bottom face
    shape: str = "box"                        # box | cylinder | torus


STATIC = [
    Obj("table", (1.4, 0.75, 0.8), (3.2, 2.6), (150, 100, 60)),
    Obj("bookshelf", (1.0, 1.8, 0.35), (1.0, 3.75), (90, 60, 40)),
    Obj("sofa", (1.8, 0.8, 0.85), (1.1, 0.6), (60, 80, 140)),
]
CHAIR_A = Obj("chair", (0.5, 0.95, 0.5), (2.4, 1.6), (200, 40, 40))
CHAIR_B = Obj("chair", (0.5, 0.95, 0.5), (3.6, 1.2), (200, 40, 40), yaw=40.0)
BOX = Obj("cardboard box", (0.45, 0.35, 0.4), (4.3, 1.0), (190, 160, 110))
SUITCASE = Obj("suitcase", (0.45, 0.7, 0.25), (1.9, 1.9), (30, 140, 90))
LAMP = Obj("lamp", (0.3, 1.5, 0.3), (4.6, 3.6), (230, 220, 120), shape="cylinder")  # hidden from B

SCENE_A = STATIC + [CHAIR_A, BOX, LAMP]
SCENE_B = STATIC + [CHAIR_B, SUITCASE, LAMP]
# a small object on the table that moves only 8 cm (tests: objects on a shared surface, small moves)
SHOEBOX_A = Obj("shoebox", (0.32, 0.12, 0.2), (3.0, 2.4), (230, 120, 40), base=0.75)
SHOEBOX_B = Obj("shoebox", (0.32, 0.12, 0.2), (3.08, 2.4), (230, 120, 40), base=0.75)
# a magazine lying flat on the floor: too thin for the depth test, only colour can find it
BOOK = Obj("magazine", (0.3, 0.015, 0.22), (2.95, 1.95), (40, 70, 170))
# a free-standing door panel that swings 70 degrees about its hinge edge (articulation)
DOOR_A = Obj("door", (0.8, 1.9, 0.04), (4.6, 2.0), (200, 190, 170), yaw=90.0)
# a blanket lying on the floor, folded in half between the walks (non-rigid change)
BLANKET_A = Obj("blanket", (0.5, 0.04, 0.36), (3.0, 1.85), (120, 160, 200))
BLANKET_B = Obj("blanket", (0.25, 0.09, 0.36), (2.875, 1.85), (120, 160, 200))


def swung(o: Obj, deg: float, edge: float = -1.0) -> Obj:
    """The same panel rotated by `deg` about its vertical edge at local x = edge * size_x / 2."""
    R = lambda a: Rotation.from_euler("y", a, degrees=True).as_matrix()
    hinge = np.array([o.pos[0], 0, o.pos[1]]) + R(o.yaw) @ np.array([edge * o.size[0] / 2, 0, 0])
    centre = hinge - R(o.yaw + deg) @ np.array([edge * o.size[0] / 2, 0, 0])
    return Obj(o.name, o.size, (float(centre[0]), float(centre[2])), o.color, yaw=o.yaw + deg)


DOOR_B = swung(DOOR_A, 70.0)
# an object no word list expects: an upright purple ring (used to test class-agnostic detection)
ODD = Obj("ring sculpture", (0.66, 0.66, 0.16), (2.9, 0.7), (140, 60, 170), shape="torus")


def _mesh(o: Obj) -> o3d.geometry.TriangleMesh:
    sx, sy, sz = o.size
    if o.shape == "torus":
        r_tube = sz / 2
        m = o3d.geometry.TriangleMesh.create_torus(torus_radius=sx / 2 - r_tube, tube_radius=r_tube)
        m.translate((0, sy / 2, 0))  # ring stands upright in the x-y plane
    elif o.shape == "cylinder":
        m = o3d.geometry.TriangleMesh.create_cylinder(radius=sx / 2, height=sy, resolution=24)
        m.rotate(Rotation.from_euler("x", -90, degrees=True).as_matrix(), center=(0, 0, 0))
        m.translate((0, sy / 2, 0))
    else:
        m = o3d.geometry.TriangleMesh.create_box(sx, sy, sz)
        m.translate((-sx / 2, 0, -sz / 2))
    m.rotate(Rotation.from_euler("y", o.yaw, degrees=True).as_matrix(), center=(0, 0, 0))
    m.translate((o.pos[0], o.base, o.pos[1]))
    return m


def _room_meshes() -> list[tuple[str, o3d.geometry.TriangleMesh, tuple[int, int, int]]]:
    W, H, D = ROOM
    t = 0.05
    parts = [
        ("floor", (W, t, D), (0, -t, 0), (170, 170, 175)),
        ("ceiling", (W, t, D), (0, H, 0), (240, 240, 240)),
        ("wall", (t, H, D), (-t, 0, 0), (220, 215, 200)),
        ("wall", (t, H, D), (W, 0, 0), (215, 220, 205)),
        ("wall", (W, H, t), (0, 0, -t), (205, 215, 220)),
        ("wall", (W, H, t), (0, 0, D), (225, 210, 210)),
    ]
    out = []
    for name, size, off, col in parts:
        m = o3d.geometry.TriangleMesh.create_box(*size)
        m.translate(off)
        out.append((name, m, col))
    return out


def _to_local(o: Obj) -> np.ndarray:
    """World -> object frame, so an object's texture moves with it (like a real object)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("y", o.yaw, degrees=True).as_matrix()
    T[:3, 3] = (o.pos[0], o.base, o.pos[1])
    return np.linalg.inv(T)


class Renderer:
    def __init__(self, objs: list[Obj]):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.colors, self.to_local = [], []
        room = [(name, m, col, None) for name, m, col in _room_meshes()]
        for name, m, col, T in room + [(o.name, _mesh(o), o.color, _to_local(o)) for o in objs]:
            self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
            self.colors.append(np.array(col, np.float32))
            self.to_local.append(T)
        self.colors = np.stack(self.colors)

    def render(self, T_wc: np.ndarray, w: int, h: int, fx: float):
        u, v = np.meshgrid(np.arange(w), np.arange(h))  # same pixel convention as the loader
        d_cam = np.stack([(u - w / 2) / fx, (v - h / 2) / fx, np.ones_like(u)], -1).reshape(-1, 3)
        d_w = d_cam @ T_wc[:3, :3].T
        o_w = np.broadcast_to(T_wc[:3, 3], d_w.shape)
        rays = o3d.core.Tensor(np.concatenate([o_w, d_w], 1).astype(np.float32))
        ans = self.scene.cast_rays(rays)
        z = ans["t_hit"].numpy().reshape(h, w)  # direction has unit camera-z, so t == depth
        gid = ans["geometry_ids"].numpy().reshape(h, w)
        z[~np.isfinite(z)] = 0
        hit = (o_w + d_w * z.reshape(-1, 1)).reshape(h, w, 3)
        return z, gid, hit

    def rgb(self, T_wc: np.ndarray) -> np.ndarray:
        z, gid, hit = self.render(T_wc, RGB_W, RGB_H, FX)
        base = self.colors[np.clip(gid, 0, len(self.colors) - 1)]
        local = hit.copy()
        for g, T in enumerate(self.to_local):
            if T is not None:
                m = gid == g
                local[m] = hit[m] @ T[:3, :3].T + T[:3, 3]
        img = base * _texture(local)[..., None]
        return np.clip(img, 0, 255).astype(np.uint8)

    def depth(self, T_wc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        z, _, _ = self.render(T_wc, D_W, D_H, FX * D_W / RGB_W)
        noise = rng.normal(0, 0.004 + 0.004 * z)
        return np.where(z > 0, z + noise, 0).astype(np.float32)


def _texture(p: np.ndarray) -> np.ndarray:
    """Solid 3D value-noise texture, so features are consistent across views."""
    out = np.zeros(p.shape[:2], np.float32)
    for scale, amp in ((0.25, 0.25), (0.08, 0.15), (0.03, 0.1)):
        c = np.floor(p / scale).astype(np.int64)
        hsh = (c[..., 0] * 73856093) ^ (c[..., 1] * 19349663) ^ (c[..., 2] * 83492791)
        out += amp * ((hsh % 1000) / 1000.0 - 0.5)
    return 0.85 + out


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world, OpenCV axes (x right, y down, z forward), world +Y up."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 1, 0])  # right = forward x up  (with y-down camera)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = eye
    return T


def trajectory(kind: str, n: int) -> list[np.ndarray]:
    poses = []
    for t in np.linspace(0, 1, n):
        if kind == "A":  # walk along the front wall, look into the room
            eye = np.array([0.7 + 3.6 * t, 1.45, 0.35 + 0.15 * np.sin(6 * t)])
            target = np.array([2.5 + 1.8 * np.sin(2 * np.pi * t), 0.6, 2.6])
        else:  # walk along the back-left of the room, look towards the front-right
            eye = np.array([0.6 + 1.8 * t, 1.3, 2.4 + 0.9 * t])
            target = np.array([4.2 - 1.2 * t, 0.4, 0.8 - 0.2 * t])
        poses.append(look_at(eye, target))
    return poses


def write_stray(folder: Path, renderer: Renderer, poses: list[np.ndarray], G: np.ndarray, seed: int):
    """Render poses (true world) and write them in a session frame: T_session = G @ T_true."""
    rng = np.random.default_rng(seed)
    (folder / "depth").mkdir(parents=True, exist_ok=True)
    (folder / "confidence").mkdir(exist_ok=True)
    vw = cv2.VideoWriter(str(folder / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (RGB_W, RGB_H))
    rows = []
    for i, T in enumerate(poses):
        vw.write(cv2.cvtColor(renderer.rgb(T), cv2.COLOR_RGB2BGR))
        d = renderer.depth(T, rng)
        cv2.imwrite(str(folder / "depth" / f"{i:06d}.png"), (d * 1000).astype(np.uint16))
        cv2.imwrite(str(folder / "confidence" / f"{i:06d}.png"), np.full(d.shape, 2, np.uint8))
        Ts = G @ T
        q = Rotation.from_matrix(Ts[:3, :3]).as_quat()
        rows.append([i / 30, i, *Ts[:3, 3], *q])
    vw.release()
    np.savetxt(folder / "odometry.csv", np.array(rows), delimiter=",",
               header="timestamp, frame, x, y, z, qx, qy, qz, qw", comments="")
    K = np.array([[FX, 0, RGB_W / 2], [0, FX, RGB_H / 2], [0, 0, 1]])
    np.savetxt(folder / "camera_matrix.csv", K, delimiter=",")


def yaw_transform(deg: float, t: tuple[float, float, float]) -> np.ndarray:
    G = np.eye(4)
    G[:3, :3] = Rotation.from_euler("y", deg, degrees=True).as_matrix()
    G[:3, 3] = t
    return G


def make_pair(out: str | Path, n_frames: int = 120, seed: int = 0, odd: bool = False, small: bool = False,
              flat: bool = False, hinge: bool = False, folded: bool = False) -> Path:
    """odd=True also adds an object with a shape and colour outside any word list.
    small=True also puts a shoebox on the table and moves it by 8 cm.
    flat=True also adds a 1.5 cm magazine lying on the floor.
    hinge=True adds a door panel that swings 70 degrees about its edge.
    folded=True adds a blanket on the floor that is folded in half (a non-rigid change)."""
    out = Path(out)
    scene_a = SCENE_A + ([SHOEBOX_A] if small else []) + ([DOOR_A] if hinge else []) + ([BLANKET_A] if folded else [])
    scene_b = SCENE_B + ([ODD] if odd else []) + ([SHOEBOX_B] if small else []) + ([BOOK] if flat else []) + ([DOOR_B] if hinge else []) + ([BLANKET_B] if folded else [])
    G_a = np.eye(4)
    G_b = yaw_transform(37.0, (0.8, 0.0, -1.3))  # B session frame differs from A
    write_stray(out / "before", Renderer(scene_a), trajectory("A", n_frames), G_a, seed)
    write_stray(out / "after", Renderer(scene_b), trajectory("B", n_frames), G_b, seed + 1)

    def centre(o: Obj, G: np.ndarray) -> list[float]:
        p = np.array([o.pos[0], o.base + o.size[1] / 2, o.pos[1]])
        return (G[:3, :3] @ p + G[:3, 3]).round(3).tolist()

    gt = {
        "frame": "after",  # coordinates below are in the 'after' session frame
        "T_after_before": G_b.tolist(),
        "changes": [
            {"label": "chair", "status": "moved", "before": centre(CHAIR_A, G_b), "after": centre(CHAIR_B, G_b),
             "distance": float(np.hypot(CHAIR_B.pos[0] - CHAIR_A.pos[0], CHAIR_B.pos[1] - CHAIR_A.pos[1])),
             "yaw": CHAIR_B.yaw - CHAIR_A.yaw},
            {"label": "cardboard box", "status": "removed", "before": centre(BOX, G_b)},
            {"label": "suitcase", "status": "added", "after": centre(SUITCASE, G_b)},
        ],
        "unobserved_in_after": [{"label": "lamp", "center": centre(LAMP, G_b)}],
    }
    if odd:
        gt["changes"].append({"label": ODD.name, "status": "added", "after": centre(ODD, G_b)})
    if hinge:
        gt["changes"].append({"label": "door", "status": "moved", "motion": "hinge", "before": centre(DOOR_A, G_b),
                              "after": centre(DOOR_B, G_b), "yaw": 70.0,
                              "distance": float(np.hypot(DOOR_B.pos[0] - DOOR_A.pos[0], DOOR_B.pos[1] - DOOR_A.pos[1]))})
    if folded:
        gt["changes"].append({"label": "blanket", "status": "modified", "before": centre(BLANKET_A, G_b),
                              "after": centre(BLANKET_B, G_b)})
    if flat:
        gt["changes"].append({"label": "magazine", "status": "added", "after": centre(BOOK, G_b)})
    if small:
        gt["changes"].append({"label": "shoebox", "status": "moved", "before": centre(SHOEBOX_A, G_b),
                              "after": centre(SHOEBOX_B, G_b), "distance": 0.08, "yaw": 0.0})
    (out / "gt.json").write_text(json.dumps(gt, indent=2))
    return out
