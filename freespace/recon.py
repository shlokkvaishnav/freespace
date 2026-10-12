"""Per-scan TSDF fusion -> mesh and an oriented surface point cloud."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c

from .scan import Scan


@dataclass
class Surface:
    points: np.ndarray   # (N, 3)
    normals: np.ndarray  # (N, 3), unit, oriented towards the cameras
    colors: np.ndarray   # (N, 3) float 0..1
    mesh: o3d.geometry.TriangleMesh | None = None

    def pcd(self) -> o3d.geometry.PointCloud:
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.points))
        p.normals = o3d.utility.Vector3dVector(self.normals)
        p.colors = o3d.utility.Vector3dVector(self.colors)
        return p

    def transformed(self, T: np.ndarray) -> "Surface":
        mesh = None
        if self.mesh is not None:
            mesh = o3d.geometry.TriangleMesh(self.mesh).transform(T)
        R = T[:3, :3]
        return Surface(self.points @ R.T + T[:3, 3], self.normals @ R.T, self.colors, mesh)


def fuse(scan: Scan, cfg) -> Surface:
    """TSDF fusion with Open3D's tensor VoxelBlockGrid (the legacy ScalableTSDFVolume
    returns an empty surface with Open3D 0.20 on macOS arm64)."""
    rc = cfg.recon
    cpu = o3c.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=rc.voxel,
        block_resolution=16,
        block_count=60000,
        device=cpu,
    )
    trunc_mult = rc.sdf_trunc / rc.voxel
    for f in scan.frames:
        h, w = f.depth.shape
        rgb = cv2.resize(f.rgb, (w, h), interpolation=cv2.INTER_AREA)
        depth = o3d.t.geometry.Image(np.ascontiguousarray(f.depth))
        color = o3d.t.geometry.Image(np.ascontiguousarray(rgb.astype(np.float32) / 255.0))
        K = o3c.Tensor(f.K_depth, o3c.float64)
        E = o3c.Tensor(np.ascontiguousarray(f.T_cw), o3c.float64)
        coords = vbg.compute_unique_block_coordinates(depth, K, E, 1.0, cfg.io.max_depth, trunc_mult)
        vbg.integrate(coords, depth, color, K, E, 1.0, cfg.io.max_depth, trunc_mult)

    mesh = vbg.extract_triangle_mesh().to_legacy()
    mesh.compute_vertex_normals()
    pcd = vbg.extract_point_cloud().to_legacy().voxel_down_sample(rc.points_voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rc.points_voxel * 3, max_nn=30))
    pts = np.asarray(pcd.points)
    nrm = _orient_to_cameras(pts, np.asarray(pcd.normals), scan)
    return Surface(pts, nrm, np.clip(np.asarray(pcd.colors), 0, 1), mesh)


def _orient_to_cameras(pts: np.ndarray, nrm: np.ndarray, scan: Scan) -> np.ndarray:
    """Flip each normal to face the nearest camera centre (TSDF surfaces are seen from that side)."""
    cams = np.stack([f.T_wc[:3, 3] for f in scan.frames])
    from scipy.spatial import cKDTree

    _, j = cKDTree(cams).query(pts)
    to_cam = cams[j] - pts
    flip = np.einsum("ij,ij->i", nrm, to_cam) < 0
    nrm = nrm.copy()
    nrm[flip] *= -1
    return nrm
