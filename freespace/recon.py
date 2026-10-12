"""Per-scan TSDF fusion -> mesh and an oriented surface point cloud."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d

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
    rc = cfg.recon
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=rc.voxel,
        sdf_trunc=rc.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for f in scan.frames:
        h, w = f.depth.shape
        rgb = cv2.resize(f.rgb, (w, h), interpolation=cv2.INTER_AREA)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(np.ascontiguousarray(f.depth)),
            depth_scale=1.0,
            depth_trunc=cfg.io.max_depth,
            convert_rgb_to_intensity=False,
        )
        K = f.K_depth
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        vol.integrate(rgbd, intr, f.T_cw)

    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = vol.extract_point_cloud().voxel_down_sample(rc.points_voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rc.points_voxel * 3, max_nn=30))
    pts = np.asarray(pcd.points)
    nrm = _orient_to_cameras(pts, np.asarray(pcd.normals), scan)
    return Surface(pts, nrm, np.asarray(pcd.colors), mesh)


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
