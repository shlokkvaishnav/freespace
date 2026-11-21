"""Optional sensor realism in the synthetic generator: pose drift and mixed LiDAR confidence."""
import numpy as np
from scipy.spatial.transform import Rotation

from freespace.io.stray import read_poses
from freespace.synth import (STATIC, Renderer, apply_drift, drift_errors, mixed_confidence, trajectory,
                             write_stray, yaw_transform)


def test_drift_off_is_exact():
    R, t = drift_errors(50, 0.0, 0.0, np.random.default_rng(0))
    assert np.allclose(R, np.eye(3)) and np.allclose(t, 0)


def test_drift_walk_has_the_asked_size():
    n, yaws, ends = 60, [], []
    for s in range(400):
        R, t = drift_errors(n, 2.0, 0.03, np.random.default_rng(s))
        assert np.allclose(R[0], np.eye(3)) and np.allclose(t[0], 0)  # starts at the session origin
        yaws.append(Rotation.from_matrix(R[-1]).as_euler("xyz", degrees=True)[1])
        ends.append(t[-1])
    ends = np.array(ends)
    assert 1.7 < np.std(yaws) < 2.3                      # yaw sd at the end of the walk ~ 2 degrees
    assert 0.025 < ends[:, 0].std() < 0.035              # horizontal sd ~ 3 cm
    assert ends[:, 1].std() < 0.6 * ends[:, 0].std()     # vertical drifts less


def test_drift_keeps_camera_centre_up_to_offset():
    poses = trajectory("A", 10)
    R, t = drift_errors(10, 1.0, 0.0, np.random.default_rng(1))
    out = apply_drift(poses, R, t)
    assert np.allclose([p[:3, 3] for p in out], [p[:3, 3] for p in poses])  # rotation only: same centres
    assert not np.allclose(out[-1][:3, :3], poses[-1][:3, :3])


def test_mixed_confidence_marks_edges_and_far_depth():
    d = np.full((60, 80), 2.0, np.float32)
    d[:, 40:] = 1.0                       # a depth step -> edge column
    d[:5] = 4.0                           # far rows
    noisy, conf = mixed_confidence(d, np.random.default_rng(0))
    assert set(np.unique(conf)) <= {0, 1, 2}
    assert (conf[10:, 39:41] < 2).all()   # the step is low or medium confidence
    assert (conf[:4, 5:30] == 1).all()    # far depth is medium
    mid = conf[10:, 5:30]
    assert 0.9 < (mid == 2).mean() < 0.99  # most plain surface stays high (5% random medium)


def test_defaults_write_exact_poses_and_full_confidence(tmp_path):
    import cv2

    poses = trajectory("A", 2)
    G = yaw_transform(37.0, (0.8, 0.0, -1.3))
    write_stray(tmp_path, Renderer(STATIC), poses, G, seed=0)
    _, T = read_poses(tmp_path)
    assert np.allclose(T, [G @ p for p in poses], atol=1e-6)
    conf = cv2.imread(str(tmp_path / "confidence" / "000001.png"), cv2.IMREAD_UNCHANGED)
    assert (conf == 2).all()
