import numpy as np

from freespace.align import fit_yaw, ransac_yaw, yaw_matrix
from freespace.change import CHANGED, UNCHANGED, UNKNOWN, classify
from freespace.config import load_config
from freespace.scan import transform_points

UP = np.array([0.0, 1.0, 0.0])


def make_T(yaw_deg, t):
    T = np.eye(4)
    T[:3, :3] = yaw_matrix(np.radians(yaw_deg), UP)
    T[:3, 3] = t
    return T


def test_fit_yaw_exact():
    rng = np.random.default_rng(0)
    pa = rng.uniform(-2, 2, (50, 3))
    T = make_T(37, [0.8, 0.1, -1.3])
    est = fit_yaw(pa, transform_points(T, pa), UP)
    assert np.allclose(est, T, atol=1e-9)


def test_ransac_yaw_with_outliers():
    rng = np.random.default_rng(1)
    pa = rng.uniform(-2, 2, (400, 3))
    T = make_T(-120, [2.0, 0.0, 0.5])
    pb = transform_points(T, pa) + rng.normal(0, 0.01, pa.shape)
    out = rng.random(len(pa)) < 0.6  # 60% outliers (moved objects, bad matches)
    pb[out] = rng.uniform(-3, 3, (out.sum(), 3))
    est, inl = ransac_yaw(pa, pb, UP, 2000, 0.05)
    assert np.abs(est[:3, 3] - T[:3, 3]).max() < 0.03
    assert np.abs(est[:3, :3] - T[:3, :3]).max() < 0.02
    assert inl[~out].mean() > 0.9


def test_classify_three_states():
    cfg = load_config()
    n_hit = np.array([10, 1, 0, 2])
    n_free = np.array([0, 9, 1, 0])
    s = classify(n_hit, n_free, cfg)
    assert list(s) == [UNCHANGED, CHANGED, UNKNOWN, UNKNOWN]
