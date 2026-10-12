"""Tiny learned fusion of the ray evidence: logistic regression with 4 readable weights.

    score = sigmoid(w0 + w1 * see_through_ratio + w2 * log(1 + n_observations) + w3 * hit_residual)

Trained on random synthetic rooms where every point's label is known (freespace/train/).
The weights live in config.yaml (fusion.weights). Empty weights = the hand rule in change.classify.
"""
from __future__ import annotations

import numpy as np

FEATURES = ["see_through_ratio", "log_observations", "hit_residual"]


def features(n_hit: np.ndarray, n_free: np.ndarray, hit_res: np.ndarray | None) -> np.ndarray:
    n_obs = n_hit + n_free
    ratio = n_free / np.maximum(n_obs, 1)
    res = np.zeros(len(n_hit)) if hit_res is None else hit_res
    return np.stack([ratio, np.log1p(n_obs), res], 1)


def score(X: np.ndarray, w) -> np.ndarray:
    w = np.asarray(w, float)
    return 1 / (1 + np.exp(-(w[0] + X @ w[1:])))
