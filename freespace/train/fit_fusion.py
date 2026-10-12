"""Fit the 4 fusion weights on random synthetic rooms and compare with the hand rule.

usage: python -m freespace.train.fit_fusion data/train --test 4

Per point features come from the ray test (freespace/fusion.py). Labels come from the object
boxes: a point is 'changed' if it lies on an object that is not in the same place in the other walk.
Point-level F1 on held-out rooms is printed for the hand rule and for the learned weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..change import accumulate
from ..config import load_config
from ..fusion import FEATURES, features
from ..io import load_stray
from ..recon import fuse
from ..scan import invert, transform_points


def _same(o: dict, others: list[dict]) -> bool:
    return any(o["size"] == x["size"] and o["pos"] == x["pos"] and o["yaw"] == x["yaw"] for x in others)


def _on_objects(p_true: np.ndarray, objs: list[dict], margin: float = 0.02) -> np.ndarray:
    """Points on the object surface; the floor under and around it (base + 2 cm) is not the object."""
    hit = np.zeros(len(p_true), bool)
    for o in objs:
        c, s = np.cos(np.radians(o["yaw"])), np.sin(np.radians(o["yaw"]))
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # rotation about +y, as in synth._mesh
        local = (p_true - np.array([o["pos"][0], o["base"], o["pos"][1]])) @ R
        sx, sy, sz = o["size"]
        hit |= ((np.abs(local[:, 0]) <= sx / 2 + margin) & (local[:, 1] >= 0.02) & (local[:, 1] <= sy + margin)
                & (np.abs(local[:, 2]) <= sz / 2 + margin))
    return hit


def pair_data(folder: Path, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Features, labels, and the (n_hit, n_free) counts for the hand rule, for both directions."""
    scene = json.loads((folder / "scene.json").read_text())
    G = np.array(scene["T_after_before"])
    A, B = load_stray(folder / "before", cfg), load_stray(folder / "after", cfg)
    sa, sb = fuse(A, cfg).transformed(G), fuse(B, cfg)
    A = A.transformed(G)
    changed_a = [o for o in scene["before"] if not _same(o, scene["after"])]
    changed_b = [o for o in scene["after"] if not _same(o, scene["before"])]
    Xs, ys, counts = [], [], []
    for surf, other, objs in ((sa, B, changed_a), (sb, A, changed_b)):
        n_hit, n_free, res = accumulate(surf.points, surf.normals, other, cfg)
        keep = (n_hit + n_free) >= cfg.change.min_obs
        y = _on_objects(transform_points(invert(G), surf.points), objs)
        Xs.append(features(n_hit, n_free, res)[keep])
        ys.append(y[keep])
        counts.append(np.stack([n_hit, n_free], 1)[keep])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(counts), None


def f1(pred: np.ndarray, y: np.ndarray) -> float:
    tp = (pred & y).sum()
    return float(2 * tp / max(1, pred.sum() + y.sum()))


def fit(X: np.ndarray, y: np.ndarray, steps: int = 3000) -> np.ndarray:
    """Plain logistic regression, then the bias is shifted so that score > 0.5 is the
    F1-best threshold on the training points (no class re-weighting: it moves the boundary)."""
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    lin = torch.nn.Linear(X.shape[1], 1)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(lin(Xt).squeeze(1), yt)
        loss.backward()
        opt.step()
    w = np.concatenate([lin.bias.detach().numpy(), lin.weight.detach().numpy()[0]]).astype(float)
    logit = w[0] + X @ w[1:]
    best = max(np.linspace(-4, 4, 161), key=lambda t: f1(logit > t, y))
    w[0] -= best
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--test", type=int, default=4, help="number of held-out pairs")
    a = ap.parse_args()
    cfg = load_config()
    pairs = sorted(p for p in a.data.iterdir() if (p / "scene.json").exists())
    data = [pair_data(p, cfg) for p in pairs]
    train, test = data[:-a.test], data[-a.test:]
    X = np.concatenate([d[0] for d in train]); y = np.concatenate([d[1] for d in train])
    w = fit(X, y)
    print(f"train points {len(y)}, changed {y.mean():.1%}")
    print("weights [bias, " + ", ".join(FEATURES) + "] =", np.round(w, 3).tolist())

    from ..fusion import score
    for name, rows in (("train", train), ("test", test)):
        Xe = np.concatenate([d[0] for d in rows]); ye = np.concatenate([d[1] for d in rows])
        C = np.concatenate([d[2] for d in rows])
        hand = C[:, 1] >= cfg.change.free_ratio * C.sum(1)
        learned = score(Xe, w) > 0.5
        print(f"{name:5s}: point F1 hand rule {f1(hand, ye):.3f}  learned {f1(learned, ye):.3f}  "
              f"(false-positive rate hand {(hand & ~ye).sum() / max(1, (~ye).sum()):.4f}, "
              f"learned {(learned & ~ye).sum() / max(1, (~ye).sum()):.4f})")


if __name__ == "__main__":
    main()
