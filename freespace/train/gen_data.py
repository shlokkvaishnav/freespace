"""Random synthetic room pairs with per-point ground truth, for training the fusion weights.

Each pair: the 3 static pieces of furniture + 5-8 random boxes. Between the two walks,
1-2 boxes move (5 cm to 1.5 m, random turn), 1 is removed, 1 is added. Labels come from
the object boxes, so no hand labelling is needed.

usage: python -m freespace.train.gen_data data/train --pairs 14
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..synth import STATIC, Obj, Renderer, trajectory, write_stray, yaw_transform


def _free_spot(rng, objs: list[Obj], size, tries: int = 200):
    for _ in range(tries):
        p = (rng.uniform(0.5, 4.5), rng.uniform(0.5, 3.5))
        if all(np.hypot(p[0] - o.pos[0], p[1] - o.pos[1]) > 0.5 * (max(size[0], size[2]) + max(o.size[0], o.size[2])) + 0.1
               for o in objs):
            return p
    return None


def _random_box(rng, objs: list[Obj], name: str) -> Obj | None:
    size = (rng.uniform(0.15, 0.9), rng.uniform(0.1, 1.1), rng.uniform(0.15, 0.7))
    pos = _free_spot(rng, objs, size)
    if pos is None:
        return None
    col = tuple(int(c) for c in rng.integers(30, 230, 3))
    return Obj(name, size, pos, col, yaw=float(rng.uniform(0, 90)))


def random_scenes(seed: int) -> tuple[list[Obj], list[Obj]]:
    rng = np.random.default_rng(seed)
    a = list(STATIC)
    for k in range(int(rng.integers(5, 9))):
        o = _random_box(rng, a, f"box{k}")
        if o:
            a.append(o)
    movable = list(range(len(STATIC), len(a)))
    rng.shuffle(movable)
    b = list(a)
    n_move = int(rng.integers(1, 3))
    for i in movable[:n_move]:
        o = a[i]
        others = [x for j, x in enumerate(b) if j != i]
        for _ in range(100):
            d = rng.uniform(0.05, 1.5)
            ang = rng.uniform(0, 2 * np.pi)
            p = (o.pos[0] + d * np.cos(ang), o.pos[1] + d * np.sin(ang))
            if 0.4 < p[0] < 4.6 and 0.4 < p[1] < 3.6 and _free_spot_ok(p, o.size, others):
                b[i] = replace(o, pos=p, yaw=o.yaw + float(rng.uniform(-60, 60)))
                break
    if len(movable) > n_move:
        b[movable[n_move]] = None  # removed
    b = [o for o in b if o is not None]
    new = _random_box(rng, b, "new")
    if new:
        b.append(new)
    return a, b


def _free_spot_ok(p, size, objs) -> bool:
    return all(np.hypot(p[0] - o.pos[0], p[1] - o.pos[1]) > 0.5 * (max(size[0], size[2]) + max(o.size[0], o.size[2])) + 0.1
               for o in objs)


def make_random_pair(out: Path, seed: int, n_frames: int = 60) -> Path:
    a, b = random_scenes(seed)
    rng = np.random.default_rng(seed + 1000)
    G = yaw_transform(float(rng.uniform(-180, 180)), (rng.uniform(-2, 2), 0.0, rng.uniform(-2, 2)))
    kinds = ["A", "B"] if rng.random() < 0.5 else ["B", "A"]  # which walk is first
    write_stray(out / "before", Renderer(a), trajectory(kinds[0], n_frames), np.eye(4), seed)
    write_stray(out / "after", Renderer(b), trajectory(kinds[1], n_frames), G, seed + 1)
    (out / "scene.json").write_text(json.dumps({
        "T_after_before": G.tolist(),
        "before": [asdict(o) for o in a],
        "after": [asdict(o) for o in b],
    }))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--pairs", type=int, default=14)
    ap.add_argument("--frames", type=int, default=60)
    a = ap.parse_args()
    for k in range(a.pairs):
        make_random_pair(a.out / f"pair_{k:02d}", seed=100 + k, n_frames=a.frames)
        print(f"wrote {a.out}/pair_{k:02d}")


if __name__ == "__main__":
    main()
