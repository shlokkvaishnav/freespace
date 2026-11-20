"""Exploratory: how pose drift and mixed LiDAR confidence affect the change test (spec: docs/drift-eval.md).

usage: python -m eval.drift_eval --out eval/results/drift.json [--seeds 0 1 2] [--work out/drift_eval]

For each setting it renders the demo room pair (7 changes) with the given drift, runs the full pipeline
with estimated alignment, and scores:
  - object precision / recall (eval/run_eval.py rules),
  - false-change rate: changed surface outside every truly changed object, per m^2 of observed unchanged
    surface (both directions pooled). Reported for the ray test and, on the same points, the naive diff.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from eval.run_eval import evaluate
from freespace import change, pipeline
from freespace.config import load_config
from freespace.scan import invert, transform_points
from freespace.synth import SESSION_B, Obj, make_pair, scenes

DEMO = dict(small=True, hinge=True, folded=True, odd=True)  # the `freespace demo` room
# (drift_deg, drift_m, mixed_conf): reference = the old generator, then the sweep with mixed confidence
SETTINGS = [(0.0, 0.0, False)] + [(d, round(0.015 * d, 4), True) for d in (0.0, 0.5, 1.0, 2.0, 4.0)]


def near_objects(p_after: np.ndarray, objs: list[Obj], margin: float) -> np.ndarray:
    """Points (in the 'after' session frame) within `margin` of the box of any object (true world)."""
    p = transform_points(invert(SESSION_B), p_after)
    hit = np.zeros(len(p), bool)
    for o in objs:
        R = Rotation.from_euler("y", o.yaw, degrees=True).as_matrix()
        q = (p - [o.pos[0], o.base, o.pos[1]]) @ R  # world -> object axes
        sx, sy, sz = o.size
        hit |= ((np.abs(q[:, 0]) <= sx / 2 + margin) & (q[:, 1] >= -margin) & (q[:, 1] <= sy + margin)
                & (np.abs(q[:, 2]) <= sz / 2 + margin))
    return hit


def false_change(res, margin: float, area_per_point: float) -> dict:
    scene_a, scene_b = scenes(**DEMO)
    gone = [o for o in scene_a if o not in scene_b]   # surfaces that truly changed in 'before'
    new = [o for o in scene_b if o not in scene_a]    # ... and in 'after'
    out, tot = {}, np.zeros(3)
    for side, surf, ev, naive, objs in (("before", res.surf_a, res.ev_a, res.naive_a, gone),
                                       ("after", res.surf_b, res.ev_b, res.naive_b, new)):
        static = ~near_objects(surf.points, objs, margin) & (ev.state != change.UNKNOWN)
        n = np.array([static.sum(), (static & (ev.state == change.CHANGED)).sum(),
                      (static & (naive == change.CHANGED)).sum()], float)
        tot += n
        out[side] = {"observed_static_m2": round(n[0] * area_per_point, 2),
                     "ray_false_cm2_per_m2": round(1e4 * n[1] / max(n[0], 1), 1),
                     "naive_false_cm2_per_m2": round(1e4 * n[2] / max(n[0], 1), 1)}
    out["pooled"] = {"observed_static_m2": round(tot[0] * area_per_point, 2),
                     "ray_false_cm2_per_m2": round(1e4 * tot[1] / max(tot[0], 1), 1),
                     "ray_false_m2": round(tot[1] * area_per_point, 4),
                     "naive_false_cm2_per_m2": round(1e4 * tot[2] / max(tot[0], 1), 1)}
    return out


def align_error(T_est: np.ndarray, T_gt: np.ndarray) -> dict:
    D = invert(T_gt) @ T_est
    ang = np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))
    return {"trans_m": round(float(np.linalg.norm(D[:3, 3])), 3), "rot_deg": round(float(ang), 2)}


def run_one(work: Path, drift_deg: float, drift_m: float, mixed: bool, seed: int, cfg, margin: float) -> dict:
    data = work / f"d{drift_deg}_c{int(mixed)}_s{seed}"
    if not (data / "gt.json").exists():
        make_pair(data, 120, seed=seed, drift_deg=drift_deg, drift_m=drift_m, mixed_conf=mixed, **DEMO)
    gt = json.loads((data / "gt.json").read_text())
    t0 = time.time()
    res = pipeline.run(data / "before", data / "after", cfg, vlm="local")
    summary = pipeline.save(res, data / "out")
    score = evaluate(summary, gt)
    tp = sum(r["matched"] for r in score["per_change"])
    return {
        "drift_deg": drift_deg, "drift_m": drift_m, "mixed_confidence": mixed, "seed": seed,
        "precision": score["precision"], "recall": score["recall"],
        "n_pred": score["n_pred"], "n_gt": score["n_gt"], "false_detections": score["n_pred"] - tp,
        "missed": [r["gt"] for r in score["per_change"] if not r["matched"]],
        "detections": sorted(f'{c["status"]} {c["label"]}' for c in summary["changes"]),
        "unobserved_reported_as_removed": score["unobserved_reported_as_removed"],
        "false_change": false_change(res, margin, cfg.recon.points_voxel ** 2),
        "alignment_error": align_error(res.alignment.T_ba, np.array(gt["T_after_before"])),
        "seconds": round(time.time() - t0, 1),
    }


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for d, m, c in SETTINGS:
        rs = [r for r in rows if (r["drift_deg"], r["drift_m"], r["mixed_confidence"]) == (d, m, c)]
        if not rs:
            continue
        f = lambda k: [r[k] for r in rs]
        fc = [r["false_change"]["pooled"] for r in rs]
        out.append({
            "drift_deg": d, "drift_cm": round(100 * m, 2), "mixed_confidence": c, "runs": len(rs),
            "precision_mean": round(float(np.mean(f("precision"))), 3),
            "recall_mean": round(float(np.mean(f("recall"))), 3),
            "false_detections_total": int(sum(f("false_detections"))),
            "ray_false_cm2_per_m2_mean": round(float(np.mean([x["ray_false_cm2_per_m2"] for x in fc])), 1),
            "ray_false_cm2_per_m2_max": float(max(x["ray_false_cm2_per_m2"] for x in fc)),
            "naive_false_cm2_per_m2_mean": round(float(np.mean([x["naive_false_cm2_per_m2"] for x in fc])), 1),
            "align_trans_m_max": max(r["alignment_error"]["trans_m"] for r in rs),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/drift.json")
    ap.add_argument("--work", default="out/drift_eval")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--margin", type=float, default=0.10, help="metres around a changed object that do not count")
    a = ap.parse_args()
    cfg = load_config()
    rows = []
    for d, m, c in SETTINGS:
        for s in a.seeds:
            rows.append(run_one(Path(a.work), d, m, c, s, cfg, a.margin))
            r = rows[-1]
            print(f"[drift {d} deg / {100 * m:.2f} cm, mixed={c}, seed {s}] P={r['precision']} R={r['recall']} "
                  f"false={r['false_change']['pooled']['ray_false_cm2_per_m2']} cm2/m2 ({r['seconds']} s)", flush=True)
    result = {"spec": "docs/drift-eval.md", "label": "exploratory", "scene": "freespace demo room (7 changes), 120 frames",
              "alignment": "estimated", "margin_m": a.margin, "summary": summarise(rows), "runs": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
