"""Score a run against hand-made ground truth.

usage: python eval/run_eval.py out/scene1/changes.json data/scene1/gt.json

gt.json format (coordinates in the 'after' recording frame; see README for how to label):
{"changes": [{"label": "chair", "status": "moved", "before": [x,y,z], "after": [x,y,z], "distance": 1.2},
             {"label": "box", "status": "removed", "before": [x,y,z]},
             {"label": "bag", "status": "added", "after": [x,y,z]}],
 "unobserved_in_after": [{"label": "lamp", "center": [x,y,z]}]}

Matching: a prediction matches a GT change when the status is the same and the
relevant centre (after for moved/added, before for removed) is within --radius.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def key_point(c: dict) -> np.ndarray:
    side = "before" if c["status"] == "removed" else "after"
    v = c[side]
    return np.array(v["center"] if isinstance(v, dict) else v, float)


def evaluate(pred: dict, gt: dict, radius: float = 0.4) -> dict:
    P, G = pred["changes"], gt["changes"]
    used, tp, rows = set(), 0, []
    for g in G:
        best, bd = None, radius
        for i, p in enumerate(P):
            if i in used or p["status"] != g["status"]:
                continue
            d = float(np.linalg.norm(key_point(p) - key_point(g)))
            if d < bd:
                best, bd = i, d
        row = {"gt": f'{g["status"]} {g["label"]}', "matched": best is not None}
        if best is not None:
            used.add(best)
            tp += 1
            p = P[best]
            row.update(pred_label=p["label"], centre_err_m=round(bd, 3))
            if g["status"] == "moved" and "distance" in g:
                row["move_err_m"] = round(abs(p["facts"]["moved_m"] - g["distance"]), 3)
        rows.append(row)
    false_unobserved = 0
    for u in gt.get("unobserved_in_after", []):
        c = np.array(u["center"])
        false_unobserved += any(p["status"] == "removed" and np.linalg.norm(key_point(p) - c) < radius for p in P)
    return {
        "precision": round(tp / max(1, len(P)), 3),
        "recall": round(tp / max(1, len(G)), 3),
        "n_pred": len(P), "n_gt": len(G),
        "unobserved_reported_as_removed": int(false_unobserved),
        "per_change": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("gt")
    ap.add_argument("--radius", type=float, default=0.4)
    a = ap.parse_args()
    res = evaluate(json.load(open(a.pred)), json.load(open(a.gt)), a.radius)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
