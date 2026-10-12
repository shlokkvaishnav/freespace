"""Score a run against ground truth.

usage: python eval/run_eval.py out/scene1/changes.json data/scene1/gt.json
       python eval/run_eval.py out/room1/changes.json data/room1/gt.json --review out/room1/review.json

Synthetic scenes have 3D centres in gt.json and are matched automatically (below).
Real rooms use a short gt.json written while recording (no coordinates) and a review.json
made with `freespace label` (a person says which real change each detection is).

Real-room gt.json:
{"changes": [{"name": "chair", "status": "moved", "distance": 0.9, "turn": 30},
             {"name": "backpack", "status": "added"},
             {"name": "box", "status": "removed"}]}

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


def evaluate_review(pred: dict, gt: dict, review: dict) -> dict:
    """Score from a person's review: answers map each detection id to a real-change number (1-based),
    'w' (wrong: nothing changed there) or 'u' (says 'removed' where the second walk never looked)."""
    truth = gt["changes"]
    answers = review["answers"]
    by_id = {str(c["id"]): c for c in pred["changes"]}
    correct, wrong_kind, found, rows = 0, 0, set(), []
    for cid, a in answers.items():
        c = by_id[cid]
        row = {"id": int(cid), "detected": f'{c["status"]} {c["label"]}', "answer": a}
        if isinstance(a, int) and 1 <= a <= len(truth):
            g = truth[a - 1]
            row["real"] = f'{g["status"]} {g.get("name", g.get("label", ""))}'
            if c["status"] == g["status"]:
                correct += 1
                found.add(a)
                if g["status"] == "moved" and "distance" in g and "moved_m" in c["facts"]:
                    row["distance_err_m"] = round(abs(c["facts"]["moved_m"] - g["distance"]), 3)
                turn = g.get("turn", g.get("yaw"))
                if g["status"] == "moved" and turn is not None and "turned_deg" in c["facts"]:
                    row["turn_err_deg"] = round(abs(abs(c["facts"]["turned_deg"]) - abs(turn)), 1)
            else:
                wrong_kind += 1
        rows.append(row)
    dist = [r["distance_err_m"] for r in rows if "distance_err_m" in r]
    turn = [r["turn_err_deg"] for r in rows if "turn_err_deg" in r]
    return {
        "precision": round(correct / max(1, len(answers)), 3),
        "recall": round(len(found) / max(1, len(truth)), 3),
        "n_detected": len(answers), "n_real": len(truth),
        "right_object_wrong_kind": wrong_kind,
        "unseen_area_called_removed": sum(a == "u" for a in answers.values()),
        "median_distance_err_m": round(float(np.median(dist)), 3) if dist else None,
        "median_turn_err_deg": round(float(np.median(turn)), 1) if turn else None,
        "missed": [f'{g["status"]} {g.get("name", g.get("label", ""))}' for k, g in enumerate(truth, 1) if k not in found],
        "per_detection": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("gt")
    ap.add_argument("--radius", type=float, default=0.4)
    ap.add_argument("--review", help="review.json from `freespace label` (real rooms)")
    a = ap.parse_args()
    if a.review:
        res = evaluate_review(json.load(open(a.pred)), json.load(open(a.gt)), json.load(open(a.review)))
    else:
        res = evaluate(json.load(open(a.pred)), json.load(open(a.gt)), a.radius)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
