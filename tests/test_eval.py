"""Review-based scoring for real rooms (no 3D ground truth)."""
from eval.run_eval import evaluate_review

PRED = {"changes": [
    {"id": 1, "status": "moved", "label": "chair", "facts": {"moved_m": 0.95, "turned_deg": 25}},
    {"id": 2, "status": "added", "label": "bag", "facts": {}},
    {"id": 3, "status": "removed", "label": "wall", "facts": {}},
    {"id": 4, "status": "removed", "label": "box", "facts": {}},
    {"id": 5, "status": "added", "label": "box", "facts": {}},
]}
GT = {"changes": [
    {"name": "chair", "status": "moved", "distance": 0.9, "turn": 30},
    {"name": "backpack", "status": "added"},
    {"name": "box", "status": "moved", "distance": 0.5},
    {"name": "plant", "status": "removed"},
]}


def test_review_scores():
    review = {"answers": {"1": 1, "2": 2, "3": "u", "4": 3, "5": "w"}}
    s = evaluate_review(PRED, GT, review)
    assert s["precision"] == 0.4            # 2 of 5 detections right (chair, bag)
    assert s["recall"] == 0.5               # 2 of 4 real changes found
    assert s["right_object_wrong_kind"] == 1  # the box moved, but was called 'removed'
    assert s["unseen_area_called_removed"] == 1
    assert s["median_distance_err_m"] == 0.05
    assert s["median_turn_err_deg"] == 5.0
    assert s["missed"] == ["moved box", "removed plant"]
