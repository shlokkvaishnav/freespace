"""End-to-end on the synthetic room: known moved / removed / added / unseen objects."""
import json

import numpy as np
import pytest

from freespace import pipeline
from freespace.change import UNKNOWN
from freespace.config import load_config
from freespace.synth import make_pair
from eval.run_eval import evaluate


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return make_pair(tmp_path_factory.mktemp("synth"), n_frames=60)


@pytest.mark.parametrize("use_gt_align", [True, False])
def test_pipeline_finds_changes(synth, tmp_path, use_gt_align):
    cfg = load_config()
    gt = json.loads((synth / "gt.json").read_text())
    T = np.array(gt["T_after_before"]) if use_gt_align else None
    res = pipeline.run(synth / "before", synth / "after", cfg, vlm="local", T_init=T)
    summary = pipeline.save(res, tmp_path)

    if not use_gt_align:
        err = np.abs(res.alignment.T_ba - np.array(gt["T_after_before"]))
        assert err[:3, 3].max() < 0.05, res.alignment.summary()

    score = evaluate(summary, gt)
    assert score["recall"] == 1.0, score
    assert score["precision"] >= 0.75, score
    assert score["unobserved_reported_as_removed"] == 0, score

    moved = [c for c in summary["changes"] if c["status"] == "moved"]
    assert moved and abs(moved[0]["facts"]["moved_m"] - gt["changes"][0]["distance"]) < 0.15

    # the lamp is never seen by the 2nd scan -> its surface must be UNKNOWN, not changed
    lamp = np.array(gt["unobserved_in_after"][0]["center"])
    near = np.linalg.norm(res.surf_a.points - lamp, axis=1) < 0.3
    assert (res.ev_a.state[near] == UNKNOWN).mean() > 0.8


def test_object_outside_lexicon_is_detected(tmp_path_factory, tmp_path):
    """Detection is class-agnostic: an odd shape/colour no word list expects is still found and placed."""
    data = make_pair(tmp_path_factory.mktemp("odd"), n_frames=60, odd=True)
    cfg = load_config()
    gt = json.loads((data / "gt.json").read_text())
    res = pipeline.run(data / "before", data / "after", cfg, vlm="local",
                       T_init=np.array(gt["T_after_before"]))
    summary = pipeline.save(res, tmp_path)

    score = evaluate(summary, gt)
    odd = next(r for r in score["per_change"] if r["gt"] == "added ring sculpture")
    assert odd["matched"], score
    assert score["recall"] == 1.0, score
