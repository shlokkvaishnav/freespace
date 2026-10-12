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


def test_small_move_on_table(tmp_path_factory, tmp_path):
    """A shoebox on the table moves 8 cm: found as moved, not merged with the table."""
    data = make_pair(tmp_path_factory.mktemp("small"), n_frames=80, small=True)
    cfg = load_config()
    gt = json.loads((data / "gt.json").read_text())
    res = pipeline.run(data / "before", data / "after", cfg, vlm="local",
                       T_init=np.array(gt["T_after_before"]))
    score = evaluate(pipeline.save(res, tmp_path), gt)
    box = next(r for r in score["per_change"] if r["gt"] == "moved shoebox")
    assert box["matched"], score
    assert box["move_err_m"] < 0.07, score
    assert score["precision"] == 1.0, score


@pytest.mark.parametrize("colour_check", [True, False])
def test_flat_object_needs_colour_check(tmp_path_factory, tmp_path, colour_check):
    """A 1.5 cm magazine on the floor is too thin for the depth test: only the colour check finds it."""
    data = make_pair(tmp_path_factory.mktemp("flat"), n_frames=80, flat=True)
    cfg = load_config(overrides={"appearance.enabled": colour_check})
    gt = json.loads((data / "gt.json").read_text())
    res = pipeline.run(data / "before", data / "after", cfg, vlm="local",
                       T_init=np.array(gt["T_after_before"]))
    score = evaluate(pipeline.save(res, tmp_path), gt)
    mag = next(r for r in score["per_change"] if r["gt"] == "added magazine")
    assert mag["matched"] == colour_check, score
    assert score["precision"] == 1.0, score


def _run(data, cfg=None):
    gt = json.loads((data / "gt.json").read_text())
    res = pipeline.run(data / "before", data / "after", cfg or load_config(), vlm="local",
                       T_init=np.array(gt["T_after_before"]))
    import tempfile
    from pathlib import Path
    summary = pipeline.save(res, Path(tempfile.mkdtemp()))
    return summary, gt, evaluate(summary, gt)


def _moved_near(summary, centre, radius=0.4):
    for c in summary["changes"]:
        if c["status"] in ("moved", "modified") and "after" in c and \
                np.linalg.norm(np.array(c["after"]["center"]) - np.array(centre)) < radius:
            return c
    return None


def test_symmetric_box_turn_resolved_by_texture(synth):
    """The chair is a square box: 40 deg and 130 deg fit the shape equally; the texture must pick 40."""
    summary, gt, _ = _run(synth)
    chair = _moved_near(summary, gt["changes"][0]["after"])
    assert chair is not None and chair["status"] == "moved"
    assert abs(abs(chair["facts"]["turned_deg"]) - 40) < 10, chair["facts"]
    assert chair["facts"]["turn_alternatives_deg"] == []


def test_door_swing_is_a_hinge(tmp_path_factory):
    data = make_pair(tmp_path_factory.mktemp("hinge"), n_frames=80, hinge=True)
    summary, gt, score = _run(data)
    assert score["recall"] == 1.0 and score["precision"] == 1.0, score
    door = _moved_near(summary, gt["changes"][-1]["after"])
    assert door["facts"]["motion"] == "hinge", door["facts"]
    turns = [abs(door["facts"]["turned_deg"])] + door["facts"]["turn_alternatives_deg"]
    assert any(abs(t - 70) < 10 for t in turns), door["facts"]


def test_folded_blanket_is_modified(tmp_path_factory):
    data = make_pair(tmp_path_factory.mktemp("folded"), n_frames=80, folded=True)
    summary, gt, score = _run(data)
    assert score["recall"] == 1.0 and score["precision"] == 1.0, score
    blanket = next(r for r in score["per_change"] if r["gt"] == "modified blanket")
    assert blanket["matched"], score


def test_rich_scene_end_to_end(tmp_path_factory, tmp_path):
    """The demo room: every kind of change at once, with the alignment estimated (not given)."""
    data = make_pair(tmp_path_factory.mktemp("rich"), n_frames=80, small=True, hinge=True, folded=True, odd=True)
    gt = json.loads((data / "gt.json").read_text())
    res = pipeline.run(data / "before", data / "after", load_config(), vlm="local")
    score = evaluate(pipeline.save(res, tmp_path), gt)
    assert score["precision"] == 1.0 and score["recall"] == 1.0, score
    assert score["unobserved_reported_as_removed"] == 0, score
