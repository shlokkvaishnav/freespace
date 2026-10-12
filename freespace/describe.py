"""Name each change and describe it in one sentence.

Numbers (distance, turn, height, direction) always come from geometry.
The labeller only names the object:
  - local (default): CLIP zero-shot over an indoor vocabulary. No key, deterministic.
  - claude (optional): a vision model sees the crops plus the geometric facts and
    writes a richer sentence. Responses are cached on disk.
"""
from __future__ import annotations

import base64
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .objects import Change, crops, support_height
from .scan import Scan

LEXICON_PATH = Path(__file__).parent / "data" / "lexicon.txt"


@lru_cache(maxsize=1)
def lexicon() -> tuple[str, ...]:
    """Open vocabulary for names only (~1200 everyday objects). Detection never reads it."""
    words = [w.strip() for w in LEXICON_PATH.read_text().splitlines() if w.strip() and not w.startswith("#")]
    return tuple(dict.fromkeys(words))


COLOURS = {
    "black": (25, 25, 25), "white": (235, 235, 235), "grey": (128, 128, 128), "red": (190, 40, 40),
    "orange": (230, 130, 40), "yellow": (225, 205, 60), "green": (50, 140, 70), "blue": (50, 80, 170),
    "purple": (120, 60, 150), "pink": (230, 150, 170), "brown": (120, 80, 45), "beige": (210, 190, 150),
}


def colour_name(rgb01: np.ndarray) -> str:
    lab = cv2.cvtColor(np.uint8([[np.clip(rgb01, 0, 1) * 255]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    best, dmin = "", 1e9
    for name, c in COLOURS.items():
        lc = cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
        d = np.linalg.norm(lab - lc)
        if d < dmin:
            best, dmin = name, d
    return best


# ---------------------------------------------------------------- geometry facts

def facts(ch: Change, scan_a: Scan, floor_h: float, up: np.ndarray, scene) -> dict:
    """Plain numbers about a change, all in metres / degrees, in the shared frame."""
    b = ch.blob
    h = b.points @ up
    ext = b.extent(up)
    out = {
        "status": ch.status,
        "size_m": {"height": round(float(ext[0]), 2), "width": round(float(ext[1]), 2), "depth": round(float(ext[2]), 2)},
        "rests_at_height_m": round(max(0.0, support_height(scene.points, scene.normals, b.points, up, floor_h) - floor_h), 2),
        "colour": colour_name(np.median(b.colors, 0)),
        "flat": ch.flat,
    }
    if ch.status == "surface_changed":
        out["colour_before"] = colour_name(np.median(ch.before.colors, 0))
    if ch.status == "moved":
        out["moved_m"] = round(ch.distance, 2)
        out["turned_deg"] = round(ch.yaw(up))
        out["direction"] = direction_words(ch.after.centroid - ch.before.centroid, scan_a, up)
        out["motion"] = ch.motion
        out["turn_alternatives_deg"] = [round(abs(y)) for y in ch.yaw_alternatives]
    return out


def direction_words(d: np.ndarray, scan_a: Scan, up: np.ndarray) -> str:
    """Move direction as seen from where the first recording started."""
    T = scan_a.frames[0].T_wc
    fwd = T[:3, 2] - (T[:3, 2] @ up) * up
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, up)  # with +up and OpenCV forward, this points to the camera's right
    right = -right if right @ T[:3, 0] < 0 else right
    d = d - (d @ up) * up
    f, r = d @ fwd, d @ right
    parts = []
    if abs(r) > 0.15:
        parts.append("to the right" if r > 0 else "to the left")
    if abs(f) > 0.15:
        parts.append("further from the start position" if f > 0 else "closer to the start position")
    return " and ".join(parts)


def height_words(hm: float) -> str:
    if hm < 0.08:
        return "on the floor"
    return f"on a surface {hm:.2f} m above the floor"


def template_sentence(label: str, f: dict) -> str:
    name = f"{f['colour']} {'object' if label == 'unknown object' else label}".strip()
    if f["status"] == "moved":
        turn = f"{abs(f['turned_deg'])}°"
        alts = f.get("turn_alternatives_deg") or []
        if 0 < len(alts) <= 2:
            turn += f" (or {' or '.join(f'{a}°' for a in alts)}: its shape looks the same and the texture cannot tell)"
        elif len(alts) > 2:
            turn += " (uncertain: the shape and texture are too plain to measure the turn)"
        motion = f.get("motion")
        if motion == "hinge":
            return f"The {name} swung {turn} about one of its edges, like a door opening or closing."
        if motion == "turn_in_place":
            return f"The {name} turned {turn} in place."
        if motion == "slide":
            return f"The {name} slid {f['moved_m']:.2f} m in a straight line without turning" + (
                f" ({f['direction']})." if f["direction"] else ".")
        s = f"The {name} moved {f['moved_m']:.2f} m" + (f" {f['direction']}" if f["direction"] else "")
        if abs(f["turned_deg"]) >= 10:
            s += f" and turned {turn}"
        return s + f". It is now {height_words(f['rests_at_height_m'])}."
    if f["status"] == "modified":
        return (f"The {name} {height_words(f['rests_at_height_m'])} changed shape: no rigid move explains it "
                f"(folded, opened or deformed).")
    size = f"about {max(f['size_m']['width'], f['size_m']['height']):.1f} m"
    if f["status"] == "surface_changed":
        return (f"The surface {height_words(f['rests_at_height_m'])} looks different now "
                f"({f['colour_before']} before, {f['colour']} now). Seen by colour only.")
    flat = "flat " if f.get("flat") else ""
    if f["status"] == "added":
        return f"A new {flat}{name} ({size}) appeared {height_words(f['rests_at_height_m'])}."
    return f"The {name} ({size}) that was {height_words(f['rests_at_height_m'])} is gone."


# ---------------------------------------------------------------- local labeller (CLIP)

def clip_label(images: list[np.ndarray], min_conf: float) -> tuple[str, float]:
    """Best lexicon word for the averaged crop embedding; 'unknown object' when CLIP is unsure."""
    from .features import embed, name_scores

    e = embed(images).mean(0)
    p = name_scores(e, lexicon())
    k = int(p.argmax())
    if p[k] < min_conf:
        return "unknown object", float(p[k])
    return lexicon()[k], float(p[k])


# ---------------------------------------------------------------- optional Claude labeller

def _jpeg_b64(img: np.ndarray, max_side: int = 512) -> str:
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.standard_b64encode(buf.tobytes()).decode()


SYSTEM = (
    "You label changes found by a robot that compared two 3D scans of the same room. "
    "You see image crops of one changed object (before and/or after) and measured facts. "
    "Name the object, give short visual attributes, and write one plain sentence a robot "
    "operator can act on. Use the measured numbers exactly as given; do not invent numbers. "
    "If you can see nearby landmarks (desk, bed, door, window), use them for location."
)


def claude_label(images_before: list[np.ndarray], images_after: list[np.ndarray], f: dict,
                 model: str, cache_dir: Path) -> dict | None:
    from pydantic import BaseModel

    class Label(BaseModel):
        label: str
        attributes: str
        description: str

    content = []
    for tag, imgs in (("BEFORE", images_before), ("AFTER", images_after)):
        for im in imgs:
            content.append({"type": "text", "text": f"{tag} crop:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                         "data": _jpeg_b64(im)}})
    content.append({"type": "text", "text": "Measured facts (JSON):\n" + json.dumps(f)})

    key = hashlib.sha256(json.dumps([model, SYSTEM, content], sort_keys=True).encode()).hexdigest()[:20]
    cache = cache_dir / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    import anthropic

    client = anthropic.Anthropic()
    try:
        resp = client.messages.parse(
            model=model,
            max_tokens=2000,
            system=SYSTEM,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": content}],
            output_format=Label,
        )
    except anthropic.APIError as e:
        print(f"  claude call failed ({type(e).__name__}): {e}; using local label")
        return None
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        return None
    out = resp.parsed_output.model_dump()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------- entry point

def describe_all(changes: list[Change], scan_a: Scan, scan_b: Scan, floor_h: float, scene, cfg,
                 vlm: str = "local", cache_dir: Path | None = None) -> None:
    up = scan_b.up / np.linalg.norm(scan_b.up)
    n = cfg.describe.views_per_change
    for ch in changes:
        before = crops(ch.before, scan_a)[:n] if ch.before is not None else []
        after = crops(ch.after, scan_b)[:n] if ch.after is not None else []
        f = facts(ch, scan_a, floor_h, up, scene)
        imgs = [im for im in before + after if im.size and min(im.shape[:2]) >= 16]
        if imgs:
            ch.label, ch.label_score = clip_label(imgs, cfg.describe.min_name_conf)
        ch.attributes = f["colour"]
        ch.description = template_sentence(ch.label, f)
        if vlm == "claude" and imgs:
            res = claude_label(before, after, f, cfg.describe.model, cache_dir or Path(".cache/claude"))
            if res:
                ch.label, ch.attributes, ch.description = res["label"], res["attributes"], res["description"]
        ch.facts = f
