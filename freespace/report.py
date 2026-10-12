"""Single-file HTML report: interactive point cloud (three.js) + change cards + nav map."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from . import costmap
from .change import CHANGED, UNKNOWN
from .objects import crops
from .viz import STATUS_COLOUR, grid_image

MAX_POINTS = 150_000


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _jpg(img: np.ndarray, side: int = 260) -> str:
    s = side / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _layer(points, colors, sel=None, limit=MAX_POINTS, seed=0):
    idx = np.arange(len(points)) if sel is None else np.flatnonzero(sel)
    if len(idx) > limit:
        idx = np.random.default_rng(seed).choice(idx, limit, replace=False)
    col = colors[idx] if colors.ndim == 2 else np.broadcast_to(colors, (len(idx), 3))
    return {"p": _b64(points[idx].astype(np.float32)), "c": _b64((np.asarray(col) * 255).astype(np.uint8)), "n": int(len(idx))}


def _map_png(grid: np.ndarray, path, colour) -> str:
    img = grid_image(grid)
    for y, x in path or []:
        img[y, x] = colour
    img = cv2.resize(img[::-1], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    return _png(img)


def write_html(res, summary: dict, path: Path) -> None:
    sa, sb = res.surf_a, res.surf_b
    red = np.array(STATUS_COLOUR["removed"]) / 255
    green = np.array(STATUS_COLOUR["added"]) / 255
    layers = {
        "after": _layer(sb.points, sb.colors),
        "before": _layer(sa.points, sa.colors, seed=1),
        "removed": _layer(sa.points, red, res.ev_a.state == CHANGED),
        "added": _layer(sb.points, green, res.ev_b.state == CHANGED),
        "unknown": _layer(sa.points, np.array([0.6, 0.6, 0.65]), res.ev_a.state == UNKNOWN),
        "naive": _layer(sa.points, np.array([1.0, 0.55, 0.0]), res.naive_a == CHANGED),
    }
    cards = []
    for c, rec in zip(res.changes, summary["changes"]):
        imgs = {}
        if c.before is not None:
            cr = crops(c.before, res.scan_a)
            imgs["before"] = _jpg(cr[0]) if cr else None
        if c.after is not None:
            cr = crops(c.after, res.scan_b)
            imgs["after"] = _jpg(cr[0]) if cr else None
        boxes = []
        for side in ("before", "after"):
            if side in rec:
                boxes.append({"side": side, "min": rec[side]["bbox_min"], "max": rec[side]["bbox_max"],
                              "center": rec[side]["center"]})
        cards.append({**{k: rec[k] for k in ("id", "status", "label", "description", "confidence")},
                      "facts": rec["facts"], "imgs": imgs, "boxes": boxes,
                      "colour": "#%02x%02x%02x" % STATUS_COLOUR[c.status]})

    nav = None
    if res.paths:
        s, t, pb, pa = res.paths
        nav = {"before": _map_png(res.grid.before, pb, (255, 140, 0)),
               "after": _map_png(res.grid.after, pa, (0, 140, 255)),
               **summary.get("navigation", {})}

    data = {"layers": layers, "changes": cards, "nav": nav, "coverage": summary["coverage"],
            "alignment": summary["alignment"], "up": (res.scan_b.up / np.linalg.norm(res.scan_b.up)).tolist()}
    html = TEMPLATE.replace("/*DATA*/", json.dumps(data))
    path.write_text(html)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Room Change Report</title>
<style>
:root{--bg:#f6f6f4;--panel:#ffffff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e3e3e0;--canvas:#ececea}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141416;--panel:#1d1d20;--ink:#ececef;--muted:#9a9aa2;--line:#2c2c31;--canvas:#18181b}}
:root[data-theme="dark"]{--bg:#141416;--panel:#1d1d20;--ink:#ececef;--muted:#9a9aa2;--line:#2c2c31;--canvas:#18181b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:18px 20px 8px}h1{font-size:20px;margin:0 0 4px}header p{margin:0;color:var(--muted)}
main{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:16px;padding:12px 20px 24px}
@media (max-width:900px){main{grid-template-columns:1fr;padding:12px 16px}}
#view{position:relative;height:72vh;min-height:420px;background:var(--canvas);border:1px solid var(--line);border-radius:10px;overflow:hidden}
#view canvas{display:block}
.toggles{position:absolute;left:10px;top:10px;display:flex;flex-wrap:wrap;gap:6px;max-width:calc(100% - 20px)}
.toggles label{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:3px 8px;font-size:12px;cursor:pointer;user-select:none}
.legend{position:absolute;left:10px;bottom:10px;font-size:12px;color:var(--muted);background:var(--panel);padding:4px 8px;border-radius:6px;border:1px solid var(--line)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 8px;vertical-align:middle}
aside{display:flex;flex-direction:column;gap:12px;min-width:0}
.card{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--c);border-radius:8px;padding:10px 12px;cursor:pointer}
.card h3{margin:0 0 4px;font-size:15px}.tag{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--c);font-weight:600}
.card p{margin:4px 0}.imgs{display:flex;gap:6px;margin-top:6px}.imgs figure{margin:0;flex:1;min-width:0}
.imgs img{width:100%;border-radius:4px;display:block}.imgs figcaption{font-size:11px;color:var(--muted)}
.muted{color:var(--muted);font-size:12px}
section.box{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
section.box h2{font-size:14px;margin:0 0 6px}.maps{display:flex;gap:8px}.maps figure{margin:0;flex:1;min-width:0}.maps img{width:100%;image-rendering:pixelated;border-radius:4px}
table{border-collapse:collapse;width:100%;font-size:12px}td{padding:2px 0}td:last-child{text-align:right;font-variant-numeric:tabular-nums}
</style></head><body>
<header><h1>What changed in the room?</h1><p id="sub"></p></header>
<main>
<div id="view"><div class="toggles" id="toggles"></div>
<div class="legend"><span class="dot" style="background:#e63c32"></span>removed<span class="dot" style="background:#28c85a"></span>added<span class="dot" style="background:#3c78ff"></span>moved<span class="dot" style="background:#999aa5"></span>not seen in 2nd scan<span class="dot" style="background:#ff8c00"></span>naive diff</div></div>
<aside id="side"></aside>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const D = /*DATA*/;
const dec = (s, T) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };
const view = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(window.devicePixelRatio);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 1, 0.02, 100);
camera.up.set(...D.up);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
function bg(){ scene.background = new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue('--canvas').trim()); }
bg();
const groups = {};
function addLayer(name, size){
  const L = D.layers[name]; if (!L || !L.n) return;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(dec(L.p, Float32Array), 3));
  g.setAttribute('color', new THREE.BufferAttribute(dec(L.c, Uint8Array), 3, true));
  const pts = new THREE.Points(g, new THREE.PointsMaterial({size, vertexColors: true}));
  scene.add(pts); groups[name] = pts; return pts;
}
const order = [['after','2nd scan',0.012,true],['before','1st scan',0.012,false],['removed','removed',0.02,true],['added','added',0.02,true],['unknown','not seen',0.016,false],['naive','naive diff',0.016,false]];
const tg = document.getElementById('toggles');
for (const [k, lab, sz, on] of order){
  const p = addLayer(k, sz); if (!p) continue; p.visible = on;
  const l = document.createElement('label'); l.innerHTML = `<input type="checkbox" ${on?'checked':''}> ${lab}`;
  l.querySelector('input').onchange = e => { p.visible = e.target.checked; }; tg.appendChild(l);
}
const boxGroup = new THREE.Group(); scene.add(boxGroup);
for (const c of D.changes){
  const col = new THREE.Color(c.colour);
  for (const b of c.boxes){
    const s = b.max.map((v,i)=>v-b.min[i]), ctr = b.max.map((v,i)=>(v+b.min[i])/2);
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(...s)), new THREE.LineBasicMaterial({color: col, transparent: true, opacity: b.side==='before' && c.status==='moved' ? 0.45 : 1}));
    e.position.set(...ctr); boxGroup.add(e);
  }
  if (c.status === 'moved'){
    const a = new THREE.Vector3(...c.boxes[0].center), b = new THREE.Vector3(...c.boxes[1].center), d = b.clone().sub(a);
    boxGroup.add(new THREE.ArrowHelper(d.clone().normalize(), a, d.length(), col, Math.min(0.15, d.length()*0.3), 0.06));
  }
}
// initial camera: fit the 2nd scan
const box = new THREE.Box3().setFromObject(groups.after || groups.before);
const ctr = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3()).length();
const upv = new THREE.Vector3(...D.up);
camera.position.copy(ctr).add(new THREE.Vector3(size*0.35, 0, size*0.35)).add(upv.clone().multiplyScalar(size*0.45));
controls.target.copy(ctr); controls.update();
function focus(c){
  const p = new THREE.Vector3(...c.boxes[c.boxes.length-1].center);
  const dir = camera.position.clone().sub(controls.target).normalize();
  controls.target.copy(p); camera.position.copy(p).add(dir.multiplyScalar(2.2)); controls.update();
}
function resize(){ const w = view.clientWidth, h = view.clientHeight; renderer.setSize(w, h); camera.aspect = w/h; camera.updateProjectionMatrix(); }
window.addEventListener('resize', resize); resize();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', bg);
(function loop(){ requestAnimationFrame(loop); renderer.render(scene, camera); })();

// side panel
const side = document.getElementById('side');
const n = s => D.changes.filter(c => c.status === s).length;
document.getElementById('sub').textContent = `${D.changes.length} changes: ${n('moved')} moved, ${n('added')} added, ${n('removed')} removed. Click a card to fly to it.`;
const esc = s => String(s).replace(/[&<>"]/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]));
for (const c of D.changes){
  const el = document.createElement('div'); el.className = 'card'; el.style.setProperty('--c', c.colour);
  const imgs = ['before','after'].filter(k => c.imgs[k]).map(k => `<figure><img src="${c.imgs[k]}" alt="${k}"><figcaption>${k}</figcaption></figure>`).join('');
  el.innerHTML = `<span class="tag">${esc(c.status)}</span><h3>#${c.id} ${esc(c.label)}</h3><p>${esc(c.description)}</p><div class="muted">confidence ${c.confidence}</div><div class="imgs">${imgs}</div>`;
  el.onclick = () => focus(c); side.appendChild(el);
}
if (D.nav){
  const s = document.createElement('section'); s.className = 'box';
  s.innerHTML = `<h2>Robot navigation map</h2><div class="maps"><figure><img src="${D.nav.before}"><figcaption class="muted">before: path ${D.nav.path_before_m} m</figcaption></figure><figure><img src="${D.nav.after}"><figcaption class="muted">after: path ${D.nav.path_after_m} m</figcaption></figure></div>`;
  side.appendChild(s);
}
const cv = D.coverage, al = D.alignment, s2 = document.createElement('section'); s2.className = 'box';
const pct = v => (100*v).toFixed(1) + '%';
s2.innerHTML = `<h2>Why visibility matters</h2><table>
<tr><td>1st-scan surface flagged by naive distance diff</td><td>${pct(cv.naive_changed_before)}</td></tr>
<tr><td>1st-scan surface flagged by visibility test</td><td>${pct(cv.visibility_changed_before)}</td></tr>
<tr><td>1st-scan surface the 2nd scan never saw (unknown)</td><td>${pct(cv.before_surface_unknown_in_after)}</td></tr>
<tr><td>Alignment (${esc(al.method)})</td><td>rmse ${(al.icp_rmse_m*100).toFixed(1)} cm</td></tr></table>`;
side.appendChild(s2);
</script></body></html>
"""
