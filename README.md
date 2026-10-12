# Freespace — what changed in the room?

![CI](https://github.com/shlokkvaishnav/freespace/actions/workflows/ci.yml/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

Freespace compares two phone recordings of the same room. It finds the objects that
were **moved**, **added** or **removed**, shows them in 3D, and describes each one in
a sentence. Example: *"The red chair moved 1.26 m to the right and turned 40°. It is now on the floor."*

It also tells you which parts of the room it **cannot judge**, because the second
recording did not see them, and **where to look next** to finish the check. A robot must
not think that an object is gone when it only did not look.

![Freespace report: the demo room with moved, added, removed and modified objects, and three "look here next" spots in yellow](assets/report.jpg)

**Live demo report (no install):** https://shlokkvaishnav.github.io/freespace/ — a synthetic room with every kind of change (moved and turned, slid 8 cm, door swung, blanket folded, removed, added, and an area the second walk never saw). A GitHub Action rebuilds it from the current code on every push. Real-room results follow.

<!-- RESULTS: add GIF / screenshots of report.html here after recording real scenes -->

## Quick start

Requirements: macOS (Apple Silicon) or Linux, Python 3.11. A GPU is not necessary.
On macOS, Open3D needs `libusb`: `brew install libusb`. On Ubuntu: `sudo apt-get install libegl1 libgl1 libgomp1`.

```bash
git clone https://github.com/shlokkvaishnav/freespace && cd freespace
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # optional: pip install -e ".[lightglue]" for better matching

freespace demo                      # renders a synthetic room pair, runs, scores, opens the report
```

`freespace demo` needs no phone. It makes a synthetic room with ground truth (one moved chair, one removed
box, one added suitcase, and one lamp that the second walk never sees), runs the full pipeline,
prints precision / recall, and opens `out/demo/report.html`. The steps one by one:

```bash
freespace synth data/synth
freespace run --before data/synth/before --after data/synth/after --out out/synth
open out/synth/report.html          # interactive 3D report
freespace view out/synth            # same result in the Rerun viewer
python eval/run_eval.py out/synth/changes.json data/synth/gt.json
```

> If your project folder syncs with iCloud (for example `~/Documents`), put the venv outside it
> (`python3.11 -m venv ~/.venvs/freespace`). iCloud can mark files as hidden, and Python then
> skips the `.pth` file of the editable install (`ModuleNotFoundError: freespace`).

### Your own room

1. Install **Stray Scanner** (free) on an iPhone/iPad with LiDAR.
2. Record the room for 30–60 s. Move slowly. Point the phone at the floor and at the furniture.
   Keep the lights and curtains the same for both walks, and record them close in time: the colour check removes
   one global brightness difference, not a lamp switched on or a shadow in a new place.
3. Move, add or remove some objects.
4. Record again. Use a different path. Make sure that the two recordings overlap.
5. Copy the two folders to the computer (Files app → share, or AirDrop) and run:

```bash
freespace run --before data/room/before --after data/room/after --out out/room
```

Labels are local by default (CLIP, no API key). For richer sentences with landmarks:

```bash
export ANTHROPIC_API_KEY=...
freespace run ... --vlm claude
```

### Score a run on your own room

Real rooms have no 3D ground truth, so a person reviews the detections:

1. While you record, write `data/room/gt.json` — one line per real change, with tape-measured moves:
   ```json
   {"changes": [{"name": "chair", "status": "moved", "distance": 0.9, "turn": 30},
                {"name": "backpack", "status": "added"},
                {"name": "box", "status": "removed"}]}
   ```
2. Run Freespace and the naive baseline, then review each one (crops are in `out/<run>/crops/`):
   ```bash
   freespace run --before data/room/before --after data/room/after --out out/room
   freespace run --before data/room/before --after data/room/after --out out/room_naive --baseline naive
   freespace label out/room --gt data/room/gt.json         # answer: real-change number, w (wrong), u (unseen area called removed)
   freespace label out/room_naive --gt data/room/gt.json
   python eval/run_eval.py out/room/changes.json data/room/gt.json --review out/room/review.json
   ```
   The score has precision, recall, median move and turn error, and how often an unseen area was called "removed".

## Outputs

| File | Content |
|---|---|
| `report.html` | One-file interactive report: 3D point cloud with change layers, change cards with before/after crops, robot navigation map |
| `scene.rrd` | Rerun recording: both scans, change layers, 3D boxes, move arrows, camera paths, occupancy maps |
| `changes.json` | Scene-graph diff: id, status, label, description, 3D centre and box, move transform, confidence, measured facts |
| `clouds.npz` | Aligned surface points with their per-point state, for your own analysis |

## How it works

```
recording A ─┐                                 ┌─ visibility test A→B ─ removed surface ─┐
             ├─ TSDF fusion ─ 4-DoF alignment ─┤                                         ├─ clusters ─ match = moved ─ labels ─ report
recording B ─┘                                 └─ visibility test B→A ─ added surface ───┘
```

1. **Load.** Stray Scanner gives RGB, LiDAR depth, confidence and ARKit camera poses for each frame. I keep a keyframe every 5 cm or 5°.
2. **Fuse.** Each recording becomes a TSDF surface (Open3D, 2 cm voxels).
3. **Align.** Two ARKit sessions share gravity but not heading or origin. So the unknown is 4 DoF (yaw + translation), not 6.
   CLIP image embeddings pick image pairs that look alike. SIFT matches pixels (LightGlue is optional). LiDAR depth lifts the matches to 3D.
   A 2-point RANSAC finds yaw + translation. The changed objects are outliers here, so RANSAC ignores them.
   Robust point-to-plane ICP (Tukey loss) refines the result.
4. **Visibility test (the core).** For each surface point of recording A, I ask every depth frame of recording B:
   *did your sensor ray stop at this point, stop before it, or go through it?*
   - The ray goes **through** the point → B saw empty space where A had a surface → evidence of change.
   - The ray stops **at** the point → the surface is still there.
   - The ray stops **before** the point → occluded → no evidence.
   - No frame of B looks at the point → **unknown**.

   A point is CHANGED when most observations see through it, UNCHANGED when most see it, and UNKNOWN when fewer than 3 frames observe it. I do the same from B to A to find added surfaces.
5. **Objects.** DBSCAN groups changed points. Each group then **grows into the whole object** in its own scan:
   cut away the horizontal support under it (floor, table top), keep the surface points that connect to the
   changed points, grow the box, repeat. So a box on a table does not join the table, and the bottom of an object
   (which touches the support, so the ray test cannot call it changed) is part of the object again.
   Groups that grow into the same object are merged. For each object I find the frames that show it best.
   A removed object and an added object are one **moved** object only if their colour and size match (two simple gates),
   and then the closest pair by CLIP appearance + size + colour wins.
   ICP between the two whole objects gives the move distance and the turn angle. A "move" smaller than
   max(2 cm, 3 × alignment error) with a turn under 5° is noise and is dropped.
   - **Symmetric shapes.** A square box fits equally well at 40° and 130°. Every turn that fits the shape about as
     well as the best is a candidate; the surface texture picks one. If the texture cannot decide, the report says
     so: "turned 70° (or 109°: its shape looks the same and the texture cannot tell)".
   - **Kind of motion, from geometry only.** The rotation axis of the fitted move tells what happened, without
     knowing what the object is: axis inside the object → *turned in place*; axis at its edge → *swung about an
     edge* (a door); no turn and a short distance → *slid* (a drawer, a pushed box); otherwise → *moved*.
   - **Modified.** A pair whose colour and size match but that no rigid move fits, or a removed and an added
     region in the same place with the same colour, is *modified* (a folded blanket, a bag that was filled).
6. **Colour check (flat things).** A book or a magazine is too thin for the depth test. So where *both* scans saw
   the same surface (the same rule as above), I compare the fused colour of each point with the nearest point of
   the other scan that faces the same way. One gain per colour channel removes the global exposure difference. I
   compare chromaticity (hue), skip points near edges (their colour mixes two surfaces), and flag differences above
   median + 6 × MAD. The height decides the kind: higher → flat object added, lower → removed, same height →
   "surface changed" (only if the colour name changes, e.g. white → blue).
7. **Words.** CLIP names the object from an open lexicon of ~1,200 everyday object names (LVIS). If CLIP is not sure,
   the name is "unknown object" with its colour and size. The sentence comes from a template with the measured numbers.
   With `--vlm claude`, a vision model writes the sentence, but it receives the numbers as fixed facts.
   So the geometry and the words always agree.
8. **Robot view.** I project both scans to 2D occupancy grids at robot body height and plan the same A* path in each. The report shows how a new obstacle changes the path.
9. **Where to look next.** The surface the second walk never saw stays UNKNOWN. I place candidate cameras at 1.4 m
   on the free floor, each looking at one unseen region, and count the unseen points each one would really see
   (a ray cast against both meshes, so walls and furniture block the view). Unseen floor and walls count less than
   unseen objects. The best 3 spots (greedy, each must add new points) appear in yellow in the report.
   Every change also shows its evidence, e.g. *"median 24 of 24 views of the second walk saw through the old surface"*.

## Design choices and tradeoffs

- **Class-agnostic by design.** Detection, grouping, move matching, heights and the navigation map never use object
  classes. Anything that takes up space and was seen through (or is now in the way) is a change, the same rule a robot's
  obstacle map uses. Names are a separate layer, used only for the words. A test checks that an object with a shape
  and colour outside the lexicon is still found and placed.
- **One small model.** CLIP ViT-B/16 does image retrieval, crop matching and names. Everything that decides
  is classic geometry (TSDF, RANSAC, ICP, ray counts), so every decision can be checked by hand.

- **Why LiDAR + ARKit and not pure RGB?** The challenge is about change reasoning, not about reconstruction. LiDAR gives metric depth for each frame, and the visibility test needs per-frame depth. Pure RGB would need a learned depth/pose model (e.g. VGGT), which adds error that looks like change.
- **Why a 3-state answer?** A cloud-to-cloud distance marks every unseen region as "removed". That is a false alarm for a robot. The report shows this side by side: the *naive diff* layer against the *visibility* layer.
- **Why 4 DoF?** Gravity from the IMU is good. Fixing roll and pitch makes the RANSAC minimal set 2 points and removes a large class of wrong solutions in rooms with repeated structure.
- **Why numbers from geometry and names from a model?** A VLM can guess the wrong distance. Geometry cannot guess the name. Each part does what it is good at.
- **Local by default.** No API key, same output each run, and it can run on a robot. Claude is an option for richer language.

## Result: the ray test against a naive distance diff

The same pipeline with only the change test swapped (`--baseline naive`: a point is "changed" when no point of
the other scan is within 5 cm). Demo room, alignment estimated, 7 real changes:

| Change test | Detections | Precision | Recall |
|---|---|---|---|
| Naive distance diff | 14 | 0.36 | 0.71 |
| **Freespace ray test** | **7** | **1.00** | **1.00** |

The naive diff calls every area the second walk did not see "changed": walls, the back of the sofa, the floor
behind the table. Those false objects also break the move matching. The ray test calls them *unknown* instead.
Reproduce: `freespace demo`, then `freespace run --before data/demo/before --after data/demo/after --out out/naive --baseline naive`
and `python eval/run_eval.py out/naive/changes.json data/demo/gt.json`.

## Experiment: does a learned rule beat the hand rule?

The ray test decides with a hand rule: a point is CHANGED when at least 60% of its observations see through it.
To check this rule, I trained a logistic regression (4 readable weights, PyTorch) on 10 random synthetic room pairs
(`python -m freespace.train.gen_data data/train`, then `python -m freespace.train.fit_fusion data/train`).
Features per point: see-through ratio, log(1 + observations), mean depth residual of the hits. Labels come free from
the object boxes. The threshold is chosen for the best F1 on the training rooms; 4 other rooms are held out.

| | point F1 (held-out rooms) | false-positive rate | shoebox 8 cm move error |
|---|---|---|---|
| hand rule (default) | 0.808 | 1.39% | 0.05 m |
| learned weights `[-6.95, 6.73, 0.17, 1.95]` | 0.812 | 1.29% | 0.09 m |

The learned rule is not clearly better, so the simpler hand rule stays the default (`fusion.weights: []`).
The weights still teach something: the learned rule wants almost all rays to see through a point, and a hit with
a large depth residual counts as evidence of change (an object that moved a little).

## Where it fails

I tested every case below on purpose. These are the known failures and the honest answers it gives instead.

| Situation | What happens | Why |
|---|---|---|
| Object smaller than ~6 cm (a pen, keys) | Not reported | Filtered as noise; LiDAR depth noise is 1–2 cm and pixels are ~1 cm at 2 m |
| Move under ~5 cm | Can be missed or measured roughly | Close to the alignment and depth error (an 8 cm move is found with a 1 cm error) |
| Glass, mirrors, shiny black surfaces | Wrong or missing depth there | LiDAR does not return from them; Freespace cannot see through what the sensor cannot see |
| A lamp switched on, sunlight, a new shadow | Can look like "surface changed" | The colour check removes one global brightness change, not local light |
| Flat object with the same colour as its support (white paper on a white desk) | Not reported | Too thin for depth, no colour difference |
| Plain symmetric object (no texture) turned | Two or more turn angles, or "uncertain" | Shape and texture fit equally; it says so instead of guessing |
| Folded blanket, filled bag | "Modified", without detail | No rigid motion explains it; it does not model *how* it changed |
| Something moved where the second walk never looked | "Unknown" + a "look here next" spot | By design: no evidence, no claim |
| Two walks recorded at very different times of day | More false "surface changed" | Same as the light problem above |

All results so far are on synthetic rooms. Real-room results follow.

## Tests

```bash
pytest -q           # unit tests + full pipeline on a synthetic room (moved / removed / added / never-seen object)
```

## Repo layout

```
freespace/  io/stray.py  recon.py  align.py  change.py  objects.py  describe.py  costmap.py  viz.py  report.py  pipeline.py  cli.py  synth.py
eval/      run_eval.py  (precision / recall / move error against gt.json)
tests/     test_geometry.py  test_synthetic.py
```

## License

MIT. See [LICENSE](LICENSE).
