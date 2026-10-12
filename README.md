# Freespace — what changed in the room?

Freespace compares two phone recordings of the same room. It finds the objects that
were **moved**, **added** or **removed**, shows them in 3D, and describes each one in
a sentence. Example: *"The red chair moved 1.26 m to the right and turned 40°. It is now on the floor."*

It also tells you which parts of the room it **cannot judge**, because the second
recording did not see them. A robot must not think that an object is gone when it
only did not look.

**Live demo report (no install):** https://shlokkvaishnav.github.io/freespace/ — synthetic room; real-room results follow.

<!-- RESULTS: add GIF / screenshots of report.html here after recording real scenes -->

## Quick start

Requirements: macOS (Apple Silicon) or Linux, Python 3.11. A GPU is not necessary.
On macOS, Open3D needs `libusb`: `brew install libusb`.

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
   DINOv2 picks image pairs that look alike. LightGlue (or SIFT) matches pixels. LiDAR depth lifts the matches to 3D.
   A 2-point RANSAC finds yaw + translation. The changed objects are outliers here, so RANSAC ignores them.
   Robust point-to-plane ICP (Tukey loss) refines the result.
4. **Visibility test (the core).** For each surface point of recording A, I ask every depth frame of recording B:
   *did your sensor ray stop at this point, stop before it, or go through it?*
   - The ray goes **through** the point → B saw empty space where A had a surface → evidence of change.
   - The ray stops **at** the point → the surface is still there.
   - The ray stops **before** the point → occluded → no evidence.
   - No frame of B looks at the point → **unknown**.

   A point is CHANGED when most observations see through it, UNCHANGED when most see it, and UNKNOWN when fewer than 3 frames observe it. I do the same from B to A to find added surfaces.
5. **Objects.** DBSCAN groups changed points into objects. For each object I find the frames that show it best.
   A removed object and an added object with similar DINOv2 appearance, size and colour are one **moved** object.
   ICP between the two point sets gives the move distance and the turn angle.
6. **Words.** CLIP names the object from a list of indoor words. The sentence comes from a template with the measured numbers.
   With `--vlm claude`, a vision model writes the sentence, but it receives the numbers as fixed facts.
   So the geometry and the words always agree.
7. **Robot view.** I project both scans to 2D occupancy grids at robot body height and plan the same A* path in each. The report shows how a new obstacle changes the path.

## Design choices and tradeoffs

- **Why LiDAR + ARKit and not pure RGB?** The challenge is about change reasoning, not about reconstruction. LiDAR gives metric depth for each frame, and the visibility test needs per-frame depth. Pure RGB would need a learned depth/pose model (e.g. VGGT), which adds error that looks like change.
- **Why a 3-state answer?** A cloud-to-cloud distance marks every unseen region as "removed". That is a false alarm for a robot. The report shows this side by side: the *naive diff* layer against the *visibility* layer.
- **Why 4 DoF?** Gravity from the IMU is good. Fixing roll and pitch makes the RANSAC minimal set 2 points and removes a large class of wrong solutions in rooms with repeated structure.
- **Why numbers from geometry and names from a model?** A VLM can guess the wrong distance. Geometry cannot guess the name. Each part does what it is good at.
- **Local by default.** No API key, same output each run, and it can run on a robot. Claude is an option for richer language.

## Limits

- Rigid moves only. A folded blanket or an opened door is "removed + added".
- Objects smaller than about 6 cm are filtered out (LiDAR noise is about 1–2 cm).
- Glass, mirrors and black surfaces give bad LiDAR depth.
- Symmetric objects (a box) can give a turn angle that is wrong by 90° or 180°.
- Objects moved only a few centimetres are not reliable.

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
