# Pose drift and mixed confidence (exploratory)

**Status: exploratory.** Written before the run. Nothing here is a pre-registered claim. It is a
stress test of the synthetic generator, not evidence about real rooms.

## Question

The 1.00 / 1.00 result in the README comes from 7 synthetic changes rendered with exact camera
poses and all depth at confidence 2. A real phone reports drifting poses (ARKit visual-inertial
odometry) and a confidence map with low and medium pixels. Drift is the most likely cause of false
changes: if a wall in walk B is drawn 3 cm away from where walk A put it, B's rays can "see through"
A's wall. How fast do precision, recall and the false-change rate fall as drift grows?

## Generator changes (off by default)

`freespace/synth.py`, new options of `make_pair` (defaults keep the old output exactly):

- `drift_deg`, `drift_m`: each walk's reported poses follow a random walk that starts at zero.
  At the last frame the yaw error has standard deviation `drift_deg`, roll and pitch a tenth of it
  (the IMU observes gravity), and the position error `drift_m` per horizontal axis (half vertically).
  Each camera turns about its own centre, so far surfaces move more than near ones.
  The images and depth still come from the true poses; only `odometry.csv` drifts.
- `mixed_conf`: a 0/1/2 confidence map. Depth edges get 0 and a "flying pixel" depth between the
  two sides (a fifth of them get 1, so some bad depth passes the default `min_confidence: 1`);
  depth beyond 3.5 m and a random 5% of pixels get 1 with 3x the depth noise; the rest stay 2.

**The magnitudes are assumptions.** Apple publishes no drift figure for ARKit. Visual-inertial
odometry is often described as drifting about 1% of the distance walked, which for a 3-5 m room walk
is 1-5 cm and about 0.5-2° of heading. I did not find a source I trust for ARKit specifically, so
the sweep goes past that range (to 4° / 6 cm) instead of picking one value. The confidence rules
follow how ARKit confidence maps look (low at edges and far away), not a published model.

## Settings

Scene: the `freespace demo` room (moved chair, 8 cm shoebox slide, swung door, folded blanket,
removed box, added suitcase, added ring sculpture; a lamp the second walk never sees), 120 frames
per walk. Alignment is estimated (not given), as for a real user. Default `config.yaml`.

| drift (yaw sd at end of walk) | position sd | confidence |
|---|---|---|
| 0 | 0 | all 2 (reference: the old generator) |
| 0 | 0 | mixed |
| 0.5° | 0.75 cm | mixed |
| 1° | 1.5 cm | mixed |
| 2° | 3 cm | mixed |
| 4° | 6 cm | mixed |

3 seeds each (seed changes depth noise, confidence and drift), 18 runs.
Command: `python -m eval.drift_eval --out eval/results/drift.json` (from the repo root).

## Metrics

- **Object precision / recall**: `eval/run_eval.py` (same status, centre within 0.4 m).
- **False-change rate**: surface called CHANGED that lies more than 10 cm outside the box of every
  object that truly changed, divided by all observed (not UNKNOWN) surface outside those boxes.
  Both directions (before→after and after→before) pooled. Reported as cm² of false change per m²
  of unchanged surface. Points come from the 2.5 cm downsampled cloud, so each stands for about the
  same area and the ratio does not depend on the exact area per point.
- The naive distance diff's rate on the same points, for scale (computed in the same run).
- Alignment error against the true session transform.

## What I expect (written before the run)

Little change up to 1°, because the ray test has a 3 cm + 1% depth tolerance and the alignment
absorbs the average offset. Some false changes on far walls by 2°, and false objects by 4°.
Whatever happens, the README will report the numbers as exploratory and synthetic.

## Outcome

Run on 2026-11-21, 18 runs, about 15 minutes in total on a laptop CPU (about 20 s of pipeline time per run). Raw numbers:
[`eval/results/drift.json`](../eval/results/drift.json). Mean over 3 seeds:

| drift (yaw / position) | confidence | precision | recall | false detections (3 runs) | false change, ray test (cm²/m²) | same points, naive diff | worst alignment error |
|---|---|---|---|---|---|---|---|
| 0 / 0 | all 2 | 0.95 | 0.95 | 1 | 0 | 80 | 1.1 cm |
| 0 / 0 | mixed | 0.76 | 0.76 | 5 | 18 | 139 | 2.5 cm |
| 0.5° / 0.75 cm | mixed | 0.73 | 0.76 | 6 | 38 | 139 | 3.9 cm |
| 1° / 1.5 cm | mixed | 0.63 | 0.71 | 9 | 51 | 153 | 5.8 cm |
| 2° / 3 cm | mixed | 0.57 | 0.67 | 11 | 136 | 214 | 10.3 cm |
| 4° / 6 cm | mixed | 0.43 | 0.62 | 17 | 912 | 1060 | 18.1 cm |

What this shows:

- **Even the old generator is not 1.00 / 1.00 on every seed.** Seeds 0 and 1 give 7 of 7; seed 2
  reports the 8 cm shoebox slide as "modified" instead of "moved". The README's 1.00 / 1.00 is seed 0.
- **Mixed confidence alone costs one change in every run**: the moved chair is reported as
  "modified" instead of "moved" (its rigid fit gets worse than the 5 cm limit), and the shoebox
  slide is not reported as moved in 2 of 3 runs. The surface test itself stays clean: 18 cm² of false change per m².
- **Drift hurts mostly through the 'moved' logic and the alignment.** Up to 1° / 1.5 cm (inside
  the range I assumed for a room walk) recall stays at 0.71-0.76 and the false-change rate is
  under 1% of the unchanged surface (51 cm²/m²). At 2° / 3 cm it is about 1.4% and false objects
  appear; at 4° / 6 cm about 9%, and the ray test is barely better than the naive diff on the same
  points.
- Most misses are status errors (moved → modified), not objects that were not found. The removed
  box, added suitcase, added ring and swung door were found in every run up to 2°; with more drift
  the small shoebox disappears entirely in some runs.
- The expectation written above ("little change up to 1°") was too optimistic for objects:
  precision falls from 0.76 to 0.63 by 1°. For the false-change rate it held.

Limits of this study: one synthetic room, 3 seeds per setting, drift and confidence models that are
my assumptions. It says which parts break first (the move test, then the alignment), not how
Freespace does on a real phone. The next step is real data (`docs/3rscan-plan.md`).
