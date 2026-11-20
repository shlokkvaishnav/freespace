# Plan: real-data evaluation on 3RScan

**Status: plan only, not run.** 3RScan is not freely downloadable. The owner must fill in the
access form (terms of use) on the 3RScan project page and receive the download script first.
No code for it is in this repo yet: I could not check the file formats without the data, so an
adapter written now would be built on guesses. Items marked *verify* must be checked on the
first downloaded scan before anything else is written.

## Why 3RScan

3RScan (Wald et al., "RIO: 3D Object Instance Re-Localization in Changing Indoor Environments",
ICCV 2019) has rooms scanned several times over days to months, with RGB-D frames and camera poses
for each scan and annotations of which object instances moved (with a rigid transform), changed
shape, or were removed between the reference scan and each rescan. That is the same question
Freespace answers, on real sensor data, with someone else's ground truth.

Differences from Freespace's input (each needs a check, not an assumption):

| | Stray Scanner (Freespace) | 3RScan |
|---|---|---|
| Device | iPhone LiDAR + ARKit | Google Tango phone (depth camera + VIO) |
| Depth confidence | 0/1/2 map | none (*verify*): use all valid depth, `min_confidence: 0` |
| Up axis | +Y | likely +Z (*verify*), set `io.up_axis` |
| Pose convention | OpenCV camera-to-world | camera-to-world per frame (*verify* axes with one scan) |
| Frames | `rgb.mp4`, `depth/*.png` | per-frame colour, depth and pose files (*verify* names, depth scale) |

## Pairs

- Reference scan vs each rescan, from the public train and validation splits only.
- Keep pairs whose annotation lists at least one rigid move or removal, and whose rescan was
  captured with the same device type.
- Target about 50 pairs. Choose them by a fixed rule (for example the first 50 such pairs in the
  sorted list of reference ids), written down before running, so the pair list is not tuned.
- The rescan-to-reference transform in the annotations is used only for scoring the alignment,
  never as input (Freespace estimates its own).

## How 3RScan's annotations map to Freespace output

3RScan lists changes per instance (*verify field names* in the metadata JSON; my understanding is
"rigid" moves with a transform, "nonrigid" changes, and "removed" instances):

| 3RScan | Freespace `status` | Ground-truth point for matching |
|---|---|---|
| rigid instance with a non-trivial transform (> 2 cm or > 5°, Freespace's own noise floor) | `moved` | instance centroid in the rescan |
| rigid instance with a near-identity transform | not a change (excluded from GT and from false-change counting) | — |
| non-rigid instance | `modified` | instance centroid in the rescan |
| removed instance | `removed` | instance centroid in the reference |
| instance present only in the rescan (if annotated; *verify*) | `added` | instance centroid in the rescan |

Centroids come from the annotated instance segmentation of each scan's mesh. They are converted to
the format of `data/*/gt.json` (`frame: after`, centres in the rescan frame) so that
`eval/run_eval.py` scores them unchanged. Matching radius 0.4 m as for the synthetic rooms; also
report 0.2 m, because 3RScan rooms are more cluttered.

Small changes below Freespace's documented limits (objects < 6 cm, moves < 5 cm) are reported as
a separate group and not counted against recall in the headline number.

## Metrics (fixed before the run)

1. **Object precision and recall** per pair and pooled, with a bootstrap 95% interval over pairs.
2. **Move error**: distance and turn error for matched `moved` objects (median, IQR).
3. **False-change rate**: area called CHANGED that lies more than 10 cm from every annotated changed
   instance, per m² of observed (not UNKNOWN) unchanged surface; both directions pooled; in cm²/m².
   Same definition as the drift study (`docs/drift-eval.md`), but the changed region is the
   annotated instance points instead of a synthetic box.
4. **Unseen-as-removed**: removed calls on surface the rescan never observed (should be 0 by design).
5. **Alignment error** against the annotated scan-to-scan transform.

Every metric is reported for `--baseline freespace` and `--baseline naive` on the same pairs.

## Expected failure modes to watch

- Tango depth is noisier and lower resolution than iPhone LiDAR: the 3 cm + 1% depth tolerance may
  be too tight. Do not tune it on the evaluation pairs; tune on 5 separate pairs, then freeze.
- Months between scans: lighting changes will trigger the colour check. Report results with and
  without `appearance.enabled`.
- Many small changes (cushions, papers) that are annotated but below Freespace's size limit.

## Work, once access is granted

1. Download 2-3 scans, check every *verify* item above, write the result here.
2. Write `freespace/io/rscan.py` (frames -> `Scan`) and `eval/rscan_gt.py` (annotations -> `gt.json`),
   each with a unit test on one real scan.
3. Freeze the pair list, the config and this metric list in a commit before the full run.
4. Run (CPU; expect tens of GB of data and a few CPU hours), save `eval/results/3rscan.json`,
   and report in the README whatever comes out.
