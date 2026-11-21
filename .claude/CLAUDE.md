# freespace: notes for the agent

Finds 3D changes between two phone scans of a room (RGB-D from an iPhone LiDAR scanner): it aligns
the scans and reports a change only when depth rays prove it (free space where there was surface).
A synthetic room generator with ground truth drives the tests and the demo report on GitHub Pages.

## Start here
1. The SessionStart hook prints the branch and the next step from `.claude/STATE.md`.
2. `README.md`: what it does, results, commands. Code: `freespace/` (pipeline, `synth.py` generator,
   `features.py` CLIP). Evaluations: `eval/` with results in `eval/results/`; plans in `docs/`.

## Hard rules
- MUST write the plan of a new evaluation in `docs/` and commit it before the run; keep results even
  when they are bad. (Why: the README's numbers rest on this; see docs/drift-eval.md.)
- The headline 1.00/1.00 is 7 synthetic changes from one seed; never present synthetic numbers as
  real-room results. Real data (3RScan) needs the owner's access form: docs/3rscan-plan.md.
- New synthetic options stay off by default, so existing results do not change.
- Dependencies: install with `-c requirements.lock`; update the lock only with a passing test run.
- Git: work on a branch; merge and push only when the owner asks. Pages deploys only by hand.
  No co-author or AI attribution lines. Plain English.
- Never Read scan data, images, videos or model weights (`data/`, `out/`); use counts or short scripts.

## Commands (repo root)
- Setup: `pip install -c requirements.lock -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu`
  (add `.[claude]` for the optional Claude labels).
- Done check: `pytest tests/` (about 4 minutes on CPU; the same as CI).
- Demo: `freespace demo`.

## Definition of done
`pytest tests/` passes, every README number comes from a file in `eval/results/` written by a committed
script, and the demo still runs. One task at a time.

## End of session
Rewrite `.claude/STATE.md` (about 20 lines): branch, next step, open items. Replace it.
