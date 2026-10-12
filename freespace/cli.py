"""Command line: `freespace run`, `freespace view`, `freespace synth`."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import typer

from .config import load_config

app = typer.Typer(add_completion=False, help="What changed in the room? 3D change detection from two phone scans.")


@app.command()
def run(
    before: Path = typer.Option(..., help="Stray Scanner folder of the first recording"),
    after: Path = typer.Option(..., help="Stray Scanner folder of the second recording"),
    out: Path = typer.Option(Path("out/run"), help="Output folder"),
    vlm: str = typer.Option(None, help="Labeller: local (default, no key) or claude"),
    config: Path = typer.Option(None, help="Config YAML (default: config.yaml in the repo)"),
    gt_align: bool = typer.Option(False, help="Use T_after_before from gt.json next to the inputs (debug)"),
    baseline: str = typer.Option("freespace", help="freespace (ray test) or naive (cloud distance, for comparison)"),
):
    """Detect, label and visualise the changes between two recordings."""
    from . import pipeline, report, viz

    cfg = load_config(config)
    vlm = vlm or cfg.describe.vlm
    T_init = None
    if gt_align:
        T_init = np.array(json.loads((before.parent / "gt.json").read_text())["T_after_before"])
    res = pipeline.run(before, after, cfg, vlm=vlm, cache_dir=out / "cache", T_init=T_init, baseline=baseline)
    summary = pipeline.save(res, out)
    viz.log_rerun(res, out / "scene.rrd")
    report.write_html(res, summary, out / "report.html")
    typer.echo(f"\nWrote {out}/changes.json, {out}/report.html, {out}/scene.rrd")


@app.command()
def demo(out: Path = typer.Option(Path("out/demo"), help="Output folder"),
         no_open: bool = typer.Option(False, help="Do not open the report in a browser")):
    """One command for reviewers: make the example pair, run it, open the report.

    The demo room has every kind of change: a chair moved and turned, a shoebox slid 8 cm on the table,
    a door swung open, a blanket folded, a box removed, a suitcase and an odd ring object added,
    and a lamp that the second walk never sees (it must stay 'unknown')."""
    import webbrowser

    from .synth import make_pair

    data = Path("data/demo")
    if not (data / "gt.json").exists():
        make_pair(data, 120, small=True, hinge=True, folded=True, odd=True)
    run(before=data / "before", after=data / "after", out=out, vlm=None, config=None, gt_align=False,
        baseline="freespace")
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "eval" / "run_eval.py"),
                    str(out / "changes.json"), str(data / "gt.json")], check=False)
    if not no_open:
        webbrowser.open((out / "report.html").resolve().as_uri())


@app.command()
def label(
    out: Path = typer.Argument(..., help="Output folder of a run (has changes.json and crops/)"),
    gt: Path = typer.Option(..., help="Your list of real changes (gt.json, see README)"),
    answers: str = typer.Option(None, help="Non-interactive answers, e.g. '1:2,2:w,3:u' (change id : answer)"),
):
    """Review each detected change against your list of real changes; writes review.json.

    Answer with the number of the real change it matches, 'w' if it is wrong (nothing changed there),
    or 'u' if it reports something removed where the second walk never looked."""
    summary = json.loads((out / "changes.json").read_text())
    truth = json.loads(gt.read_text())["changes"]
    given = dict(a.split(":") for a in answers.split(",")) if answers else {}
    if not answers:
        typer.echo("Real changes:")
        for k, g in enumerate(truth, 1):
            typer.echo(f"  {k}. {g['status']} {g.get('label', g.get('name', ''))}"
                       + (f" ({g['distance']} m)" if "distance" in g else ""))
    review = {}
    for c in summary["changes"]:
        cid = str(c["id"])
        if cid in given:
            a = given[cid]
        else:
            crops = [str(p) for p in sorted((out / "crops").glob(f"{cid}_*.jpg"))]
            typer.echo(f"\n#{cid} [{c['status']}] {c['label']}: {c['description']}")
            typer.echo(f"   evidence: {c['facts'].get('evidence', '')}")
            typer.echo(f"   crops: {', '.join(crops) or 'none'}")
            a = typer.prompt("   which real change? (number / w = wrong / u = unseen area called removed)")
        review[cid] = int(a) if a.strip().isdigit() else a.strip()
    (out / "review.json").write_text(json.dumps({"gt": str(gt), "answers": review}, indent=2))
    typer.echo(f"Wrote {out}/review.json. Score it: python eval/run_eval.py {out}/changes.json {gt} --review {out}/review.json")


@app.command()
def view(out: Path = typer.Argument(Path("out/run"))):
    """Open the 3D result in the Rerun viewer."""
    subprocess.run([sys.executable, "-m", "rerun", str(out / "scene.rrd")], check=False)


@app.command()
def synth(out: Path = typer.Argument(Path("data/synth")), frames: int = 120):
    """Render a synthetic before/after room pair with ground truth (no phone needed)."""
    from .synth import make_pair

    make_pair(out, frames)
    typer.echo(f"Wrote {out}/before, {out}/after, {out}/gt.json")


if __name__ == "__main__":
    app()
