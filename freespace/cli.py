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
):
    """Detect, label and visualise the changes between two recordings."""
    from . import pipeline, report, viz

    cfg = load_config(config)
    vlm = vlm or cfg.describe.vlm
    T_init = None
    if gt_align:
        T_init = np.array(json.loads((before.parent / "gt.json").read_text())["T_after_before"])
    res = pipeline.run(before, after, cfg, vlm=vlm, cache_dir=out / "cache", T_init=T_init)
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
    run(before=data / "before", after=data / "after", out=out, vlm=None, config=None, gt_align=False)
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "eval" / "run_eval.py"),
                    str(out / "changes.json"), str(data / "gt.json")], check=False)
    if not no_open:
        webbrowser.open((out / "report.html").resolve().as_uri())


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
