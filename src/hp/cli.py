"""Command-line interface: ``hp <command>``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

import hp
from hp.config import load_config

app = typer.Typer(add_completion=False, help="3D change detection between two recordings of a room.")
console = Console()


@app.command()
def run(
    video_a: Optional[str] = typer.Option(None, "--video-a", help="Before recording (mp4/mov)."),
    video_b: Optional[str] = typer.Option(None, "--video-b", help="After recording (mp4/mov)."),
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Synthetic dataset dir (has gt.json)."),
    out: Optional[str] = typer.Option(None, "--out", help="Output directory."),
    config: Optional[str] = typer.Option(None, "--config", help="YAML config path."),
    set_: list[str] = typer.Option(None, "--set", help="Config override, e.g. change.voxel_size=0.05."),
    align: str = typer.Option("auto", "--align", help="Alignment method: auto|identity|open3d|correspondences."),
    semantics: bool = typer.Option(True, "--semantics/--no-semantics", help="Run SAM 3.1 labelling."),
    evaluate: bool = typer.Option(True, "--eval/--no-eval", help="Score against gt.json when available."),
) -> None:
    """Compare two recordings (or a synthetic dataset) and report 3D changes."""
    overrides = list(set_ or [])
    if not semantics:
        overrides.append("semantics.enabled=false")
    cfg = load_config(config, overrides)

    from hp.pipeline import run as run_pipeline

    result = run_pipeline(
        video_a=video_a, video_b=video_b, dataset=dataset,
        cfg=cfg, out_dir=out, align_method=align,
    )
    cs = result.changeset

    table = Table(title=f"3D changes: {cs.src_name} -> {cs.dst_name}")
    table.add_column("id", justify="right")
    table.add_column("kind")
    table.add_column("label")
    table.add_column("centre (m)")
    table.add_column("size (cm)")
    table.add_column("conf", justify="right")
    for lc in cs.labeled or [None] * len(cs.clusters):
        c = lc.cluster if lc else None
        if c is None:
            continue
        size = c.extent * 100.0
        table.add_row(
            str(c.id), c.kind, lc.label,
            f"({c.centroid[0]:.2f}, {c.centroid[1]:.2f}, {c.centroid[2]:.2f})",
            f"{size[0]:.0f}x{size[1]:.0f}x{size[2]:.0f}",
            f"{lc.confidence:.2f}",
        )
    console.print(table)
    console.print(f"counts: {cs.counts()}")
    if result.outputs:
        console.print(f"outputs: {result.outputs}")

    if evaluate and dataset:
        gt_file = Path(dataset) / "gt.json"
        if gt_file.exists():
            from hp.eval.metrics import evaluate_changes

            gt = json.loads(gt_file.read_text())["changes"]
            ev = evaluate_changes(cs.clusters, gt, tol_m=0.6, allow_moved_as_two=True)
            console.print(f"[bold]evaluation vs ground truth:[/bold] {ev}")
    else:
        console.print(f"[green]done.[/green] wrote outputs to {result.out_dir}")


@app.command()
def synthesize(
    out: str = typer.Option("data/raw/room01", "--out", help="Output directory."),
    name: str = typer.Option("room01", "--name"),
    frames: int = typer.Option(12, "--frames", help="Frames per pass."),
    height: int = typer.Option(220, "--height"),
    width: int = typer.Option(280, "--width"),
    noise: float = typer.Option(0.004, "--noise", help="Depth noise std (m)."),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Generate a synthetic before/after room dataset with exact ground truth."""
    from hp.synthetic import generate_dataset

    path = generate_dataset(
        out, name=name, n_frames=frames, height=height, width=width,
        depth_noise_std=noise, seed=seed,
    )
    console.print(f"[green]wrote synthetic dataset[/green] -> {path}")


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"humanoid-perception {hp.__version__}")


def main() -> None:  # pragma: no cover - entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
