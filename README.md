# humanoid-perception

Compare two phone recordings of the same small indoor space and identify what
physically changed in 3D — furniture that moved, objects that were added, or
objects that were removed — while distinguishing real change from differences in
camera viewpoint, occlusion, or incomplete observation.

```
before.mp4 ──► reconstruct ─┐
                            ├─► Sim(3) align (static geometry) ─► visibility-aware voxel diff ─► added/removed/moved + labels + report
after.mp4  ──► reconstruct ─┘
```

## What it does

1. **Reconstructs** each recording into a dense 3D model with camera poses
   (feed-forward [VGGT](https://github.com/facebookresearch/vggt) on a Colab GPU;
   COLMAP as a CPU fallback).
2. **Aligns** the two scans with a 7-DoF **Sim(3)** — monocular reconstruction has
   an unknown scale — using only *static* geometry, iterating so the changes
   themselves cannot drag the registration.
3. **Compares** them in a shared voxel grid, labelling every voxel
   `occupied` / `observed-free` / `unknown`.
4. **Reports** added / removed / moved objects, labels them with **SAM 3.1**
   (open-vocabulary, text-prompted segmentation) and writes a natural-language
   report plus an interactive 3D figure.

## Why the results are trustworthy

| Failure mode | How it is handled |
| --- | --- |
| Occlusion / unobserved regions | A voxel that was never *observed* in a scan stays `unknown` and is **never** reported as a change (`hp/change/detect.py`). |
| Depth noise / registration error | A `surface_band_m` tolerance suppresses differences within a few voxels of a surface in either scan. |
| Scale drift between the two takes | Alignment is a full similarity transform (Umeyama + RANSAC), not rigid SE(3). |
| Changes corrupting the alignment | Alignment is driven by static geometry only and re-run after an initial detection pass. |
| A removed object next to an added one | Moved objects are paired by proximity **and** volume ratio **and** sampled colour, so unrelated nearby changes are not merged. |

## Quick start — Google Colab (recommended)

Open [`notebooks/hp_colab.ipynb`](notebooks/hp_colab.ipynb) in Colab, set your
`HF_TOKEN` (see below), pick a runtime with a GPU (T4 is enough), and run the
cells. The notebook installs everything, and can run on either your two phone
clips or a bundled synthetic room with **zero external data**.

You need a Hugging Face token with access to the gated
[`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1) checkpoint
(request access, then put `HF_TOKEN=hf_...` in a local `.env` — see
`.env.example`).

## Quick start — local, synthetic (no GPU, seconds)

```bash
uv venv --python 3.11 .venv && . .venv/bin/activate
uv pip install -e ".[dev]"

hp synthesize --out data/raw/room01          # procedural room + exact ground truth
hp run --dataset data/raw/room01             # detect changes, write report + figure
```

Expected result: the pipeline finds the moved chair, the added box and the
removed plant, and scores **F1 = 1.00** against the synthetic ground truth
(`--eval`, on by default for datasets).

## Quick start — your own phone videos

1. Record two clips following [`docs/capture_protocol.md`](docs/capture_protocol.md)
   (smooth walk around the room, both passes covering everything, fixed lighting).
2. On Colab, upload/Drive-mount them, then:

```bash
hp run --video-a before.mp4 --video-b after.mp4 --out outputs/room1
```

Outputs land in `outputs/room1/`: `report.md`, `changes.json`, `changes.html`.

## CLI

```text
hp run         [--video-a F --video-b F | --dataset DIR] [--out DIR]
               [--config YAML] [--set key=value ...] [--align auto|identity|open3d|correspondences]
               [--semantics/--no-semantics] [--eval/--no-eval]
hp synthesize  [--out DIR] [--frames N] [--noise M] [--seed S]
hp version
```

Everything is configurable via [`configs/default.yaml`](configs/default.yaml) or
`--set`, e.g. `--set change.voxel_size=0.05 --set semantics.enabled=false`.

## Architecture

```
src/hp/
  io/          video decode (ffmpeg/OpenCV) + sharpness-based keyframe selection
  recon/       reconstruction backends: gt (synthetic) | vggt (Colab GPU) | colmap (CPU)
  align/       Sim(3): Umeyama, RANSAC, cross-video registration
  change/      voxel fusion (free-space carving), visibility-aware diff, clustering,
               moved-object pairing, appearance sampling
  semantics/   SAM 3.1 open-vocabulary labelling + grounded captions
  viz/         Plotly (Colab), Open3D (local), report writer
  eval/        matching of detected changes against ground truth
  synthetic/   procedural room, analytic RGB-D renderer, before/after scenarios
  pipeline.py  orchestration      cli.py  typer CLI
tests/         unit + end-to-end tests (CPU-only, no model weights)
notebooks/     hp_colab.ipynb
docs/          capture_protocol.md
```

## Models & dependencies

- **VGGT** (`facebook/VGGT-1B`) — feed-forward multi-view reconstruction; runs on
  the Colab T4 in chunked passes (`recon.frames_chunk_size`).
- **SAM 3.1** (`facebook/sam3.1`) — text-prompted concept segmentation for labels,
  object-level masks, and video tracking. ~3.4 GB, ~4 GB VRAM.
- **COLMAP** (`pycolmap`) — classic SfM fallback, CPU.
- Core deps are light (numpy, scipy, OpenCV, Plotly); heavy deps live in the
  `recon`, `semantics`, `colmap` and `viz` extras and are only imported on demand.

## Testing

```bash
python -m pytest -q          # CPU-only, no GPU, no model weights
```

The end-to-end test generates a synthetic before/after room and asserts the
detected changes match the ground truth.
