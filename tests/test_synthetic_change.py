"""End-to-end test: synthetic before/after room -> detection scores well vs GT."""

import json

import numpy as np
import pytest

from hp.config import load_config
from hp.pipeline import compare_scenes
from hp.recon.gt_recon import load_synthetic_pair
from hp.synthetic import generate_dataset


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("room")
    generate_dataset(out, n_frames=12, height=200, width=260, depth_noise_std=0.004, seed=7)
    return out


def test_synthetic_change_detection_scores_high(dataset):
    scene_a, scene_b = load_synthetic_pair(dataset)
    cfg = load_config(overrides=["semantics.enabled=false"])

    changeset, grid_a, grid_b, _, _ = compare_scenes(scene_a, scene_b, cfg, align_method="identity")

    assert grid_a.dims == grid_b.dims
    assert grid_a.voxel_size == grid_b.voxel_size
    kinds = changeset.counts()
    assert sum(kinds.values()) >= 3, f"expected >=3 changes, got {kinds}"

    gt = json.loads((dataset / "gt.json").read_text())["changes"]
    from hp.eval.metrics import evaluate_changes

    ev = evaluate_changes(changeset.clusters, gt, tol_m=0.6, allow_moved_as_two=True)
    assert ev.recall >= 0.66, f"recall too low: {ev}"
    assert ev.f1 >= 0.6, f"f1 too low: {ev}"


def test_detects_the_right_kinds(dataset):
    scene_a, scene_b = load_synthetic_pair(dataset)
    cfg = load_config(overrides=["semantics.enabled=false"])
    changeset, *_ = compare_scenes(scene_a, scene_b, cfg, align_method="identity")
    kinds = {c.kind for c in changeset.clusters}
    # the scenario moves a chair, adds a box and removes a plant
    assert "added" in kinds
    assert "removed" in kinds


def test_labeled_changes_have_descriptions(dataset):
    scene_a, scene_b = load_synthetic_pair(dataset)
    cfg = load_config(overrides=["semantics.enabled=false"])
    changeset, *_ = compare_scenes(scene_a, scene_b, cfg, align_method="identity")
    assert changeset.labeled
    for lc in changeset.labeled:
        assert lc.description
        assert lc.description[0].isupper()


def test_alignment_identity_is_noop(dataset):
    from hp.align import Sim3, align_scene

    scene_a, _ = load_synthetic_pair(dataset)
    aligned = align_scene(scene_a, Sim3.identity())
    assert np.allclose(aligned.frames[0].pose, scene_a.frames[0].pose)
