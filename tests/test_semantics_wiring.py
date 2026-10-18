"""Wiring test: pipeline -> labeler -> captions (with a stubbed SAM 3.1)."""

import numpy as np
import pytest

from hp.config import load_config
from hp.pipeline import compare_scenes
from hp.recon.gt_recon import load_synthetic_pair
from hp.semantics import sam3 as sam3_mod
from hp.synthetic import generate_dataset
from hp.types import ChangeCluster, LabeledChange


class _StubLabeler:
    """Stands in for Sam3Labeler without torch/weights."""

    calls = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def label_changes(self, changes, scene_a, scene_b, top_k_frames=3):
        _StubLabeler.calls += 1
        out = []
        for c in changes:
            name = {"moved": "chair", "added": "box", "removed": "plant"}.get(c.kind, "object")
            out.append(
                LabeledChange(cluster=c, label=name, confidence=0.87,
                              description=f"A {name} ({c.kind}).",
                              evidence=[{"prompt": name, "score": 0.87}])
            )
        return out


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("room_sem")
    generate_dataset(out, n_frames=10, height=180, width=240, depth_noise_std=0.004, seed=11)
    return out


def test_pipeline_uses_labeler_and_produces_descriptions(dataset, monkeypatch):
    monkeypatch.setattr(sam3_mod, "Sam3Labeler", _StubLabeler)
    scene_a, scene_b = load_synthetic_pair(dataset)
    cfg = load_config(overrides=["semantics.enabled=true"])

    changeset, *_ = compare_scenes(scene_a, scene_b, cfg, align_method="identity")

    assert _StubLabeler.calls == 1
    assert changeset.labeled, "labeler output should be attached to the ChangeSet"
    for lc in changeset.labeled:
        assert lc.label in {"chair", "box", "plant", "object"}
        assert lc.confidence > 0
    labels = {lc.label for lc in changeset.labeled}
    assert labels & {"chair", "box", "plant"}, labels


def test_disabled_semantics_uses_description_fallback(dataset):
    scene_a, scene_b = load_synthetic_pair(dataset)
    cfg = load_config(overrides=["semantics.enabled=false"])
    changeset, *_ = compare_scenes(scene_a, scene_b, cfg, align_method="identity")
    assert changeset.labeled
    assert all(lc.label == "object" for lc in changeset.labeled)
    assert all(lc.description for lc in changeset.labeled)


def test_caption_uses_object_size_for_moved():
    from hp.semantics.caption import describe

    c = ChangeCluster(
        id=0, kind="moved",
        centroid=np.array([1.0, 1.0, 0.5]),
        bbox_min=np.array([0.0, 0.0, 0.0]), bbox_max=np.array([2.0, 1.0, 0.5]),
        n_voxels=10, volume=0.05, voxel_indices=np.zeros((4, 3), dtype=int),
        object_extent=np.array([0.4, 0.4, 0.5]),
        move_from=np.array([0.0, 0.0, 0.25]), move_to=np.array([2.0, 1.0, 0.25]),
    )
    lc = LabeledChange(cluster=c, label="chair", confidence=0.9, description="")
    text = describe(lc)
    assert "40 x 40 x 50 cm" in text          # object size, not the 2 m union bbox
    assert "moved" in text
