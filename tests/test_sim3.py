"""Tests for Sim(3) estimation and RANSAC alignment."""

import numpy as np
import pytest

from hp.align import Sim3, ransac_sim3, umeyama


def _random_sim3(rng: np.random.Generator) -> Sim3:
    # random rotation via QR
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return Sim3(float(rng.uniform(0.5, 2.0)), q, rng.normal(size=3) * 0.5)


def test_umeyama_recovers_known_transform():
    rng = np.random.default_rng(0)
    truth = _random_sim3(rng)
    src = rng.normal(size=(50, 3))
    dst = truth.apply_points(src)

    est = umeyama(src, dst)
    assert est.scale == pytest.approx(truth.scale, rel=1e-6)
    assert np.allclose(est.rotation, truth.rotation, atol=1e-6)
    assert np.allclose(est.translation, truth.translation, atol=1e-6)
    assert est.rms_error(src, dst) < 1e-8


def test_umeyama_is_noise_robust():
    rng = np.random.default_rng(1)
    truth = _random_sim3(rng)
    src = rng.normal(size=(200, 3))
    dst = truth.apply_points(src) + rng.normal(scale=0.01, size=(200, 3))
    est = umeyama(src, dst)
    assert est.scale == pytest.approx(truth.scale, rel=0.05)
    assert est.rms_error(src, dst) < 0.05


def test_ransac_sim3_rejects_outliers():
    rng = np.random.default_rng(2)
    truth = Sim3(1.5, np.eye(3), np.array([0.2, -0.3, 0.1]))
    src = rng.normal(size=(100, 3))
    dst = truth.apply_points(src)
    # corrupt 40% of the correspondences
    outliers = rng.choice(100, size=40, replace=False)
    dst[outliers] += rng.normal(scale=3.0, size=(40, 3))

    est = ransac_sim3(src, dst, threshold=0.05, iterations=1000, seed=0)
    assert est.scale == pytest.approx(1.5, rel=1e-3)
    assert est.inliers is not None
    assert est.inliers.sum() >= 55
    # inliers fit very well, outliers are excluded
    assert est.rms_error(src[est.inliers], dst[est.inliers]) < 0.05


def test_sim3_pose_transform_is_rigid():
    rng = np.random.default_rng(3)
    sim = _random_sim3(rng)
    pose = np.eye(4)
    pose[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    pose[:3, 3] = rng.normal(size=3)
    out = sim.apply_pose(pose)
    rot = out[:3, :3]
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(rot) == pytest.approx(1.0)


def test_sim3_inverse_and_compose():
    rng = np.random.default_rng(4)
    a = _random_sim3(rng)
    b = _random_sim3(rng)
    pts = rng.normal(size=(20, 3))
    # compose applies b first, then a
    composed = a.compose(b)
    assert np.allclose(composed.apply_points(pts), a.apply_points(b.apply_points(pts)), atol=1e-9)
    # inverse round-trips
    assert np.allclose(a.inverse().apply_points(a.apply_points(pts)), pts, atol=1e-9)


def test_identity_is_noop():
    pts = np.arange(9, dtype=float).reshape(3, 3)
    ident = Sim3.identity()
    assert np.allclose(ident.apply_points(pts), pts)
