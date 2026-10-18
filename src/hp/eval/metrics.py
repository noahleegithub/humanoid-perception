"""Metrics for change detection against ground truth."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from hp.types import ChangeCluster

__all__ = ["GTChange", "evaluate_changes", "EvaluationResult"]

_KINDS = ("added", "removed", "moved")


class GTChange:
    """A single ground-truth change."""

    def __init__(self, kind: str, label: str, name: str = "", before=None, after=None):
        self.kind = kind
        self.label = label
        self.name = name or label
        self.before = None if before is None else np.asarray(before, dtype=np.float64).reshape(3)
        self.after = None if after is None else np.asarray(after, dtype=np.float64).reshape(3)

    @classmethod
    def from_dict(cls, d: dict) -> "GTChange":
        before = d["before"] if isinstance(d.get("before"), dict) else None
        after = d["after"] if isinstance(d.get("after"), dict) else None
        return cls(
            kind=d["kind"],
            label=d.get("label", "object"),
            name=d.get("name", d.get("label", "object")),
            before=None if before is None else before["center"],
            after=None if after is None else after["center"],
        )


def _cluster_error(cluster: ChangeCluster, gt: GTChange) -> float:
    """Location error (metres) between a detected cluster and a GT change."""
    if gt.kind == "moved" and gt.before is not None and gt.after is not None:
        if cluster.move_from is None or cluster.move_to is None:
            return float("inf")
        return float(
            max(
                np.linalg.norm(cluster.move_from - gt.before),
                np.linalg.norm(cluster.move_to - gt.after),
            )
        )
    if gt.kind == "added":
        if gt.after is None:
            return float("inf")
        return float(np.linalg.norm(cluster.centroid - gt.after))
    if gt.before is None:
        return float("inf")
    return float(np.linalg.norm(cluster.centroid - gt.before))


class EvaluationResult:
    def __init__(self, tp: int, fp: int, fn: int, matches: list[dict], tol_m: float):
        self.tp, self.fp, self.fn = tp, fp, fn
        self.matches = matches
        self.tol_m = tol_m

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tol_m": self.tol_m,
            "matches": self.matches,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"tp={self.tp} fp={self.fp} fn={self.fn} "
            f"P={self.precision:.2f} R={self.recall:.2f} F1={self.f1:.2f}"
        )


def evaluate_changes(
    clusters: Sequence[ChangeCluster],
    gt_changes: Iterable[GTChange | dict],
    *,
    tol_m: float = 0.5,
    allow_moved_as_two: bool = False,
) -> EvaluationResult:
    """Greedily match detected clusters to ground-truth changes.

    A detection is a true positive if a cluster of the same ``kind`` is within
    ``tol_m`` of the ground-truth location.  With ``allow_moved_as_two`` a moved
    ground truth may also be satisfied by a matched removed+added pair (and a
    moved detection may satisfy a single added/removed change), which mirrors
    how a downstream user would read the report.
    """
    gts = [g if isinstance(g, GTChange) else GTChange.from_dict(g) for g in gt_changes]
    used: set[int] = set()
    tp = 0
    matches: list[dict] = []

    for gt in gts:
        best_idx, best_err = None, float("inf")
        for idx, cluster in enumerate(clusters):
            if idx in used:
                continue
            if cluster.kind == gt.kind:
                err = _cluster_error(cluster, gt)
            elif allow_moved_as_two and gt.kind == "moved" and cluster.kind in ("added", "removed"):
                ref = gt.after if cluster.kind == "added" else gt.before
                err = float("inf") if ref is None else float(np.linalg.norm(cluster.centroid - ref))
            elif allow_moved_as_two and cluster.kind == "moved" and gt.kind in ("added", "removed"):
                ref = gt.after if gt.kind == "added" else gt.before
                if ref is None or cluster.move_from is None or cluster.move_to is None:
                    err = float("inf")
                else:
                    err = float(
                        min(
                            np.linalg.norm(cluster.move_from - ref),
                            np.linalg.norm(cluster.move_to - ref),
                        )
                    )
            else:
                continue
            if err < best_err:
                best_err, best_idx = err, idx

        if best_idx is not None and best_err <= tol_m:
            used.add(best_idx)
            tp += 1
            matches.append(
                {"gt": gt.name, "gt_kind": gt.kind, "cluster_id": clusters[best_idx].id,
                 "cluster_kind": clusters[best_idx].kind, "error_m": best_err}
            )

    fn = len(gts) - tp
    fp = len(clusters) - len(used)
    return EvaluationResult(tp, fp, fn, matches, tol_m)
