"""Tier 3 cluster assignment concepts."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .base import Concept


def make_cluster_concepts(assignments: np.ndarray) -> List[Concept]:
    """Build one concept per cluster id from external assignments.

    Assignments must be a 1D integer array with non-negative ids.
    Concepts are generated for cluster ids 0..k-1 where k=max_id+1.
    """
    if assignments.ndim != 1:
        raise ValueError("tier3_assignments must be 1D")
    if assignments.size == 0:
        return []

    if not np.issubdtype(assignments.dtype, np.integer):
        if np.all(np.equal(assignments, assignments.astype(int))):
            assignments = assignments.astype(int)
        else:
            raise ValueError("tier3_assignments must contain integer cluster ids")

    if np.any(assignments < 0):
        raise ValueError("tier3_assignments must be non-negative")

    k = int(assignments.max()) + 1
    concepts: List[Concept] = []
    for cluster_id in range(k):

        def _fn(texts: Sequence[str], _cid: int = cluster_id) -> np.ndarray:
            if len(texts) != assignments.shape[0]:
                raise ValueError("texts length must match tier3_assignments length")
            return (assignments == _cid).astype(np.uint8)

        concepts.append(
            Concept(
                name=f"cluster::{cluster_id}",
                tier=3,
                description=f"External assignment indicates cluster {cluster_id}",
                fn=_fn,
            )
        )
    return concepts
