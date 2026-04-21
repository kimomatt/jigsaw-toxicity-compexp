"""Tier 3 cluster assignment concepts."""

from __future__ import annotations

from ast import Is
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
    
    # Is type A a subtype/category of type B?
    if not np.issubdtype(assignments.dtype, np.integer):
        # can it be converted to integers without loss of information? (e.g. floats that are all whole numbers)
        if np.all(np.equal(assignments, assignments.astype(int))):
            assignments = assignments.astype(int)
        else:
            raise ValueError("tier3_assignments must contain integer cluster ids")

    # for assignments < 0 NumPy performs the comparison elementwise across the entire array and returns a new boolean array.
    if np.any(assignments < 0):
        raise ValueError("tier3_assignments must be non-negative")

    k = int(assignments.max()) + 1
    concepts: List[Concept] = []

    # going through each cluster id from 0 to k-1 and creating a concept for each cluster id
    # define a function _fn that takes in a list of texts and returns a binary numpy array where each element is 1 if the corresponding assignment is equal to the current cluster id, and 0 otherwise, part of the definition of the concept for that cluster id
    for cluster_id in range(k):
        
        # does _cid:int = cluster_id to capture the current value of cluster_id in the loop so that it can be used in the definition of _fn without being affected by the next iterations of the loop? (if we just used cluster_id directly in _fn, it would refer to the variable cluster_id which changes in each iteration of the loop, so all concepts would end up using the final value of cluster_id after the loop finishes, which is not what we want)
        
        def _fn(texts: Sequence[str], _cid: int = cluster_id) -> np.ndarray:
            # texts is not actually used in the function, but we include it as an argument to match the expected signature of a concept function, which takes in a list of texts and returns a numpy array of concept values for those texts. In this case, the concept values are determined solely by the external cluster assignments and do not depend on the input texts.
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
