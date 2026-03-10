"""ConceptSet builder."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .base import ConceptSet
from .tier1_words import make_word_concepts
from .tier2_primitives import make_primitive_concepts
from .tier3_clusters import make_cluster_concepts
from .utils import validate_binary_matrix


def build_concept_set(
    texts: Sequence[str],
    tier: int,
    tier1_words: Optional[Sequence[str]] = None,
    tier3_assignments: Optional[np.ndarray] = None,
    meta: Optional[Dict] = None,
    text_ids: Optional[Sequence[str]] = None,
) -> ConceptSet:
    """Build a ConceptSet for the given tier."""
    n = len(texts)
    if text_ids is not None and len(text_ids) != n:
        raise ValueError("text_ids must align with texts")

    if tier == 1:
        if not tier1_words:
            raise ValueError("Tier 1 requires tier1_words")
        concepts = make_word_concepts(tier1_words)
    elif tier == 2:
        concepts = make_primitive_concepts()
    elif tier == 3:
        if tier3_assignments is None:
            raise ValueError("Tier 3 requires tier3_assignments")
        assignments = np.asarray(tier3_assignments)
        if assignments.shape[0] != n:
            raise ValueError("tier3_assignments must align with texts")
        concepts = make_cluster_concepts(assignments)
    else:
        raise ValueError(f"Unsupported tier: {tier}")

    concept_names_seen = set()
    deduped = []
    for concept in concepts:
        if concept.name in concept_names_seen:
            continue
        concept_names_seen.add(concept.name)
        deduped.append(concept)
    concepts = sorted(deduped, key=lambda c: c.name)

    cols = [concept.fn(texts).astype(np.uint8, copy=False) for concept in concepts]
    values = np.column_stack(cols) if cols else np.zeros((n, 0), dtype=np.uint8)
    validate_binary_matrix(values)

    out_meta: Dict = dict(meta or {})
    out_meta.setdefault("tier", tier)
    out_meta.setdefault("concept_count", len(concepts))

    return ConceptSet(
        concepts=concepts,
        values=values,
        text_ids=list(text_ids) if text_ids is not None else None,
        meta=out_meta,
    )
