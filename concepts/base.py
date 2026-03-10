"""Core datatypes for concept annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Concept:
    """A single binary concept defined over a batch of texts."""

    name: str
    tier: int
    description: str
    fn: Callable[[Sequence[str]], np.ndarray]


@dataclass
class ConceptSet:
    """Concept annotation output matrix and metadata."""

    concepts: List[Concept]
    values: np.ndarray
    text_ids: Optional[List[str]] = None
    meta: Optional[Dict] = None

    @property
    def concept_names(self) -> List[str]:
        return [c.name for c in self.concepts]
