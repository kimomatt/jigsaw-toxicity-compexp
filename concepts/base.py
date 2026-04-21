"""Core datatypes for concept annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# setting as frozen bc it should not be mutated after created
# the decorator @dataclass is a convenient way to create classes that are primarily used to store data, and it automatically generates methods like __init__, __repr__, and __eq__ based on the class attributes.
@dataclass(frozen=True)
class Concept:
    """A single binary concept defined over a batch of texts."""

    name: str
    tier: int
    description: str
    fn: Callable[[Sequence[str]], np.ndarray]

# not frozen bc u might mutate things like meta after creating the ConceptSet
@dataclass
class ConceptSet:
    """Concept annotation output matrix and metadata."""

    concepts: List[Concept]
    values: np.ndarray
    text_ids: Optional[List[str]] = None
    meta: Optional[Dict] = None

    # turns a method into an attribute-like computed field so u can do object.concept_names instead of object.concept_names()
    @property
    def concept_names(self) -> List[str]:
        return [c.name for c in self.concepts]
