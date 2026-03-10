"""Utility helpers for concept annotation."""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def tokenize(text: str) -> List[str]:
    """Tokenize into lowercase alphabetic/apostrophe words."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def validate_binary_matrix(values: np.ndarray) -> None:
    """Assert matrix is uint8 with only 0/1 values."""
    if values.dtype != np.uint8:
        raise AssertionError(f"Expected dtype=uint8, got {values.dtype}")
    uniq = np.unique(values)
    if not np.all(np.isin(uniq, [0, 1])):
        raise AssertionError(f"Expected binary values in {{0,1}}, got {uniq}")


def coverage_stats(values: np.ndarray, names: Sequence[str]) -> List[Tuple[str, float]]:
    """Return firing rates sorted descending."""
    rates = values.mean(axis=0) if values.size else np.array([], dtype=float)
    pairs = [(str(n), float(r)) for n, r in zip(names, rates)]
    return sorted(pairs, key=lambda x: x[1], reverse=True)


def pretty_print_single(text: str, names: Sequence[str], values: np.ndarray) -> None:
    """Print concept name/value pairs for the first row."""
    print("Text:")
    print(text)
    print("\nConcept outputs (row 0):")
    for idx, name in enumerate(names):
        print(f"{name:20s} {int(values[0, idx])}")
