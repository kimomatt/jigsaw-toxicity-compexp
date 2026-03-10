"""Tier 1 word presence concepts."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence

import numpy as np

from .base import Concept
from .utils import tokenize

# Small dataset-agnostic anchor lexicon.
BASE_LEXICON: List[str] = [
    "idiot",
    "stupid",
    "moron",
    "dumb",
    "loser",
    "trash",
    "hate",
    "awful",
    "terrible",
    "toxic",
    "insult",
    "threat",
    "kill",
    "hurt",
    "attack",
    "fight",
    "love",
    "good",
    "great",
    "nice",
    "amazing",
    "please",
    "thanks",
    "sorry",
    "money",
    "cash",
    "dollar",
    "pay",
    "report",
    "ban",
    "block",
]

# Compact English stopword set for top-k extraction.
DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def _normalize_words(words: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw_word in words:
        word = raw_word.strip().lower()
        if not word or word in seen:
            continue
        seen.add(word)
        out.append(word)
    return out


def build_tier1_vocabulary(
    texts: Sequence[str],
    *,
    base_lexicon: Sequence[str] = BASE_LEXICON,
    top_k: int = 200,
    min_doc_freq: int = 5,
    max_doc_frac: float = 0.4,
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
) -> List[str]:
    """Build Tier 1 vocabulary = base lexicon + top-k frequent non-stopword tokens.

    Frequency is document frequency (how many texts contain the token).
    """
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if min_doc_freq < 1:
        raise ValueError("min_doc_freq must be >= 1")
    if not (0.0 < max_doc_frac <= 1.0):
        raise ValueError("max_doc_frac must be in (0.0, 1.0]")

    base_words = _normalize_words(base_lexicon)
    stopword_set = {w.strip().lower() for w in stopwords}
    base_set = set(base_words)
    n_docs = len(texts)
    max_doc_count = max(1, int(np.floor(max_doc_frac * n_docs)))

    doc_freq: Counter[str] = Counter()
    for text in texts:
        toks = set(tokenize(text))
        for tok in toks:
            if tok in stopword_set:
                continue
            if tok in base_set:
                continue
            if len(tok) < 2:
                continue
            doc_freq[tok] += 1

    candidates = [
        (w, c)
        for w, c in doc_freq.items()
        if c >= min_doc_freq and c <= max_doc_count
    ]
    candidates.sort(key=lambda x: (-x[1], x[0]))
    top_words = [w for w, _ in candidates[:top_k]]
    return base_words + top_words


def make_word_concepts(words: Sequence[str]) -> List[Concept]:
    """Build one concept per word with case-insensitive token matching."""
    concepts: List[Concept] = []
    for word in _normalize_words(words):

        def _fn(texts: Sequence[str], _word: str = word) -> np.ndarray:
            out = np.zeros(len(texts), dtype=np.uint8)
            for i, text in enumerate(texts):
                tokens = tokenize(text)
                out[i] = np.uint8(_word in tokens)
            return out

        concepts.append(
            Concept(
                name=f"has_word::{word}",
                tier=1,
                description=f"Text contains token '{word}'",
                fn=_fn,
            )
        )
    return concepts
