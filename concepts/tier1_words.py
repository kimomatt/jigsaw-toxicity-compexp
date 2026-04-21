"""Tier 1 word presence concepts built from spaCy tokenization."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import string
from typing import Iterable, List, Sequence

import numpy as np
import spacy

from .base import Concept

# Minimal exclusion list for top-k extraction, matching the spirit of Compexp.
DEFAULT_STOPWORDS = {
    "a",
    "an",
    "of",
    "the",
    ".",
    ",",
}


# we use the decorator @lru_cache(maxsize=1) to cache the result of loading the spaCy model, so that we only load it once and reuse it for subsequent calls to _get_tier1_nlp. This can improve performance by avoiding the overhead of loading the model multiple times, especially if we are processing many texts and need to tokenize them repeatedly. The maxsize=1 argument means that we only want to cache one instance of the model, which is sufficient since we only need one instance for our purposes.
@lru_cache(maxsize=1)
def _get_tier1_nlp():
    """Use Compexp-style English tokenization with non-token components disabled."""
    return spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])


def _spacy_tokens(text: str) -> List[str]:
    # doc is an iterable of tokens
    doc = _get_tier1_nlp().make_doc(text)

    #  Filter (drops items)
    # [x for x in data if x > 0]
    # → [1, 2, 3]

    # Ternary (keeps all items)
    # [x if x > 0 else 0 for x in data]
    # → [1, 2, 3, 0, 0]

    return [token.lower_ for token in doc if not token.is_space]

# takes any iterable of words and returns cleaned list 
def _normalize_words(words: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw_word in words:
        # strip removes surrounding whitespace and lower converts to lowercase
        word = raw_word.strip().lower()
        if not word or word in seen:
            continue
        seen.add(word)
        out.append(word)
    return out

# checks whether token is pure punctuation
def _is_pure_punctuation(token: str) -> bool:
    return bool(token) and all(char in string.punctuation for char in token)

# takes a list of sentences and returns a list of sets of tokens for each sentence, using spaCy tokenization. This allows for fast membership testing of whether a word is present in a sentence by checking if it is in the corresponding set as long as u know the index of the sentence in the original list
# the other reason for it being a set is that it enforces uniqueness of tokens, so then when measuring doc frequency, we count whether a token appears in a document at least once rather than how many times it appears total
def _token_sets(texts: Sequence[str]) -> List[set[str]]:
    """Tokenize each text once and cache set membership for fast binary lookups."""
    return [set(_spacy_tokens(text)) for text in texts]

# after the *, the following parameters must be passed as keyword arguments, not positional arguments. This can help improve code readability and prevent errors by making it clear which arguments are being passed to the function, especially when there are multiple optional parameters with default values, order then doesn't matter for the keyword arguments
def build_tier1_vocabulary(
    texts: Sequence[str],
    *,
    top_k: int = 200,
    min_doc_freq: int = 5,
    max_doc_frac: float = 0.4,
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
) -> List[str]:
    """Build Tier 1 vocabulary from top-k frequent non-stopword spaCy tokens.

    Frequency is document frequency (how many texts contain the token).
    """
    # takes in a sequence of texts and the number of top tokens to return, as well as how many times a word needs to appaear to be included in the candidates and the maximum fraction of documents a word can appear in to be included in the candidates (if it appears in more than that fraction of documents, it is likely not a useful concept for distinguishing between different texts)
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if min_doc_freq < 1:
        raise ValueError("min_doc_freq must be >= 1")
    if not (0.0 < max_doc_frac <= 1.0):
        raise ValueError("max_doc_frac must be in (0.0, 1.0]")

    stopword_set = {w.strip().lower() for w in stopwords}
    n_docs = len(texts)
    max_doc_count = max(1, int(np.floor(max_doc_frac * n_docs)))

    # going thru each token in each text
    doc_freq: Counter[str] = Counter()
    for toks in _token_sets(texts):
        for tok in toks:
            if tok in stopword_set:
                continue
            if _is_pure_punctuation(tok):
                continue
            doc_freq[tok] += 1

    candidates = [
        (w, c)
        for w, c in doc_freq.items()
        if c >= min_doc_freq and c <= max_doc_count
    ]
    candidates.sort(key=lambda x: (-x[1], x[0]))
    return [w for w, _ in candidates[:top_k]]

# turn a list of words into a list of concept objects
def make_word_concepts(words: Sequence[str]) -> List[Concept]:
    """Build one concept per word with case-insensitive spaCy token matching."""
    concepts: List[Concept] = []
    for word in _normalize_words(words):

        def _fn(texts: Sequence[str], _word: str = word) -> np.ndarray:
            out = np.zeros(len(texts), dtype=np.uint8)
            for i, text in enumerate(texts):
                tokens = _spacy_tokens(text)
                out[i] = np.uint8(_word in tokens)
            return out
            # returns binary vector saying whether each text contains the target word

        concepts.append(
            Concept(
                name=f"has_word::{word}",
                tier=1,
                description=f"Text contains token '{word}'",
                fn=_fn,
            )
        )
    return concepts


def build_word_concept_values(texts: Sequence[str], words: Sequence[str]) -> np.ndarray:
    """Build the full Tier 1 binary matrix with one tokenization pass over the batch."""
    normalized_words = _normalize_words(words)
    token_sets = _token_sets(texts)
    values = np.zeros((len(texts), len(normalized_words)), dtype=np.uint8)

    for row_idx, token_set in enumerate(token_sets):
        for col_idx, word in enumerate(normalized_words):
            # check if each word is in each token set
            values[row_idx, col_idx] = np.uint8(word in token_set)

    return values
