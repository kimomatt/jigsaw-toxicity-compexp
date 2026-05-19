"""Tier 1 word presence concepts built from spaCy tokenization."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import string
from typing import Iterable, List, Sequence

import spacy
import numpy as np

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

# will do variant with nltk stopword list to see how that affects the quality of the explanations and if it makes them more interpretable
# copied from the classic NLTK English stopword list:
# https://gist.githubusercontent.com/sebleier/554280/raw/
NLTK_STOPWORDS = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
}

# will also experiment with different doc frequencies 

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
# this is still useful for downstream binary concept construction even though vocabulary selection now uses total token frequency rather than document frequency
def _token_sets(texts: Sequence[str]) -> List[set[str]]:
    """Tokenize each text once and cache set membership for fast binary lookups."""
    return [set(_spacy_tokens(text)) for text in texts]

# after the *, the following parameters must be passed as keyword arguments, not positional arguments. This can help improve code readability and prevent errors by making it clear which arguments are being passed to the function, especially when there are multiple optional parameters with default values, order then doesn't matter for the keyword arguments
# ignores min_freq and max_freq in total mode but uses in doc mode 
def build_tier1_vocabulary(
    texts: Sequence[str],
    *,
    top_k: int = 200,
    min_freq: float | None = None,
    max_freq: float | None = None,
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
    freq_type: str = "total",
) -> List[str]:
    """Build Tier 1 vocabulary from top-k total-frequency non-stopword spaCy tokens."""
    if top_k < 0:
        raise ValueError("top_k must be >= 0")

    stopword_set = {w.strip().lower() for w in stopwords}

    if freq_type == "total":
        total_freq: Counter[str] = Counter()
        for text in texts:
            for tok in _spacy_tokens(text):
                if tok in stopword_set:
                    continue
                if _is_pure_punctuation(tok):
                    continue
                total_freq[tok] += 1

        candidates = []
        for w, c in total_freq.items():
            candidates.append((w, c))

        candidates.sort(key=lambda x: (-x[1], x[0]))
        return [w for w, _ in candidates[:top_k]]
    
    elif freq_type == "document":

        if min_freq is not None and (min_freq > 1 or min_freq < 0):
            raise ValueError("min_freq must be >= 0 and <= 1")
        if max_freq is not None and (max_freq < 0 or max_freq > 1):
            raise ValueError("max_freq must be >= 0 and <= 1")
        if min_freq is not None and max_freq is not None and min_freq > max_freq:
            raise ValueError("min_freq must be <= max_freq")
        doc_freq: Counter[str] = Counter()
        for text in texts:
            tokens = set(_spacy_tokens(text))
            for tok in tokens:
                if tok in stopword_set:
                    continue
                if _is_pure_punctuation(tok):
                    continue
                doc_freq[tok] += 1

        candidates = []
        for w, c in doc_freq.items():
            if min_freq is not None and c / len(texts) < min_freq:
                continue
            if max_freq is not None and c / len(texts) > max_freq:
                continue
            candidates.append((w, c))

        candidates.sort(key=lambda x: (-x[1], x[0]))
        return [w for w, _ in candidates[:top_k]]

    else:
        raise ValueError(f"Unsupported frequency type: {freq_type}")


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
