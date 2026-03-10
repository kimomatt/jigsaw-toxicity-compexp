"""Tier 2 portable rule-based primitive concepts."""

from __future__ import annotations

import re
from typing import List, Sequence

import numpy as np

from .base import Concept
from .utils import tokenize

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves", "u", "ur"}
NEGATION = {"not", "no", "never", "none", "cannot"}
MODALS = {"can", "could", "should", "would", "may", "might", "must", "will", "shall"}
NEGATIVE_WORDS = {"idiot", "stupid", "dumb", "hate", "awful", "terrible", "trash", "loser"}
POSITIVE_WORDS = {"good", "great", "love", "nice", "excellent", "amazing", "thanks", "happy"}
VIOLENCE_WORDS = {"kill", "hurt", "attack", "fight", "punch", "shoot", "stab", "destroy"}
MONEY_WORDS = {"money", "cash", "dollar", "dollars", "bucks", "price", "pay", "paid"}

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
ALL_CAPS_RE = re.compile(r"\b[A-Z]{3,}\b")


def _binary_from_texts(texts: Sequence[str], predicate) -> np.ndarray:
    out = np.zeros(len(texts), dtype=np.uint8)
    for i, text in enumerate(texts):
        out[i] = np.uint8(1 if predicate(text) else 0)
    return out


def make_primitive_concepts() -> List[Concept]:
    """Construct Tier 2 rule-based concepts."""
    concepts: List[Concept] = []

    concepts.append(
        Concept(
            name="has_first_person",
            tier=2,
            description="Contains first-person pronoun tokens.",
            fn=lambda texts: _binary_from_texts(
                texts, lambda t: any(tok in FIRST_PERSON for tok in tokenize(t))
            ),
        )
    )
    concepts.append(
        Concept(
            name="has_second_person",
            tier=2,
            description="Contains second-person pronoun tokens including 'u' and 'ur'.",
            fn=lambda texts: _binary_from_texts(
                texts, lambda t: any(tok in SECOND_PERSON for tok in tokenize(t))
            ),
        )
    )
    concepts.append(
        Concept(
            name="has_negation",
            tier=2,
            description="Contains negation such as not/no/never/n't.",
            fn=lambda texts: _binary_from_texts(
                texts,
                lambda t: any(tok in NEGATION or tok.endswith("n't") for tok in tokenize(t)),
            ),
        )
    )
    concepts.append(
        Concept(
            name="has_modal",
            tier=2,
            description="Contains a modal verb.",
            fn=lambda texts: _binary_from_texts(
                texts, lambda t: any(tok in MODALS for tok in tokenize(t))
            ),
        )
    )
    concepts.append(
        Concept(
            name="is_question",
            tier=2,
            description="Text contains a question mark.",
            fn=lambda texts: _binary_from_texts(texts, lambda t: "?" in t),
        )
    )
    concepts.append(
        Concept(
            name="is_exclamation",
            tier=2,
            description="Text contains an exclamation mark.",
            fn=lambda texts: _binary_from_texts(texts, lambda t: "!" in t),
        )
    )
    concepts.append(
        Concept(
            name="has_url",
            tier=2,
            description="Text contains a URL (http(s):// or www.).",
            fn=lambda texts: _binary_from_texts(texts, lambda t: bool(URL_RE.search(t))),
        )
    )
    concepts.append(
        Concept(
            name="has_email",
            tier=2,
            description="Text contains an email address.",
            fn=lambda texts: _binary_from_texts(texts, lambda t: bool(EMAIL_RE.search(t))),
        )
    )
    concepts.append(
        Concept(
            name="has_number",
            tier=2,
            description="Text contains a numeric token.",
            fn=lambda texts: _binary_from_texts(texts, lambda t: bool(NUMBER_RE.search(t))),
        )
    )
    concepts.append(
        Concept(
            name="has_all_caps_word",
            tier=2,
            description="Contains an all-caps word with length >= 3.",
            fn=lambda texts: _binary_from_texts(texts, lambda t: bool(ALL_CAPS_RE.search(t))),
        )
    )
    concepts.append(
        Concept(
            name="long_sentence",
            tier=2,
            description="Roughly long sentence (>=25 tokens).",
            fn=lambda texts: _binary_from_texts(texts, lambda t: len(tokenize(t)) >= 25),
        )
    )
    concepts.append(
        Concept(
            name="has_negative_word",
            tier=2,
            description="Contains a negative lexicon word.",
            fn=lambda texts: _binary_from_texts(
                texts, lambda t: any(tok in NEGATIVE_WORDS for tok in tokenize(t))
            ),
        )
    )
    concepts.append(
        Concept(
            name="has_positive_word",
            tier=2,
            description="Contains a positive lexicon word.",
            fn=lambda texts: _binary_from_texts(
                texts, lambda t: any(tok in POSITIVE_WORDS for tok in tokenize(t))
            ),
        )
    )
    concepts.append(
        Concept(
            name="has_violence_word",
            tier=2,
            description="Contains a violence-related lexicon word.",
            fn=lambda texts: _binary_from_texts(
                texts, lambda t: any(tok in VIOLENCE_WORDS for tok in tokenize(t))
            ),
        )
    )
    concepts.append(
        Concept(
            name="has_money_word",
            tier=2,
            description="Contains a money lexicon word or '$'.",
            fn=lambda texts: _binary_from_texts(
                texts, lambda t: ("$" in t) or any(tok in MONEY_WORDS for tok in tokenize(t))
            ),
        )
    )

    return concepts
