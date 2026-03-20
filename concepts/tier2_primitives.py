"""Tier 2 explicit linguistic concepts built from spaCy annotations."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, List, Sequence

import numpy as np
import spacy
from spacy.tokens import Doc

from .base import Concept

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves", "u", "ur"}
THIRD_PERSON = {
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "they",
    "them",
    "their",
    "theirs",
    "themself",
    "themselves",
}
MODAL_LEMMAS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
QUOTE_CHARS = {'"', "“", "”", "`", "``", "''", "«", "»"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "expl"}


@lru_cache(maxsize=1)
def _get_tier2_nlp():
    """Load the full English pipeline for explicit linguistic annotation."""
    return spacy.load("en_core_web_sm")


def _parse_docs(texts: Sequence[str]) -> List[Doc]:
    return list(_get_tier2_nlp().pipe(texts, batch_size=128))


def _has_token(doc: Doc, vocab: set[str]) -> bool:
    return any(token.lower_ in vocab for token in doc)


def _has_entity(doc: Doc, label: str) -> bool:
    return any(ent.label_ == label for ent in doc.ents)


def _has_question_clause(doc: Doc, text: str) -> bool:
    if "?" in text:
        return True
    return any(token.tag_ in {"WDT", "WP", "WP$", "WRB"} for token in doc)


def _has_exclamation_clause(doc: Doc, text: str) -> bool:
    return "!" in text


def _has_quoted_span(doc: Doc, text: str) -> bool:
    del doc
    return any(char in text for char in QUOTE_CHARS)


def _has_negated_predicate(doc: Doc, text: str) -> bool:
    del text
    return any(token.dep_ == "neg" for token in doc)


def _has_modalized_predicate(doc: Doc, text: str) -> bool:
    del text
    return any(token.tag_ == "MD" or (token.dep_ == "aux" and token.lemma_.lower() in MODAL_LEMMAS) for token in doc)


def _has_future_construction(doc: Doc, text: str) -> bool:
    del text
    for token in doc:
        if token.lower_ in {"will", "shall", "'ll"}:
            return True
        if token.lower_ == "going" and token.dep_ in {"aux", "ROOT"}:
            next_tokens = list(doc[token.i + 1 : token.i + 3])
            if next_tokens and any(tok.lower_ == "to" for tok in next_tokens):
                return True
    return False


def _has_imperative_clause(doc: Doc, text: str) -> bool:
    del text
    if not doc:
        return False
    for sent in doc.sents:
        tokens = [token for token in sent if not token.is_punct and not token.is_space]
        if not tokens:
            continue
        root = sent.root
        if root.pos_ != "VERB":
            continue
        if any(child.dep_ in SUBJECT_DEPS for child in root.children):
            continue
        if tokens[0] == root:
            return True
    return False


def _has_coordination(doc: Doc, text: str) -> bool:
    del text
    return any(token.dep_ in {"cc", "conj"} for token in doc)


def _has_copular_predication(doc: Doc, text: str) -> bool:
    del text
    return any(token.dep_ == "cop" for token in doc)


def _has_adjectival_predication(doc: Doc, text: str) -> bool:
    del text
    return any(token.dep_ in {"acomp", "amod"} or (token.dep_ == "ROOT" and token.pos_ == "ADJ") for token in doc)


def _has_direct_address(doc: Doc, text: str) -> bool:
    del text
    return any(token.dep_ == "vocative" for token in doc) or (
        _has_token(doc, SECOND_PERSON) and any(token.text == "," for token in doc)
    )


Extractor = tuple[str, str, Callable[[Doc, str], bool]]

ACTIVE_EXTRACTORS: List[Extractor] = sorted([
    (
        "has_first_person_reference",
        "Contains a first-person reference token.",
        lambda doc, text: _has_token(doc, FIRST_PERSON),
    ),
    (
        "has_second_person_reference",
        "Contains a second-person reference token.",
        lambda doc, text: _has_token(doc, SECOND_PERSON),
    ),
    (
        "has_third_person_reference",
        "Contains a third-person reference token.",
        lambda doc, text: _has_token(doc, THIRD_PERSON),
    ),
    ("has_exclamation_clause", "Contains an exclamation mark.", _has_exclamation_clause),
    ("has_quoted_span", "Contains quoted text.", _has_quoted_span),
    ("has_negated_predicate", "Contains dependency-marked predicate negation.", _has_negated_predicate),
    ("has_modalized_predicate", "Contains a modalized predicate.", _has_modalized_predicate),
    ("has_direct_address", "Contains direct address / vocative structure.", _has_direct_address),
], key=lambda item: item[0])

DISABLED_EXTRACTORS: List[Extractor] = sorted([
    ("has_person_entity", "Contains a named entity with PERSON label.", lambda doc, text: _has_entity(doc, "PERSON")),
    ("has_org_entity", "Contains a named entity with ORG label.", lambda doc, text: _has_entity(doc, "ORG")),
    ("has_gpe_entity", "Contains a named entity with GPE label.", lambda doc, text: _has_entity(doc, "GPE")),
    ("has_question_clause", "Contains a question clause or wh-question form.", _has_question_clause),
    ("has_future_construction", "Contains an explicit future construction such as will/shall/going to.", _has_future_construction),
    ("has_imperative_clause", "Contains an imperative-like clause.", _has_imperative_clause),
    ("has_coordination", "Contains a coordination structure.", _has_coordination),
    ("has_copular_predication", "Contains a copular predicate.", _has_copular_predication),
    ("has_adjectival_predication", "Contains adjectival predication or modification.", _has_adjectival_predication),
], key=lambda item: item[0])


def make_linguistic_concepts() -> List[Concept]:
    """Construct explicit linguistic Tier 2 concepts."""
    concepts: List[Concept] = []
    for name, description, extractor in ACTIVE_EXTRACTORS:

        def _fn(texts: Sequence[str], _extractor: Callable[[Doc, str], bool] = extractor) -> np.ndarray:
            docs = _parse_docs(texts)
            return np.array([np.uint8(_extractor(doc, text)) for doc, text in zip(docs, texts)], dtype=np.uint8)

        concepts.append(Concept(name=name, tier=2, description=description, fn=_fn))
    return concepts


def build_linguistic_concept_values(texts: Sequence[str]) -> np.ndarray:
    """Build the full Tier 2 matrix in one parse pass over the batch."""
    docs = _parse_docs(texts)
    values = np.zeros((len(texts), len(ACTIVE_EXTRACTORS)), dtype=np.uint8)

    for row_idx, (doc, text) in enumerate(zip(docs, texts)):
        for col_idx, (_, _, extractor) in enumerate(ACTIVE_EXTRACTORS):
            values[row_idx, col_idx] = np.uint8(extractor(doc, text))

    return values
