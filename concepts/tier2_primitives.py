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
QUOTE_CHARS = {'"', "“", "”", "`", "``", "''", "«", "»"}

# dependency labels denote what the words do in a sentence, for example if they are a subject
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "expl"}


@lru_cache(maxsize=1)
def _get_tier2_nlp():
    """Load the full English pipeline for explicit linguistic annotation."""
    return spacy.load("en_core_web_sm")


def _parse_docs(texts: Sequence[str]) -> List[Doc]:
    # convert each text into a spacy doc object, get list of those doc objects, using the pipe method which is more efficient for processing batches of texts than calling nlp on each text individually, and we set batch_size to 128 to control how many texts are processed together in each batch, which can help optimize performance by balancing memory usage and processing speed

    # essentailly a glorified grouped for loop that processes texts in batches for efficiency purpose 
    return list(_get_tier2_nlp().pipe(texts, batch_size=128))

# doc is the spacy object - text as well as all the linguistic annotations 
def _has_token(doc: Doc, vocab: set[str]) -> bool:
    # checks whether any token in the doc is in the given vocab set, which is used to check for the presence of first-person, second-person, or third-person reference tokens in the text. By converting the tokens to lowercase and using a set for the vocab, we can perform this check efficiently and in a case-insensitive manner.
    return any(token.lower_ in vocab for token in doc)


def _has_entity(doc: Doc, label: str) -> bool:
    # checks whether any named entity in the doc has the specified label, which is used to check for the presence of certain types of named entities such as PERSON, ORG, or GPE in the text. By iterating over the named entities in the doc and checking their labels, we can determine if the text contains any entities of the desired type.
    return any(ent.label_ == label for ent in doc.ents)


def _has_question_clause(doc: Doc, text: str) -> bool:
    # checks whether the text contains a question clause or wh-question form, which is used to identify texts that are likely to be questions.
    if "?" in text:
        return True
    return any(token.tag_ in {"WDT", "WP", "WP$", "WRB"} for token in doc)


def _has_exclamation_clause(doc: Doc, text: str) -> bool:
    del doc
    return "!" in text


def _has_quoted_span(doc: Doc, text: str) -> bool:
    # deletes doc because we don't actually need it for this function, and we want to avoid accidentally using it in a way that would cause unnecessary overhead from the linguistic annotations when all we need to do is check for the presence of certain characters in the text. By deleting the doc variable, we can make it clear that we are not using it and avoid any confusion about whether it is needed for this function.
    del doc
    return any(char in text for char in QUOTE_CHARS)


def _has_negated_predicate(doc: Doc, text: str) -> bool:
    # meaning of predicate is flipped, predicate is what the subject is or does
    del text
    return any(token.dep_ == "neg" for token in doc)


def _has_modalized_predicate(doc: Doc, text: str) -> bool:
    del text
    return any(token.tag_ == "MD" for token in doc)


def _has_future_construction(doc: Doc, text: str) -> bool:
    # checks for the presence of explicit future constructions in the text, such as modal verbs like "will" or "shall", or the "going to" construction followed by a verb. By identifying these constructions, we can determine if the text is referring to future events or actions.
    del text
    for token in doc:
        if token.lower_ in {"will", "shall", "'ll"}:
            return True
        if token.lower_ == "going":
            if token.i + 2 < len(doc):
                if (
                    doc[token.i + 1].lower_ == "to"
                    and doc[token.i + 2].pos_ == "VERB"
                ):
                    return True
    return False


def _has_imperative_clause(doc: Doc, text: str) -> bool:
    del text
    if not doc:
        return False
    for sent in doc.sents:
        # iterating thru each sentence
        tokens = [token for token in sent if not token.is_punct and not token.is_space]
        if not tokens:
            continue
        # root is the core action/state
        root = sent.root
        if root.pos_ != "VERB":
            continue

        # children is words that depend directly on the root 
        # for imperatives the subject is typically omitted
        if any(child.dep_ in SUBJECT_DEPS for child in root.children):
            continue

        # set generated by chatgpt
        if tokens[0] == root or (tokens[0].lower_ in {"please", "kindly", "just", "now", "simply", "always", "never", "do", "don't"} and len(tokens) > 1 and tokens[1] == root):
            return True
    return False


def _has_coordination(doc: Doc, text: str) -> bool:
    # does sentence contain coordination structure, which is a grammatical construction where two or more elements (such as words, phrases, or clauses) are connected by a coordinating conjunction (such as "and", "or", "but") and typically have the same syntactic function in the sentence. 
    del text
    return any(token.dep_ in {"cc", "conj"} for token in doc)


def _has_copular_predication(doc: Doc, text: str) -> bool:
    # subject is being described rather than there being an action
    del text
    return any(token.dep_ == "cop" for token in doc)


def _has_adjectival_predication(doc: Doc, text: str) -> bool:
    # subject is being described (subset of copular predication) but specifially with an adjective
    del text
    for token in doc:
        # acomp means an adjective that completes the meaning of a verb and describes the subject
        if token.dep_ == "acomp":
            return True
        # checking for cases where the adjective is the root of the sentence and is linked to the subject by a copular verb, which is a verb that connects the subject to a subject complement that describes or identifies it, such as "is", "was", "seems", etc. In these cases, we look for a root token that is an adjective and has a child with the "cop" dependency label, which indicates that it is linked to the subject by a copular verb.
        if token.dep_ == "ROOT" and token.pos_ == "ADJ":
            if any(child.dep_ == "cop" for child in token.children):
                return True
            
        # do both checks for more robust coverage 

    return False


def _has_direct_address(doc: Doc, text: str) -> bool:
    del text
    # vocative means you directly address someone by name or title
    return any(token.dep_ == "vocative" for token in doc)

# type alias, extractor is tuple of concept name, description, extractor function, where extractor function takes in a spacy doc and the original text and returns a boolean indicating whether the concept is present in the text based on the linguistic annotations in the doc
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

AUDIT_EXTRACTORS: List[Extractor] = sorted([
    # add any extractors here that we want to audit for potential future inclusion as active concepts, but that we don't want to include in the default set of active concepts for now either because they are too noisy or too computationally expensive or for any other reason. By keeping them in a separate list, we can easily keep track of them and evaluate their performance and usefulness over time without affecting the current set of active concepts.
    # going to include them all here
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


# by making concepts with extractor functions that capture the specific extractor for that concept, we can ensure that each concept is correctly associated with its intended linguistic feature and you can easily call the concept function having the concept object 
# all concepts look the same, package of name, description, tier, function for computing the values
def make_linguistic_concepts() -> List[Concept]:
    """Construct explicit linguistic Tier 2 concepts."""
    concepts: List[Concept] = []
    for name, description, extractor in AUDIT_EXTRACTORS:
        # _extractor to capture current extractor for the concept
        def _fn(texts: Sequence[str], _extractor: Callable[[Doc, str], bool] = extractor) -> np.ndarray:
            docs = _parse_docs(texts)
            # calls the extractor function on each doc and text pair to determine whether the concept is present, and constructs a binary numpy array where each element is 1 if the concept is present in the corresponding text and 0 otherwise. 
            return np.array([np.uint8(_extractor(doc, text)) for doc, text in zip(docs, texts)], dtype=np.uint8)

        concepts.append(Concept(name=name, tier=2, description=description, fn=_fn))
    return concepts


def build_linguistic_concept_values(texts: Sequence[str]) -> np.ndarray:
    """Build the full Tier 2 matrix in one parse pass over the batch."""
    docs = _parse_docs(texts)
    values = np.zeros((len(texts), len(AUDIT_EXTRACTORS)), dtype=np.uint8)

    # for each doc/text pair, apply each extractor function to determine whether the corresponding concept is present, and fill in the values array with the results. 
    for row_idx, (doc, text) in enumerate(zip(docs, texts)):
        for col_idx, (_, _, extractor) in enumerate(AUDIT_EXTRACTORS):
            values[row_idx, col_idx] = np.uint8(extractor(doc, text))

    return values
