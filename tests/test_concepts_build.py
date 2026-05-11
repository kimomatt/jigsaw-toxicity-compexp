import pytest
import numpy as np


# conceptset = build_concept_set(
#   texts=texts,
#   tier=1,
#   tier1_words=vocab,
#   text_ids=ids,
#   meta={
#       "dataset": "val_metadata_csv",
#       "tier1_top_k": 300,
#       "tier1_min_freq": 20,
#       "tier1_max_freq": max_freq,
#       "tier1_vocab_size": len(vocab),
#       "fit_rows": len(texts),
#   },
# )

from concepts.build import build_concept_set

# wanna make sure that given a list of words it builds the expected concept set with the expected concept names and values. This is basically testing the make_word_concepts and build_word_concept_values functions that are called from build_concept_set when tier=1.
# test what happens if text is not provided, make sure that it errors
# then have a happy path small test
# edge cases: - duplicate words in the tier1_words list, make sure that it doesn't create duplicate concepts and that the concept names are correct, also test normalization of words (e.g. case insensitivity)

def test_build_concept_set_tier1_happy():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com."
    ]
    tier1_words = ["love", "hate", "awful"]
    ids = [1, 2, 3]
    conceptset = build_concept_set(
        texts=texts,
        tier=1,
        tier1_words=tier1_words,
        text_ids=ids,
        meta={"test": "test_build_concept_set_tier1"},
    )
    assert conceptset.text_ids == ids
    
    # order of concept names becomes ["has_word::awful", "has_word::hate", "has_word::love"] since the concepts are sorted alphabetically by name in build_concept_set, and the concept names are generated in make_word_concepts as "has_word::{word}".
    assert conceptset.concept_names == ["has_word::awful", "has_word::hate", "has_word::love"]

    assert conceptset.meta["tier"] == 1
    assert conceptset.meta["concept_count"] == 3

    expected_values = np.array([
        [0, 0, 1],  # "I love this product." contains "love"
        [1, 1, 0],  # "You are awful and I hate this." contains "awful" and "hate"
        [0, 0, 0],  # "Email me at example@email.com." does not contain any of the words

    ])
    assert np.array_equal(conceptset.values, expected_values)

def test_build_concept_set_tier1_missing_text():
    tier1_words = ["love", "hate", "awful"]
    ids = [1, 2, 3]
    with pytest.raises(ValueError, match="texts must be provided"):
        build_concept_set(
            texts=None,
            tier=1,
            tier1_words=tier1_words,
            text_ids=ids,
            meta={"test": "test_build_concept_set_tier1_missing_text"},
        )

def test_build_concept_set_tier1_missing_tier1_words():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com."
    ]
    tier1_words = []
    ids = [1, 2, 3]
    with pytest.raises(ValueError, match="Tier 1 requires tier1_words"):
        build_concept_set(
            texts=texts,
            tier=1,
            tier1_words=tier1_words,
            text_ids=ids,
            meta={"test": "test_build_concept_set_tier1_missing_tier1_words"},
        )

# check misalignment between text_ids and texts
def test_build_concept_set_tier1_misaligned_text_ids():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com."
    ]
    ids = [1, 2]  # Missing ID for the third text
    with pytest.raises(ValueError, match="text_ids must align with texts"):
        build_concept_set(
            texts=texts,
            tier=1,
            tier1_words=["love", "hate", "awful"],
            text_ids=ids,
            meta={"test": "test_build_concept_set_tier1_misaligned_text_ids"},
        )

# check duplicate words
def test_build_concept_set_tier1_duplicate_words():
    
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com."
    ]

    tier1_words = ["love", "hate", "awful", "love"]  # Duplicate word
    ids = [1, 2, 3]

    conceptset = build_concept_set(
            texts=texts,
            tier=1,
            tier1_words=tier1_words,
            text_ids=ids,
            meta={"test": "test_build_concept_set_tier1_duplicate_words"},
        )

    assert conceptset.text_ids == ids
    
    # order of concept names becomes ["has_word::awful", "has_word::hate", "has_word::love"] since the concepts are sorted alphabetically by name in build_concept_set, and the concept names are generated in make_word_concepts as "has_word::{word}".
    assert conceptset.concept_names == ["has_word::awful", "has_word::hate", "has_word::love"]
    expected_values = np.array([
        [0, 0, 1],  # "I love this product." contains "love"
        [1, 1, 0],  # "You are awful and I hate this." contains "awful" and "hate"
        [0, 0, 0],  # "Email me at example@email.com." does not contain any of the words

    ])
    assert np.array_equal(conceptset.values, expected_values)


# check case insensitivity of words (e.g. "Love" and "love" should be treated as the same word and not create duplicate concepts)

def test_build_concept_set_tier1_case_insensitivity():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com."
    ]
    tier1_words = ["love", "hate", "awful", "Love"]  # "Love" is a duplicate of "love" but with different case
    ids = [1, 2, 3]
    
    conceptset = build_concept_set(
            texts=texts,
            tier=1,
            tier1_words=tier1_words,
            text_ids=ids,
            meta={"test": "test_build_concept_set_tier1_duplicate_words"},
        )

    assert conceptset.text_ids == ids
    
    # order of concept names becomes ["has_word::awful", "has_word::hate", "has_word::love"] since the concepts are sorted alphabetically by name in build_concept_set, and the concept names are generated in make_word_concepts as "has_word::{word}".
    assert conceptset.concept_names == ["has_word::awful", "has_word::hate", "has_word::love"]
    expected_values = np.array([
        [0, 0, 1],  # "I love this product." contains "love"
        [1, 1, 0],  # "You are awful and I hate this." contains "awful" and "hate"
        [0, 0, 0],  # "Email me at example@email.com." does not contain any of the words

    ])
    assert np.array_equal(conceptset.values, expected_values)

# test unsupported tier
def test_build_concept_set_unsupported_tier():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com." 
    ]
    ids = [1, 2, 3]
    with pytest.raises(ValueError, match="Unsupported tier: 99"):
        build_concept_set(
            texts=texts,
            tier=99,  # Unsupported tier
            tier1_words=["love", "hate", "awful"],
            text_ids=ids,
            meta={"test": "test_build_concept_set_unsupported_tier"},
        )

# this is called from build concept set:
# concepts = make_word_concepts(tier1_words)

# def make_word_concepts(words: Sequence[str]) -> List[Concept]:
#     """Build one concept per word with case-insensitive spaCy token matching."""
#     concepts: List[Concept] = []
#     for word in _normalize_words(words):

#         def _fn(texts: Sequence[str], _word: str = word) -> np.ndarray:
#             out = np.zeros(len(texts), dtype=np.uint8)
#             for i, text in enumerate(texts):
#                 tokens = _spacy_tokens(text)
#                 out[i] = np.uint8(_word in tokens)
#             return out
#             # returns binary vector saying whether each text contains the target word

#         concepts.append(
#             Concept(
#                 name=f"has_word::{word}",
#                 tier=1,
#                 description=f"Text contains token '{word}'",
#                 fn=_fn,
#             )
#         )
#     return concepts
