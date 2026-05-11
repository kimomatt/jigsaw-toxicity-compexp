import pytest

from concepts.tier1_words import build_tier1_vocabulary


def test_build_tier1_vocabulary_happy():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com.",
    ]
    vocab = build_tier1_vocabulary(texts, min_freq=1, top_k=5)
    assert isinstance(vocab, list)
    assert all(isinstance(w, str) for w in vocab)
    assert len(vocab) == 5
    assert vocab == ["i", "this", "and", "are", "at"]


def test_build_tier1_vocabulary_top_k_zero():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com.",
    ]
    vocab = build_tier1_vocabulary(texts, top_k=0)
    assert isinstance(vocab, list)
    assert len(vocab) == 0
    assert vocab == []


def test_build_tier1_vocabulary_min_freq():
    texts = [
        "bad bad bad",
        "bad dog",
        "dog cat",
    ]
    vocab = build_tier1_vocabulary(texts, min_freq=2, top_k=10)
    assert isinstance(vocab, list)
    assert vocab == ["bad", "dog"]


def test_build_tier1_vocabulary_max_freq():
    texts = [
        "bad bad bad",
        "bad dog",
        "dog cat",
    ]
    vocab = build_tier1_vocabulary(texts, min_freq=1, max_freq=2, top_k=10)
    assert isinstance(vocab, list)
    assert vocab == ["dog", "cat"]


def test_build_tier1_vocabulary_invalid_top_k():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com.",
    ]
    with pytest.raises(ValueError, match="top_k must be >= 0"):
        build_tier1_vocabulary(texts, top_k=-1)


def test_build_tier1_vocabulary_invalid_min_freq():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com.",
    ]
    with pytest.raises(ValueError, match="min_freq must be >= 1"):
        build_tier1_vocabulary(texts, min_freq=0)


def test_build_tier1_vocabulary_invalid_max_freq():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com.",
    ]
    with pytest.raises(ValueError, match="max_freq must be >= 1"):
        build_tier1_vocabulary(texts, max_freq=0)


def test_build_tier1_vocabulary_invalid_freq_range():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com.",
    ]
    with pytest.raises(ValueError, match="min_freq must be <= max_freq"):
        build_tier1_vocabulary(texts, min_freq=3, max_freq=2)


def test_build_tier1_vocabulary_stopword_punctuation_filter():
    texts = [
        "I love a product.",
        "You are an awful person, I hate all of this.",
        "The email of your kid; it sucks.",
    ]
    vocab = build_tier1_vocabulary(texts, min_freq=1)
    assert isinstance(vocab, list)
    assert "an" not in vocab
    assert "the" not in vocab
    assert "of" not in vocab
    assert "a" not in vocab
    assert "." not in vocab
    assert "," not in vocab
    assert ";" not in vocab
