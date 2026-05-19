import pytest

from concepts.tier1_words import build_tier1_vocabulary


def test_build_tier1_vocabulary_happy_doc():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com.",
    ]
    vocab = build_tier1_vocabulary(texts, min_freq=0.3, top_k=5, freq_type="document")
    assert isinstance(vocab, list)
    assert all(isinstance(w, str) for w in vocab)
    assert len(vocab) == 5
    assert vocab == ["i", "this", "and", "are", "at"]

def test_build_tier1_vocabulary_happy_total():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com.",
    ]
    vocab = build_tier1_vocabulary(texts, top_k=5, freq_type="total")
    assert isinstance(vocab, list)
    assert all(isinstance(w, str) for w in vocab)
    assert len(vocab) == 5
    assert vocab == ["i", "this", "and", "are", "at"]

def test_build_tier1_vocabulary_happy_total_ignore_freq():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com.",
    ]
    vocab = build_tier1_vocabulary(texts, top_k=5, freq_type="total", min_freq=100, max_freq=90)
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


def test_build_tier1_vocabulary_min_freq_doc():
    texts = [
        "bad bad bad",
        "bad dog",
        "dog cat",
    ]
    vocab = build_tier1_vocabulary(texts, min_freq=0.6, top_k=10, freq_type="document")
    assert isinstance(vocab, list)
    assert vocab == ["bad", "dog"]


def test_build_tier1_vocabulary_max_freq_doc():
    texts = [
        "bad bad bad",
        "bad dog",
        "dog cat",
    ]
    vocab = build_tier1_vocabulary(texts, max_freq=0.35, top_k=10, freq_type="document")
    assert isinstance(vocab, list)
    assert vocab == ["cat"]


def test_build_tier1_vocabulary_invalid_top_k():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com.",
    ]
    with pytest.raises(ValueError, match="top_k must be >= 0"):
        build_tier1_vocabulary(texts, top_k=-1)


def test_build_tier1_vocabulary_invalid_min_freq_doc():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com.",
    ]
    with pytest.raises(ValueError, match="min_freq must be >= 0 and <= 1"):
        build_tier1_vocabulary(texts, min_freq=-1, freq_type="document")


def test_build_tier1_vocabulary_invalid_max_freq_doc():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com.",
    ]
    with pytest.raises(ValueError, match="max_freq must be >= 0 and <= 1"):
        build_tier1_vocabulary(texts, max_freq=1.5, freq_type="document")


def test_build_tier1_vocabulary_invalid_freq_range_doc():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com.",
    ]
    with pytest.raises(ValueError, match="min_freq must be <= max_freq"):
        build_tier1_vocabulary(texts, min_freq=0.6, max_freq=0.35, freq_type="document")


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

def test_document_frequency_differs_from_total_frequency():
    texts = [
        "bad bad bad",
        "dog dog",
        "dog",
    ]

    vocab_total = build_tier1_vocabulary(texts, top_k=2, freq_type="total")
    vocab_doc = build_tier1_vocabulary(texts, top_k=2, freq_type="document")

    assert vocab_total == ["bad", "dog"]
    assert vocab_doc == ["dog", "bad"]

def test_unsupported_freq_type():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com.",
    ]
    with pytest.raises(ValueError, match="Unsupported frequency type"):
        build_tier1_vocabulary(texts, top_k=5, freq_type="unsupported")

def test_stopwords_override():
    texts = [
        "I love this product don.",
        "You are awful and I hate this.",
        "Email me at example@email.com",
    ]
    vocab = build_tier1_vocabulary(texts, top_k=100, freq_type="document", max_freq=0.4, stopwords=["don"])
    assert isinstance(vocab, list)
    assert "don" not in vocab
