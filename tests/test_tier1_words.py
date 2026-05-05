import pytest
from concepts.tier1_words import build_tier1_vocabulary

# after the *, the following parameters must be passed as keyword arguments, not positional arguments. This can help improve code readability and prevent errors by making it clear which arguments are being passed to the function, especially when there are multiple optional parameters with default values, order then doesn't matter for the keyword arguments
# def build_tier1_vocabulary(
#     texts: Sequence[str],
#     *,
#     top_k: int = 200,
#     min_doc_freq: int = 5,
#     max_doc_frac: float = 0.4,
#     stopwords: Iterable[str] = DEFAULT_STOPWORDS,
# ) -> List[str]:
#     """Build Tier 1 vocabulary from top-k frequent non-stopword spaCy tokens.

#     Frequency is document frequency (how many texts contain the token).
#     """
#     # takes in a sequence of texts and the number of top tokens to return, as well as how many times a word needs to appaear to be included in the candidates and the maximum fraction of documents a word can appear in to be included in the candidates (if it appears in more than that fraction of documents, it is likely not a useful concept for distinguishing between different texts)
#     if top_k < 0:
#         raise ValueError("top_k must be >= 0")
#     if min_doc_freq < 1:
#         raise ValueError("min_doc_freq must be >= 1")
#     if not (0.0 < max_doc_frac <= 1.0):
#         raise ValueError("max_doc_frac must be in (0.0, 1.0]")

#     stopword_set = {w.strip().lower() for w in stopwords}
#     n_docs = len(texts)
#     max_doc_count = max(1, int(np.floor(max_doc_frac * n_docs)))

#     # going thru each token in each text
#     doc_freq: Counter[str] = Counter()
#     for toks in _token_sets(texts):
#         for tok in toks:
#             if tok in stopword_set:
#                 continue
#             if _is_pure_punctuation(tok):
#                 continue
#             doc_freq[tok] += 1

#     candidates = [
#         (w, c)
#         for w, c in doc_freq.items()
#         if c >= min_doc_freq and c <= max_doc_count
#     ]
#     candidates.sort(key=lambda x: (-x[1], x[0]))
#     return [w for w, _ in candidates[:top_k]]

def test_build_tier1_vocabulary_happy():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com."
    ]
    vocab = build_tier1_vocabulary(texts, min_doc_freq=1, max_doc_frac=1.0, top_k=5)
    assert isinstance(vocab, list)
    assert all(isinstance(w, str) for w in vocab)
    assert len(vocab) == 5
    # "this" appears in 2 texts, "i" appears in 2 texts, "and", "are", "at" appear once. So the top 5 by document frequency would be ["i", "this", "and", "are", "at"] (with ties broken alphabetically)
    # "." appears in all 3 texts but is a stopword, so it should be filtered out. 
    assert vocab == ["i", "this", "and", "are", "at"]

def test_build_tier1_vocabulary_top_k_zero():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com."
    ]
    vocab = build_tier1_vocabulary(texts, top_k=0)
    assert isinstance(vocab, list)
    assert len(vocab) == 0
    assert vocab == []

def test_build_tier1_vocabulary_min_doc_freq():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com."
    ]
    vocab = build_tier1_vocabulary(texts, min_doc_freq=2, max_doc_frac=1.0)
    assert isinstance(vocab, list)
    assert len(vocab) == 2
    assert set(vocab) == {"this", "i"}

def test_build_tier1_vocabulary_max_doc_frac():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com."
    ]

    vocab = build_tier1_vocabulary(texts, min_doc_freq=1, max_doc_frac=0.5)

    assert isinstance(vocab, list)
    assert "this" not in vocab
    assert "i" not in vocab

def test_build_tier1_vocabulary_invalid_top_k():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com."
    ]
    with pytest.raises(ValueError, match="top_k must be >= 0"):
        build_tier1_vocabulary(texts, top_k=-1)

def test_build_tier1_vocabulary_invalid_min_doc_freq():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com."
    ]
    with pytest.raises(ValueError, match="min_doc_freq must be >= 1"):
        build_tier1_vocabulary(texts, min_doc_freq=0)

def test_build_tier1_vocabulary_invalid_max_doc_frac():
    texts = [
        "I love this product.",
        "You are awful and I hate this.",
        "Email me at example@email.com."
    ]
    with pytest.raises(ValueError,match=r"max_doc_frac must be in \(0.0, 1.0\]"):
        build_tier1_vocabulary(texts, min_doc_freq=1, max_doc_frac=1.5)


# DEFAULT_STOPWORDS = {
#     "a",
#     "an",
#     "of",
#     "the",
#     ".",
#     ",",
# }

def test_build_tier1_vocabulary_stopword_punctuation_filter():
    texts = [
        "I love a product.",
        "You are an awful person, I hate all of this.",
        "The email of your kid; it sucks."
    ]
    vocab = build_tier1_vocabulary(texts, min_doc_freq=1, max_doc_frac=1.0)
    assert isinstance(vocab, list)
    assert "an" not in vocab
    assert "the" not in vocab
    assert "of" not in vocab
    assert "a" not in vocab
    assert "." not in vocab
    assert "," not in vocab
    assert ";" not in vocab

