import numpy as np
import pytest

from compexp_toxicity.compexp.analyze import iou, lift, support, extract_concept_indices, canonicalize, get_quantiles, quantile_features
from compexp_toxicity.compexp import settings



def test_iou_returns_expected_overlap_ratio():
    # Arrange: overlap at index 2, union at indices 0, 1, 2.
    a = np.array([1, 0, 1, 0], dtype=bool)
    b = np.array([0, 1, 1, 0], dtype=bool)

    # Act
    result = iou(a, b)

    # Assert: intersection = 1, union = 3.
    assert np.isclose(result, 1 / 3)


def test_iou_returns_zero_when_both_vectors_are_all_zero():
    a = np.array([0, 0, 0], dtype=bool)
    b = np.array([0, 0, 0], dtype=bool)

    result = iou(a, b)

    assert result == 0.0


def test_support_returns_joint_positive_rate():
    # Both are positive only at indices 0 and 3.
    a = np.array([1, 1, 0, 1], dtype=bool)
    b = np.array([1, 0, 1, 1], dtype=bool)

    result = support(a, b)

    assert np.isclose(result, 2 / 4)


def test_lift_matches_hand_computed_value():
    # p(a) = 2/4, p(b) = 2/4, p(a and b) = 1/4, so lift = 1.
    a = np.array([1, 1, 0, 0], dtype=bool)
    b = np.array([1, 0, 1, 0], dtype=bool)

    result = lift(a, b)

    assert np.isclose(result, 1.0)
    

def test_extract_concept_indices():
    formula = ("or", ("and", ("leaf", 1), ("not", ("leaf", 2))), ("leaf", 3))
    expected_indices = {1, 2, 3}

    result = extract_concept_indices(formula)

    assert result == expected_indices

def test_canonicalize_double_negation_elimination():
    formula = ("not", ("not", ("leaf", 1)))
    expected = ("leaf", 1)

    result = canonicalize(formula)
    assert result == expected

def test_canonicalize_dedupes_repeated_terms():
    formula = ("and", ("leaf", 1), ("and", ("leaf", 2), ("leaf", 1)))
    expected = ("and", ("leaf", 1), ("leaf", 2))

    result = canonicalize(formula)
    assert result == expected

def test_canonicalize_simplifies_single_item_after_dedupe():
    formula = ("and", ("leaf", 1), ("leaf", 1))
    expected = ("leaf", 1)

    result = canonicalize(formula)
    assert result == expected 

def test_canonicalize_or_commutativity():
    formula1 = ("or", ("leaf", 1), ("leaf", 2))
    formula2 = ("or", ("leaf", 2), ("leaf", 1))

    result1 = canonicalize(formula1)
    result2 = canonicalize(formula2)

    assert result1 == result2

def test_canonicalize_and_commutativity():
    formula1 = ("and", ("leaf", 1), ("leaf", 2))
    formula2 = ("and", ("leaf", 2), ("leaf", 1))

    result1 = canonicalize(formula1)
    result2 = canonicalize(formula2)

    assert result1 == result2

def test_canonicalize_associativity():
    formula1 = ("or", ("leaf", 1), ("or", ("leaf", 2), ("leaf", 3)))
    formula2 = ("or", ("or", ("leaf", 1), ("leaf", 2)), ("leaf", 3))

    result1 = canonicalize(formula1)
    result2 = canonicalize(formula2)

    assert result1 == result2

def test_canonicalize_raises_on_unknown_formula_type():
    formula = ("xor", ("leaf", 1), ("leaf", 2))

    with pytest.raises(ValueError, match="Unknown formula type: xor"):
        canonicalize(formula)

def test_get_quantiles():
    feats = np.array([[1, 2, 3], [4, 5, 6], [70, 80, 90]])
    alpha = 0.5

    quantiles = get_quantiles(feats, alpha)

    expected_quantiles = np.array([4, 5, 6])
    assert np.allclose(quantiles, expected_quantiles)


def test_quantile_features_with_alpha(monkeypatch):
    feats = np.array([[1, 2, 3], [4, 5, 6], [70, 80, 90]])
    monkeypatch.setattr(settings, 'ALPHA', 0.5)

    expected = np.array([[False, False, False], [False, False, False], [True, True, True]])
    result = quantile_features(feats)

    assert np.array_equal(result, expected)

def test_quantile_features_with_alpha_none(monkeypatch):
    feats = np.array([[-1, 2, 3], [4, 5, 6], [-70, 80, 90]])
    monkeypatch.setattr(settings, 'ALPHA', None)

    expected = np.array([[False, True, True], [True, True, True], [False, True, True]])
    result = quantile_features(feats)

    assert np.array_equal(result, expected)