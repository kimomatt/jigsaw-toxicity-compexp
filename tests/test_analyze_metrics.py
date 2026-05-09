import numpy as np
import pytest

from compexp_toxicity.compexp.analyze import iou, lift, support, extract_concept_indices, canonicalize, get_quantiles, quantile_features, binarize_activations, compute_activation_intervals
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

# need to test that the activation intervals functions work correctly, and that the binarization based on intervals also works correctly, maybe use a simple set of activation values and known intervals to verify the output is correct

# def compute_activation_intervals(neuron_values, num_clusters):
#     # if num clusters is 1, then maybe have it default to activation thresholding based on quantiles, and then if num clusters is greater than 1, we can do k-means clustering on the activations to find clusters of activation values, and then we can compute the min and max activation value for each cluster to define the intervals. This way, we can capture more complex patterns in the activations beyond just a single threshold, which can help us identify more nuanced explanations for the neuron activations.
#     if num_clusters <= 1:
#         # shouldn't call this function with num_clusters = 1, raise an error if that happens since we want to make sure we're not accidentally using activation thresholding when we meant to be doing clustering, and vice versa. If num_clusters is 1, that means we're just doing activation thresholding based on quantiles, so we shouldn't be calling this function at all since it's meant for computing intervals based on clustering.
#         raise ValueError("num_clusters must be greater than 1 for compute_activation_intervals")
    
#     if len(neuron_values) == 0:
#         raise ValueError("No activation values provided to compute_activation_intervals")

#     # if num clusters is greater than number of inputs we should set num clusters to number of inputs, add logging
#     if num_clusters > len(neuron_values):
#         num_clusters = len(np.unique(neuron_values))
#         print(f"Warning: num_clusters is greater than number of inputs, setting num_clusters to {num_clusters}")
    
#     values = np.asarray(neuron_values).reshape(-1, 1)  # reshape to 2D array for k-means, -1 tells numpy to figure out this dimension (number of rows) based on the number of columns which we set

#     clusters = KMeans(n_clusters=num_clusters, random_state=0).fit(values)
#     labels = clusters.labels_

#     activation_ranges = []
#     for cluster_id in range(num_clusters):
#         cluster_values = values[labels == cluster_id]
#         activation_ranges.append((float(cluster_values.min()), float(cluster_values.max())))

#     activation_ranges.sort(key=lambda r: r[0])
#     return activation_ranges

def test_compute_activation_intervals_happy():
    neuron_values = [1, 2, 5, 6, 10, 11]
    num_clusters = 3
    result = compute_activation_intervals(neuron_values, num_clusters)
    assert len(result) == num_clusters
    assert all(isinstance(interval, tuple) and len(interval) == 2 for interval in result)
    # check that the intervals are sorted by their lower bound
    assert all(result[i][0] <= result[i+1][0] for i in range(len(result)-1))
    assert result == [(1.0, 2.0), (5.0, 6.0), (10.0, 11.0)]

def test_compute_activation_intervals_num_clusters_greater_than_inputs():
    neuron_values = [1, 2]
    num_clusters = 5
    result = compute_activation_intervals(neuron_values, num_clusters)
    assert len(result) == 2  # should be set to number of unique inputs
    assert result == [(1.0, 1.0), (2.0, 2.0)]

def test_compute_activation_intervals_num_clusters_one():
    neuron_values = [1, 2, 3]
    num_clusters = 1
    with pytest.raises(ValueError, match="num_clusters must be greater than 1 for compute_activation_intervals"):
        compute_activation_intervals(neuron_values, num_clusters)

def test_compute_activation_intervals_no_values():
    neuron_values = []
    num_clusters = 3
    with pytest.raises(ValueError, match="No activation values provided to compute_activation_intervals"):
        compute_activation_intervals(neuron_values, num_clusters)

def test_compute_activation_intervals_caps_clusters_by_unique_values():
    neuron_values = [1, 1, 1, 5, 5, 5]
    result = compute_activation_intervals(neuron_values, 3)
    assert result == [(1.0, 1.0), (5.0, 5.0)]


# then function to binarize based on whether the activation falls into the chosen interval, should be very similar to quantile features, takes in one chosen interval, will loop over intervals and call this function w each interval
# def binarize_activations(activation_values, interval):
#     # make sure that valid interval and set of activation values are provided, then convert the activation values to a numpy array and return a binary vector indicating whether each activation value falls within the specified interval (inclusive). This allows us to create binary features based on the activation intervals we computed, which can be useful for analyzing the relationship between neuron activations and concepts in our compositional explanations.
#     if interval is None or len(interval) != 2 or interval[0] > interval[1]:
#         raise ValueError("Invalid interval provided to binarize_activations")
#     if len(activation_values) == 0:
#         raise ValueError("No activation values provided to binarize_activations")

#     lower, upper = interval
#     values = np.asarray(activation_values)
#     return (values >= lower) & (values <= upper)

def test_binarize_activations_happy():
    activation_values = [1, 2, 3, 4, 5]
    interval = (2, 4)
    result = binarize_activations(activation_values, interval)
    expected = np.array([False, True, True, True, False])
    assert np.array_equal(result, expected)

def test_binarize_activations_invalid_interval():
    activation_values = [1, 2, 3, 4, 5]
    interval = (4, 2)  # Invalid interval
    with pytest.raises(ValueError, match="Invalid interval provided to binarize_activations"):
        binarize_activations(activation_values, interval)

def test_binarize_activations_no_values():
    activation_values = []
    interval = (2, 4)
    with pytest.raises(ValueError, match="No activation values provided to binarize_activations"):
        binarize_activations(activation_values, interval)