
import argparse
import json
import os
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans


from compexp_toxicity.compexp import settings

def get_quantiles(feats, alpha):
    # goes thru each column (feature) of the activation matrix feats and computes the quantile at 1 - alpha for that feature across all examples. This gives us a threshold value for each feature such that only the top alpha fraction of activations for that feature will be above the threshold. We can then use these quantiles to create binary features that indicate whether each activation is above its respective quantile threshold, which can help us focus on the most significant activations for our analysis.
    quantiles = np.apply_along_axis(lambda a: np.quantile(a, 1 - alpha), 0, feats)
    return quantiles


def quantile_features(feats):
    if settings.ALPHA is None:
        return np.stack(feats) > 0

    # generating binary features by comparing each activation to its respective quantile threshold, resulting in a binary matrix where each entry is 1 if the activation is above the threshold and 0 otherwise. This allows us to focus on the most significant activations for our analysis.
    quantiles = get_quantiles(feats, settings.ALPHA)
    # use np.newaxis to add a new axis to the quantiles array so that it can be broadcasted correctly when comparing to the feats matrix. This way, we are comparing each activation in feats to the corresponding quantile threshold for that feature across all examples.
    return feats > quantiles[np.newaxis]

# want to build a function to cluster the activations into intervals, return intervals
# code in Mattia's repo:
# def build_ranges_from_clusters(
#         activations: torch.Tensor, clusters: List[int],
#         num_clusters: int) -> List[tuple]:
#     """Build activation ranges from clusters.

#     Args:
#         activations (torch.Tensor): Activations of the unit.
#         clusters (List[int]): Clusters indexes of the activations.
#         num_clusters (int): Number of clusters.

#     Returns:
#         activation_ranges (List[tuple]): Activation ranges for each cluster.
#     """

#     activations_ranges = []
#     for label in range(num_clusters):
#         cluster_activations = activations[clusters == label]
#         lower_bound = torch.min(cluster_activations)
#         upper_bound = torch.max(cluster_activations)
#         activations_ranges.append((lower_bound.item(), upper_bound.item()))
#     return activations_ranges
# def compute_activation_ranges(
#         activations: torch.Tensor, num_clusters: int) -> List[Tuple]:
#     """Compute activation ranges for each unit.

#     Args:
#         activations (torch.Tensor): Activations of the unit.
#         num_clusters (int): Number of clusters.
#         algorithm (str): Algorithm to use for clustering.

#     Returns:
#         activation_ranges (List[tuple]): Activation ranges for each unit.
#     """
#     if num_clusters == 1:
#         # Case vanilla compositional and netdissect range
#         # Avoid zero is set to false like in the compositional paper
#         threshold = quantile_threshold(
#             activations, quantile=C.NETDISSECT_QUANTILE, avoid_zero=False
#         )
#         activation_ranges = [(threshold, torch.tensor(float("inf")))]
#     else:
#         activations = activations.reshape(-1, 1)
#         # Remove zeros from activations if there is a relu activation
#         if torch.all(activations >= 0):
#             activations = activations[activations > 0]
#             activations = activations.reshape(-1, 1)
#         # Compute activation ranges
#         clusters = scikit_cluster.KMeans(
#             n_clusters=num_clusters, random_state=0
#             ).fit(activations)
#         activation_ranges = build_ranges_from_clusters(
#             activations, clusters.labels_, num_clusters)
#     return activation_ranges
def compute_activation_intervals(neuron_values, num_clusters):
    # if num clusters is 1, then maybe have it default to activation thresholding based on quantiles, and then if num clusters is greater than 1, we can do k-means clustering on the activations to find clusters of activation values, and then we can compute the min and max activation value for each cluster to define the intervals. This way, we can capture more complex patterns in the activations beyond just a single threshold, which can help us identify more nuanced explanations for the neuron activations.
    if num_clusters <= 1:
        # shouldn't call this function with num_clusters = 1, raise an error if that happens since we want to make sure we're not accidentally using activation thresholding when we meant to be doing clustering, and vice versa. If num_clusters is 1, that means we're just doing activation thresholding based on quantiles, so we shouldn't be calling this function at all since it's meant for computing intervals based on clustering.
        raise ValueError("num_clusters must be greater than 1 for compute_activation_intervals")
    
    if len(neuron_values) == 0:
        raise ValueError("No activation values provided to compute_activation_intervals")

    # if num clusters is greater than number of inputs we should set num clusters to number of inputs, add logging
    if num_clusters > len(np.unique(neuron_values)):
        num_clusters = len(np.unique(neuron_values))
        print(f"Warning: num_clusters is greater than number of unique inputs, setting num_clusters to {num_clusters}")
    
    values = np.asarray(neuron_values).reshape(-1, 1)  # reshape to 2D array for k-means, -1 tells numpy to figure out this dimension (number of rows) based on the number of columns which we set

    clusters = KMeans(n_clusters=num_clusters, random_state=0).fit(values)
    labels = clusters.labels_

    activation_ranges = []
    for cluster_id in range(num_clusters):
        cluster_values = values[labels == cluster_id]
        activation_ranges.append((float(cluster_values.min()), float(cluster_values.max())))

    activation_ranges.sort(key=lambda r: r[0])
    return activation_ranges


# then function to binarize based on whether the activation falls into the chosen interval, should be very similar to quantile features, takes in one chosen interval, will loop over intervals and call this function w each interval
def binarize_activations(activation_values, interval):
    # make sure that valid interval and set of activation values are provided, then convert the activation values to a numpy array and return a binary vector indicating whether each activation value falls within the specified interval (inclusive). This allows us to create binary features based on the activation intervals we computed, which can be useful for analyzing the relationship between neuron activations and concepts in our compositional explanations.
    if interval is None or len(interval) != 2 or interval[0] > interval[1]:
        raise ValueError("Invalid interval provided to binarize_activations")
    if len(activation_values) == 0:
        raise ValueError("No activation values provided to binarize_activations")

    lower, upper = interval
    values = np.asarray(activation_values)
    return (values >= lower) & (values <= upper)


def iou(a, b):
    # intersection is the number of positions where both a and b are 1, and union is the number of positions where either a or b is 1. The IoU is then computed as the intersection divided by the union, which gives us a measure of how well the concept represented by vector a overlaps with the activations represented by vector b. A higher IoU indicates a stronger correlation between the concept and the neuron activations.
    intersection = (a & b).sum()
    union = (a | b).sum()

    # adding a tiny value to the denominator to prevent division by zero in case both a and b are all zeros (i.e., no activations), which would result in an undefined IoU. This way, if both a and b are all zeros, the IoU will be defined as 0 instead of causing an error.
    return intersection / (union + np.finfo(np.float32).tiny)

def lift(a, b):
    # lift is computed as the ratio of the joint probability of a and b being 1 (i.e., both the concept and the neuron are active) to the product of their individual probabilities of being 1. This gives us a measure of how much more likely it is for the concept and the neuron to be active together than we would expect if they were independent. A lift value greater than 1 indicates a positive association between the concept and the neuron, while a value less than 1 indicates a negative association.
    p_a = np.mean(a)
    p_b = np.mean(b)
    p_ab = np.mean(a & b)

    # adding a tiny value to the denominator to prevent division by zero in case either p_a or p_b is zero (i.e., if either the concept or the neuron is never active), which would result in an undefined lift. This way, if either p_a or p_b is zero, the lift will be defined as 0 instead of causing an error.
    return p_ab / (p_a * p_b + np.finfo(np.float32).tiny)

def support(a, b):
    # support is simply the joint probability of a and b being 1 (i.e., both the concept and the neuron are active), which gives us a measure of how frequently the concept and the neuron are active together in the dataset. A higher support indicates that the concept and the neuron co-occur more frequently, which can be an important factor to consider alongside measures like IoU and lift when evaluating potential explanations.
    return np.mean(a & b)

def extract_concept_indices(formula):
    if formula[0] == "leaf":
        return {formula[1]}
    elif formula[0] in {"and", "or"}:
        return extract_concept_indices(formula[1]) | extract_concept_indices(formula[2])
    elif formula[0] == "not":
        return extract_concept_indices(formula[1])
    else:
        raise ValueError(f"Unknown formula type: {formula[0]}")
    
def _gather_same_kind(kind, formula):
    if formula[0] != kind:
        return [formula]
    else:
        return _gather_same_kind(kind, formula[1]) + _gather_same_kind(kind, formula[2])

def _rebuild_binary(kind, items):
    out = items[0]
    for item in items[1:]:
        out = (kind, out, item)
    return out

    
def canonicalize(formula):
    kind = formula[0]

    if kind == "leaf":
        return formula
    
    # accounts for double negation
    if kind == "not":
        child = canonicalize(formula[1])
        if child[0] == "not":
            return child[1]
        return ("not", child)
    
    if kind in {"and", "or"}:
        items = _gather_same_kind(kind, formula)
        items = [canonicalize(item) for item in items]
        items.sort()

        # deduping
        deduped = []
        for item in items:
            if not deduped or item != deduped[-1]:
                deduped.append(item)
        items = deduped

        if len(items) == 1:
            return items[0]
        
        return _rebuild_binary(kind, items)


    raise ValueError(f"Unknown formula type: {kind}")

def pretty_print_formula(formula, concept_names):
    # formulas made of indices, want to display the actual concept names, which we can get from the tier 1 concept names list using the index from the leaf nodes in the formula. This function recursively traverses the formula tree and constructs a human-readable string representation of the formula, using parentheses to indicate the structure of the formula and operators like AND, OR, and NOT to indicate how the concepts are combined.
    kind = formula[0]
    if kind == "leaf":
        concept_name = concept_names[formula[1]].split("::")[1] if "::" in concept_names[formula[1]] else concept_names[formula[1]]
        return f"{concept_name}"
    if kind == "not":
        return f"NOT({pretty_print_formula(formula[1], concept_names)})"
    if kind in {"and", "or"}:
        left = pretty_print_formula(formula[1], concept_names)
        right = pretty_print_formula(formula[2], concept_names)
        op = " AND " if kind == "and" else " OR "
        return f"({left}{op}{right})"
    raise ValueError(f"Unknown formula type: {kind}")

# this assumes that the extract last token activations script as well as the make tier1 concept matrix script (after) have already been run, since it needs the extracted features and the tier1 concept matrix to do the mask search and then visualize the features in the sentence report

# so we shouldn't have to do model or vocab loading really, we just need to go from the matrices we have to the masks

# from activation extraction we should have a set of neuron activations for each example in the dataset

# in analysis we will choose one neuron at a time that we want to generate a compositional explanation for
# so off of that matrix we will extract one column and that is like our target that hopefully we are able to find a good compositional explanation for through our beam search

# from the tier 1 concept matrix for each example in the dataset we have a binary vector saying whether that example has each of the tier 1 concepts or not

# go through each respective column of the tier 1 concept matrix and see which one has the highest iou with the target neuron activations, and then we can use that as a starting point for our beam search to find a compositional explanation that has high iou with the target neuron activations

def run_analysis(path_to_activations: Path, path_to_concept_matrix: Path, path_to_concept_names: Path, result_dir: Path):
    results = []
    
    # load up activations
    activations = np.load(path_to_activations)


    # load up tier 1 concept matrix
    tier1_concept_matrix = np.load(path_to_concept_matrix)
    tier1_concept_matrix = tier1_concept_matrix.astype(bool)

    with open(path_to_concept_names, "r", encoding="utf-8") as f:
      tier1_concept_names = [line.strip() for line in f]


    # os.makedirs(settings.RESULT, exist_ok=True)

    # print("Computing quantiles")
    # acts = quantile_features(activations)

    # at this point we can start doing the search for each chosen neuron, and then we can save the results and also visualize them in the sentence report

    # for now we can just print out for each neuron each tier 1 concept iou iou
    # also wanna print like the top 10 concepts by iou for each neuron, and then we can use those as starting points for our beam search to find compositional explanations that have high iou with the target neuron activations, and we can save those top correlated concepts and their ious as part of our results for analysis and visualization in the sentence report

    


    for neuron in settings.NEURONS:
        print(f"Analyzing neuron {neuron}")
        # score_cache[key] = {
        #     "key": ...,
        #     "iou": ...,
        #     "lift": ...,
        #     "support": ...,
        #     "mask": ...
        # }
        score_cache = {}

        intervals = compute_activation_intervals(activations[:, neuron], settings.NUM_CLUSTERS)
        for interval in intervals:
            neuron_vector = binarize_activations(activations[:, neuron], interval)

            # reset score_cache and beam for each interval
            score_cache = {}
            beam = []


            for concept_idx in range(tier1_concept_matrix.shape[1]):
                concept_vector = tier1_concept_matrix[:, concept_idx]
                iou_score = iou(concept_vector, neuron_vector)
                concept_name = tier1_concept_names[concept_idx].split("::")[1] if "::" in tier1_concept_names[concept_idx] else tier1_concept_names[concept_idx]
                # print(f"  Concept: {concept_name} ({concept_idx}), IoU: {iou_score}, lift: {lift(tier1_concept_matrix[:, concept_idx], acts[:, neuron])}, support: {support(tier1_concept_matrix[:, concept_idx], acts[:, neuron])}")
                score_cache[("leaf", concept_idx)] = {
                    "formula": ("leaf", concept_idx),
                    "iou": iou_score,
                    "lift": lift(tier1_concept_matrix[:, concept_idx], neuron_vector),
                    "support": support(tier1_concept_matrix[:, concept_idx], neuron_vector),
                    "mask": concept_vector,
                    "complexity": 1,  # complexity of 1 for individual concepts
                }
            
            # sort concepts by iou and print top concept for this neuron
            sorted_concepts = sorted(score_cache.values(), key=lambda x: x["iou"], reverse=True)
            # trim to beam size
            beam = sorted_concepts[:settings.BEAM_SIZE]
            # beam now looks like [{'formula': ('leaf', concept_idx), 'iou': iou_score, 'lift': lift_score, 'support': support_score, 'mask': concept_vector, 'complexity': 1}, ...] for the top concepts based on iou with the target neuron activations, and we can use this as a starting point for our beam search to find compositional explanations that have high iou with the target neuron activations. We can save these top concepts and their scores as part of our results for analysis and visualization in the sentence report.

            for formula_len in range(2, settings.MAX_FORMULA_LENGTH + 1):
              new_beam = beam.copy()

              # begin beam search for compositional explanations starting from these top concepts, and save results for analysis and visualization in the sentence report
              for scores in beam:
                  formula = scores["formula"]
                  used_concept_indices = extract_concept_indices(formula)

                  # go through every other concept and combine it with the starting concept using AND, OR, NOT to see if we can get a higher iou with the target neuron activations, and keep track of the top combinations in our beam. We would also want to consider the complexity of the explanations (e.g., how many concepts are combined) and potentially apply a penalty for more complex explanations to encourage simpler ones.

                  # want canonical ordering to avoid duplicates
                  for cand_concept_idx in range(tier1_concept_matrix.shape[1]):
                      if cand_concept_idx in used_concept_indices:
                          continue
                      cand_concept_name = tier1_concept_names[cand_concept_idx].split("::")[1] if "::" in tier1_concept_names[cand_concept_idx] else tier1_concept_names[cand_concept_idx]
                      # try AND combination
                      and_vector = scores["mask"] & tier1_concept_matrix[:, cand_concept_idx]
                      and_iou_score = iou(and_vector, neuron_vector)
                      # print(f"    AND with concept: {cand_concept_name} ({cand_concept_idx}), IoU: {and_iou_score}, lift: {lift(and_vector, neuron_vector)}, support: {support(and_vector, neuron_vector)}")

                      # then will add to beam regardless, will trim beam to top k later, and we will also want to keep track of the complexity of the explanation (e.g., how many concepts are combined) and potentially apply a penalty for more complex explanations to encourage simpler ones. We would also want to try OR and NOT combinations in a similar way, and keep track of the top combinations in our beam based on their iou scores with the target neuron activations, while also considering their complexity.
                      canonical_and_formula = canonicalize(("and", scores["formula"], ("leaf", cand_concept_idx)))
                      if canonical_and_formula not in score_cache:
                        new_beam.append({'formula': canonical_and_formula, 'iou': and_iou_score, 'lift': lift(and_vector, neuron_vector), 'support': support(and_vector, neuron_vector), 'mask': and_vector, 'complexity': formula_len})
                        score_cache[canonical_and_formula] = {
                            "formula": canonical_and_formula,
                            "iou": and_iou_score,
                            "lift": lift(and_vector, neuron_vector),
                            "support": support(and_vector, neuron_vector),
                            "mask": and_vector,
                            "complexity": formula_len,
                        }

                      # try OR combination
                      or_vector = scores["mask"] | tier1_concept_matrix[:, cand_concept_idx]
                      or_iou_score = iou(or_vector, neuron_vector)
                      # print(f"    OR with concept: {cand_concept_name} ({cand_concept_idx}), IoU: {or_iou_score}, lift: {lift(or_vector, neuron_vector)}, support: {support(or_vector, neuron_vector)}")
                      canonical_or_formula = canonicalize(("or", scores["formula"], ("leaf", cand_concept_idx)))
                      if canonical_or_formula not in score_cache:
                        new_beam.append({'formula': canonical_or_formula, 'iou': or_iou_score, 'lift': lift(or_vector, neuron_vector), 'support': support(or_vector, neuron_vector), 'mask': or_vector, 'complexity': formula_len})  # complexity of 2 for combining 2 concepts
                        score_cache[canonical_or_formula] = {
                            "formula": canonical_or_formula,
                            "iou": or_iou_score,
                            "lift": lift(or_vector, neuron_vector),
                            "support": support(or_vector, neuron_vector),
                            "mask": or_vector,
                            "complexity": formula_len,
                        }

                      # try NOT combination (negating the candidate concept and combining with AND)
                      not_vector = scores["mask"] & (~tier1_concept_matrix[:, cand_concept_idx])
                      not_iou_score = iou(not_vector, neuron_vector)
                      # print(f"    NOT with concept: {cand_concept_name} ({cand_concept_idx}), IoU: {not_iou_score}, lift: {lift(not_vector, neuron_vector)}, support: {support(not_vector, neuron_vector)}")
                      canonical_not_formula = canonicalize(("and", scores["formula"], ("not", ("leaf", cand_concept_idx))))
                      if canonical_not_formula not in score_cache:
                        new_beam.append({'formula': canonical_not_formula, 'iou': not_iou_score, 'lift': lift(not_vector, neuron_vector), 'support': support(not_vector, neuron_vector), 'mask': not_vector, 'complexity': formula_len})  # complexity of 2 for combining 2 concepts
                        score_cache[canonical_not_formula] = {
                            "formula": canonical_not_formula,
                            "iou": not_iou_score,
                            "lift": lift(not_vector, neuron_vector),
                            "support": support(not_vector, neuron_vector),
                            "mask": not_vector,
                            "complexity": formula_len,
                        }

              # trim the beam to the top k combinations based on iou score, while also considering complexity (e.g., we could apply a penalty to the iou score based on the complexity of the explanation to encourage simpler explanations)
              new_beam.sort(key=lambda x: x['iou'] * (settings.COMPLEXITY_PENALTY ** x['complexity']), reverse=True)  # sort by iou score with a penalty for complexity
              beam = new_beam[:settings.BEAM_SIZE]
            
            # after finishing the beam search, we would have a set of top compositional explanations for this neuron based on their iou scores with the target neuron activations, and we can save these explanations and their scores as part of our results for analysis and visualization in the sentence report. We can also analyze the final explanations to see which concepts are most commonly involved in high-iou explanations for this neuron, which can give us insights into what this neuron is responding to.
            print(f"Top explanations for neuron {neuron} for interval {interval}:")
            for explanation in beam:
                formula = explanation['formula']
                iou_score = explanation['iou']
                lift_score = explanation['lift']
                support_score = explanation['support']
                complexity = explanation['complexity']
                print(f"  Explanation: {pretty_print_formula(formula, tier1_concept_names)}, IoU: {iou_score}, Lift: {lift_score}, Support: {support_score}, Complexity: {complexity}")
                # want to display concept names instead of indices in the explanation for better interpretability, so we can write a helper function to convert the formula with concept indices into a formula with concept names by looking up the concept names from the tier1_concept_names list using the indices. This way, we can have more interpretable explanations that indicate which concepts are involved in the explanation for the neuron activations.

            results.append({
              "neuron": neuron,
              "interval": [float(interval[0]), float(interval[1])],
              "num_examples_in_interval": int(neuron_vector.sum()),
              "top_explanations": [
                  {
                      "formula": explanation["formula"],
                      "pretty_formula": pretty_print_formula(explanation["formula"], tier1_concept_names),
                      "iou": float(explanation["iou"]),
                      "lift": float(explanation["lift"]),
                      "support": float(explanation["support"]),
                      "complexity": int(explanation["complexity"]),
                  }
                  for explanation in beam
              ],
          })

    os.makedirs(result_dir, exist_ok=True)
    with open(result_dir / "interval_analysis.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "neurons": settings.NEURONS,
                    "num_clusters": settings.NUM_CLUSTERS,
                    "beam_size": settings.BEAM_SIZE,
                    "max_formula_length": settings.MAX_FORMULA_LENGTH,
                    "complexity_penalty": settings.COMPLEXITY_PENALTY,
                },
                "results": results,
            },
            f,
            indent=2,
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Run analysis for compositional explanations")
    parser.add_argument("--path_to_activations", type=Path, help="Path to the extracted activations numpy file", default="/workspace/compexp_outputs_full/val_activations.npy")
    parser.add_argument("--path_to_concept_matrix", type=Path, help="Path to the tier 1 concept matrix numpy file", default="/workspace/compexp_outputs_full/conceptset_tier1/conceptset_tier1.npy")
    parser.add_argument("--path_to_concept_names", type=Path, help="Path to the tier 1 concept names text file", default="/workspace/compexp_outputs_full/conceptset_tier1/conceptset_tier1_names.txt")
    parser.add_argument("--result_dir", type=Path, help="Directory to save the analysis results", default=settings.RESULT)
    return parser.parse_args()

def main():

    args = parse_args()
    run_analysis(args.path_to_activations, args.path_to_concept_matrix, args.path_to_concept_names, args.result_dir)
    
    

    # map from concept and neuron to iou score, to find the overall highest iou concepts

    activations = np.load(args.path_to_activations)

    tier1_concept_matrix = np.load(args.path_to_concept_matrix)
    tier1_concept_matrix = tier1_concept_matrix.astype(bool)

    with open(args.path_to_concept_names, "r", encoding="utf-8") as f:
        tier1_concept_names = [line.strip() for line in f]

    # map from neuron and concept to iou score, to find the overall highest iou concepts across all neurons
    top_concepts = {}
    for neuron in range(activations.shape[1]):
        concept_iou_map = {}
        print(f"Analyzing neuron {neuron}")
        intervals = compute_activation_intervals(activations[:, neuron], settings.NUM_CLUSTERS)


        for interval in intervals:
            neuron_vector = binarize_activations(activations[:, neuron], interval)
            # go through each column of the tier 1 concept matrix and compute the iou with the target neuron activations, and then sort the concepts by iou to find the top strongest concepts that we can use as a starting point for our beam search to find a compositional explanation that has high iou with the target neuron activations. We can save these top strong concepts and their ious as part of our results for analysis and visualization in the sentence report.
            for concept_idx in range(tier1_concept_matrix.shape[1]):
                concept_vector = tier1_concept_matrix[:, concept_idx]
                iou_score = iou(concept_vector, neuron_vector)
                # save the best IoU and winning interval for this concept on this neuron
                if concept_idx not in concept_iou_map or iou_score > concept_iou_map[concept_idx]["iou"]:
                    concept_iou_map[concept_idx] = {
                        "iou": iou_score,
                        "interval": interval,
                    }

        # sort concepts by iou and print top concept for this neuron
        sorted_concepts = sorted(concept_iou_map.items(), key=lambda x: x[1]["iou"], reverse=True)
        # print(f"  Top concept for neuron {neuron}:")
        for i in range(min(1, len(sorted_concepts))):
            concept_idx, best = sorted_concepts[i]
            concept_name = tier1_concept_names[concept_idx].split("::")[1] if "::" in tier1_concept_names[concept_idx] else tier1_concept_names[concept_idx]
            top_concepts[(neuron, concept_name, concept_idx)] = best
    
    # sort top concepts across all neurons by iou score and print top 10 overall
    sorted_top_concepts = sorted(top_concepts.items(), key=lambda x: x[1]["iou"], reverse=True)
    print(f"Top concepts across all neurons:")
    for i in range(min(10, len(sorted_top_concepts))):
        (neuron, concept_name, concept_idx), best = sorted_top_concepts[i]
        interval = best["interval"]
        print(
            f"  Neuron {neuron}, Concept: {concept_name} ({concept_idx}), "
            f"IoU: {best['iou']}, Interval: ({interval[0]}, {interval[1]})"
        )

if __name__ == "__main__":
    main()
