
import os
import numpy as np

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

def main():
    
    # load up activations
    activations = np.load("/workspace/compexp_outputs_full/val_activations.npy")


    # load up tier 1 concept matrix
    tier1_concept_matrix = np.load("/workspace/compexp_outputs_full/conceptset_tier1/conceptset_tier1.npy")

    with open("/workspace/compexp_outputs_full/conceptset_tier1/conceptset_tier1_names.txt", "r", encoding="utf-8") as f:
      tier1_concept_names = [line.strip() for line in f]


    # os.makedirs(settings.RESULT, exist_ok=True)

    print("Computing quantiles")
    acts = quantile_features(activations)

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

        for concept_idx in range(tier1_concept_matrix.shape[1]):
            concept_vector = tier1_concept_matrix[:, concept_idx]
            neuron_vector = acts[:, neuron]
            iou_score = iou(concept_vector, neuron_vector)
            concept_name = tier1_concept_names[concept_idx].split("::")[1] if "::" in tier1_concept_names[concept_idx] else tier1_concept_names[concept_idx]
            # print(f"  Concept: {concept_name} ({concept_idx}), IoU: {iou_score}, lift: {lift(tier1_concept_matrix[:, concept_idx], acts[:, neuron])}, support: {support(tier1_concept_matrix[:, concept_idx], acts[:, neuron])}")
            score_cache[("leaf", concept_idx)] = {
                "formula": ("leaf", concept_idx),
                "iou": iou_score,
                "lift": lift(tier1_concept_matrix[:, concept_idx], acts[:, neuron]),
                "support": support(tier1_concept_matrix[:, concept_idx], acts[:, neuron]),
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
                  and_iou_score = iou(and_vector, acts[:, neuron])
                  # print(f"    AND with concept: {cand_concept_name} ({cand_concept_idx}), IoU: {and_iou_score}, lift: {lift(and_vector, acts[:, neuron])}, support: {support(and_vector, acts[:, neuron])}")

                  # then will add to beam regardless, will trim beam to top k later, and we will also want to keep track of the complexity of the explanation (e.g., how many concepts are combined) and potentially apply a penalty for more complex explanations to encourage simpler ones. We would also want to try OR and NOT combinations in a similar way, and keep track of the top combinations in our beam based on their iou scores with the target neuron activations, while also considering their complexity.
                  canonical_and_formula = canonicalize(("and", scores["formula"], ("leaf", cand_concept_idx)))
                  if canonical_and_formula not in score_cache:
                    new_beam.append({'formula': canonical_and_formula, 'iou': and_iou_score, 'lift': lift(and_vector, acts[:, neuron]), 'support': support(and_vector, acts[:, neuron]), 'mask': and_vector, 'complexity': formula_len})
                    score_cache[canonical_and_formula] = {
                        "formula": canonical_and_formula,
                        "iou": and_iou_score,
                        "lift": lift(and_vector, acts[:, neuron]),
                        "support": support(and_vector, acts[:, neuron]),
                        "mask": and_vector,
                        "complexity": formula_len,
                    }

                  # try OR combination
                  or_vector = scores["mask"] | tier1_concept_matrix[:, cand_concept_idx]
                  or_iou_score = iou(or_vector, acts[:, neuron])
                  # print(f"    OR with concept: {cand_concept_name} ({cand_concept_idx}), IoU: {or_iou_score}, lift: {lift(or_vector, acts[:, neuron])}, support: {support(or_vector, acts[:, neuron])}")
                  canonical_or_formula = canonicalize(("or", scores["formula"], ("leaf", cand_concept_idx)))
                  if canonical_or_formula not in score_cache:
                    new_beam.append({'formula': canonical_or_formula, 'iou': or_iou_score, 'lift': lift(or_vector, acts[:, neuron]), 'support': support(or_vector, acts[:, neuron]), 'mask': or_vector, 'complexity': formula_len})  # complexity of 2 for combining 2 concepts
                    score_cache[canonical_or_formula] = {
                        "formula": canonical_or_formula,
                        "iou": or_iou_score,
                        "lift": lift(or_vector, acts[:, neuron]),
                        "support": support(or_vector, acts[:, neuron]),
                        "mask": or_vector,
                        "complexity": formula_len,
                    }

                  # try NOT combination (negating the candidate concept and combining with AND)
                  not_vector = scores["mask"] & (~tier1_concept_matrix[:, cand_concept_idx])
                  not_iou_score = iou(not_vector, acts[:, neuron])
                  # print(f"    NOT with concept: {cand_concept_name} ({cand_concept_idx}), IoU: {not_iou_score}, lift: {lift(not_vector, acts[:, neuron])}, support: {support(not_vector, acts[:, neuron])}")
                  canonical_not_formula = canonicalize(("and", scores["formula"], ("not", ("leaf", cand_concept_idx))))
                  if canonical_not_formula not in score_cache:
                    new_beam.append({'formula': canonical_not_formula, 'iou': not_iou_score, 'lift': lift(not_vector, acts[:, neuron]), 'support': support(not_vector, acts[:, neuron]), 'mask': not_vector, 'complexity': formula_len})  # complexity of 2 for combining 2 concepts
                    score_cache[canonical_not_formula] = {
                        "formula": canonical_not_formula,
                        "iou": not_iou_score,
                        "lift": lift(not_vector, acts[:, neuron]),
                        "support": support(not_vector, acts[:, neuron]),
                        "mask": not_vector,
                        "complexity": formula_len,
                    }

          # trim the beam to the top k combinations based on iou score, while also considering complexity (e.g., we could apply a penalty to the iou score based on the complexity of the explanation to encourage simpler explanations)
          new_beam.sort(key=lambda x: x['iou'] * (settings.COMPLEXITY_PENALTY ** x['complexity']), reverse=True)  # sort by iou score with a penalty for complexity
          beam = new_beam[:settings.BEAM_SIZE]
        
        # after finishing the beam search, we would have a set of top compositional explanations for this neuron based on their iou scores with the target neuron activations, and we can save these explanations and their scores as part of our results for analysis and visualization in the sentence report. We can also analyze the final explanations to see which concepts are most commonly involved in high-iou explanations for this neuron, which can give us insights into what this neuron is responding to.
        print(f"Top explanations for neuron {neuron}:")
        for explanation in beam:
            formula = explanation['formula']
            iou_score = explanation['iou']
            lift_score = explanation['lift']
            support_score = explanation['support']
            complexity = explanation['complexity']
            print(f"  Explanation: {pretty_print_formula(formula, tier1_concept_names)}, IoU: {iou_score}, Lift: {lift_score}, Support: {support_score}, Complexity: {complexity}")
            # want to display concept names instead of indices in the explanation for better interpretability, so we can write a helper function to convert the formula with concept indices into a formula with concept names by looking up the concept names from the tier1_concept_names list using the indices. This way, we can have more interpretable explanations that indicate which concepts are involved in the explanation for the neuron activations.


    # map from concept and neuron to iou score, to find the overall highest iou concepts
    # top_concepts = {}

    # for neuron in range(activations.shape[1]):
        
    #     concept_iou_map = {}
    #     # print(f"Analyzing neuron {neuron}")

    #     # go through each column of the tier 1 concept matrix and compute the iou with the target neuron activations, and then sort the concepts by iou to find the top strongest concepts that we can use as a starting point for our beam search to find a compositional explanation that has high iou with the target neuron activations. We can save these top strong concepts and their ious as part of our results for analysis and visualization in the sentence report.

    #     for concept_idx in range(tier1_concept_matrix.shape[1]):
    #         concept_vector = tier1_concept_matrix[:, concept_idx]
    #         neuron_vector = acts[:, neuron]
    #         iou_score = iou(concept_vector, neuron_vector)
    #         # save the iou for this concept and neuron
    #         concept_iou_map[concept_idx] = iou_score

    #     # sort concepts by iou and print top concept for this neuron
    #     sorted_concepts = sorted(concept_iou_map.items(), key=lambda x: x[1], reverse=True)
    #     # print(f"  Top concept for neuron {neuron}:")
    #     for i in range(min(1, len(sorted_concepts))):
    #         concept_idx, score = sorted_concepts[i]
    #         concept_name = tier1_concept_names[concept_idx].split("::")[1] if "::" in tier1_concept_names[concept_idx] else tier1_concept_names[concept_idx]
    #         # print(f"    {concept_name} ({concept_idx}): {score}")
    #         top_concepts[(neuron, concept_name, concept_idx)] = score

    # # print 100 top concepts overall by iou
    # sorted_top_concepts = sorted(top_concepts.items(), key=lambda x: x[1], reverse=True)
    # print("Top concepts overall by IoU:")
    # for i in range(min(100, len(sorted_top_concepts))):
    #     (neuron, concept_name, concept_idx), score = sorted_top_concepts[i]
    #     print(f"  Neuron {neuron}, Concept: {concept_name}, IoU: {score}, lift: {lift(tier1_concept_matrix[:, concept_idx], acts[:, neuron])}, support: {support(tier1_concept_matrix[:, concept_idx], acts[:, neuron])}")

if __name__ == "__main__":
    main()
