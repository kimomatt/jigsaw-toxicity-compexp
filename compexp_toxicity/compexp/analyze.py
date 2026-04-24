
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
    intersection = (a & b).sum()
    union = (a | b).sum()

    # adding a tiny value to the denominator to prevent division by zero in case both a and b are all zeros (i.e., no activations), which would result in an undefined IoU. This way, if both a and b are all zeros, the IoU will be defined as 0 instead of causing an error.
    return intersection / (union + np.finfo(np.float32).tiny)

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


    # for neuron in settings.NEURONS:

    for neuron in range(activations.shape[1]):
        
        concept_iou_map = {}
        print(f"Analyzing neuron {neuron}")

        # go through each column of the tier 1 concept matrix and compute the iou with the target neuron activations, and then sort the concepts by iou to find the top strongest concepts that we can use as a starting point for our beam search to find a compositional explanation that has high iou with the target neuron activations. We can save these top strong concepts and their ious as part of our results for analysis and visualization in the sentence report.

        for concept_idx in range(tier1_concept_matrix.shape[1]):
            concept_vector = tier1_concept_matrix[:, concept_idx]
            neuron_vector = acts[:, neuron]
            iou_score = iou(concept_vector, neuron_vector)
            # save the iou for this concept and neuron
            concept_iou_map[concept_idx] = iou_score

        # sort concepts by iou and print top concept for this neuron
        sorted_concepts = sorted(concept_iou_map.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top concept for neuron {neuron}:")
        for i in range(min(1, len(sorted_concepts))):
            concept_idx, score = sorted_concepts[i]
            concept_name = tier1_concept_names[concept_idx].split("::")[1] if "::" in tier1_concept_names[concept_idx] else tier1_concept_names[concept_idx]
            print(f"    {concept_name} ({concept_idx}): {score}")

if __name__ == "__main__":
    main()
