from compexp_toxicity.compexp.analyze import run_analysis
from compexp_toxicity.compexp.make_tier1_concept_matrix import build_tier1_matrix
from compexp_toxicity.compexp.extract_last_token_activations import run_extraction
import argparse
from pathlib import Path


# should only have to call this script and it should run the whole pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Run the full compositional explanation pipeline: extraction, tier 1 concept matrix building, and analysis.")

    # EXTRACTION ARGS

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("jigsaw-toxic-comment-classification-challenge"),
        help="Directory containing train.csv",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Hugging Face model name or path",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to local model checkpoint; overrides --model-name if provided",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("compexp_toxicity/compexp/outputs"),
        help="Directory to save activations and metadata",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Hidden-state layer index to extract, e.g. -1 for final layer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for activation extraction",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Maximum tokenized sequence length",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation split fraction used when recreating the dataset split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible splitting",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples to process (for testing)",
    )

    # TIER 1 CONCEPT MATRIX ARGS

    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--min-doc-freq", type=int, default=20)
    parser.add_argument("--max-doc-frac", type=float, default=0.7)

    # ANALYSIS ARGS
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Directory to save analysis results; defaults to <output-dir>/results",
    )


    return parser.parse_args()

def main():

  args = parse_args()  

  # should first extract activations from the model
  run_extraction(
        dataset_dir=args.dataset_dir,
        model_name=args.model_name,
        model_path=args.model_path,
        output_dir=args.output_dir,
        layer=args.layer,
        batch_size=args.batch_size,
        max_len=args.max_len,
        val_size=args.val_size,
        seed=args.seed,
        limit=args.limit,
    )

  # then should build the tier 1 concept matrix
  build_tier1_matrix(
        run_output_dir=args.output_dir,
        top_k=args.top_k,
        min_doc_freq=args.min_doc_freq,
        max_doc_frac=args.max_doc_frac,
    )

  # then should run the analysis to find compositional explanations for each neuron based on the tier 1 concept matrix and the neuron activations, and save the results in a format that can be easily analyzed and visualized in the sentence report.
  run_analysis(
        path_to_activations=args.output_dir / "val_activations.npy",
        path_to_concept_matrix=args.output_dir / "conceptset_tier1" / "conceptset_tier1.npy",
        path_to_concept_names=args.output_dir / "conceptset_tier1" / "conceptset_tier1_names.txt",
        result_dir = args.result_dir if args.result_dir is not None else args.output_dir / "results",
    )
  
  
if __name__ == "__main__":
    main()