from compexp_toxicity.compexp.analyze import run_analysis
from compexp_toxicity.compexp.make_tier1_concept_matrix import build_tier1_matrix
from compexp_toxicity.compexp.extract_last_token_activations import run_extraction as run_last_token_extraction
from compexp_toxicity.compexp.extract_mean_pool_activations import run_extraction as run_mean_pool_extraction
from compexp_toxicity.compexp import settings
from concepts.tier1_words import DEFAULT_STOPWORDS, NLTK_STOPWORDS
import argparse
import json
from pathlib import Path


# should only have to call this script and it should run the whole pipeline


def parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    return int(value)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the full compositional explanation pipeline: extraction, tier 1 concept matrix building, and analysis.")

    # EXTRACTION ARGS

    parser.add_argument(
        "--pooling",
        type=str,
        choices=["last_token", "mean_pool"],
        default="last_token",
        help="Pooling method for activation extraction; 'last_token' extracts activations at the last non-pad token, while 'mean_pool' computes the mean of activations over all non-pad tokens"
    )

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
        default=1,
        help="Batch size for activation extraction",
    )
    parser.add_argument(
        "--max-len",
        type=parse_optional_int,
        default=None,
        help="Maximum tokenized sequence length; use 'none' for no truncation cap",
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

    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--min-freq", type=float, default=None)
    parser.add_argument("--max-freq", type=float, default=None)
    parser.add_argument(
        "--tier1-frequency-type",
        type=str,
        choices=["total", "document"],
        default="total",
        help="How to rank/filter Tier 1 vocabulary candidates",
    )
    parser.add_argument(
        "--stopword-mode",
        type=str,
        choices=["minimal", "nltk"],
        default="minimal",
        help="Which stopword list to use for filtering candidate words"
    )

    # ANALYSIS ARGS
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Directory to save analysis results; defaults to <output-dir>/results",
    )


    return parser.parse_args()


def write_run_config(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result_dir = args.result_dir if args.result_dir is not None else args.output_dir / "results"
    run_config = {
        "extraction": {
            "dataset_dir": args.dataset_dir.as_posix(),
            "model_name_or_path": args.model_path.as_posix() if args.model_path else args.model_name,
            "output_dir": args.output_dir.as_posix(),
            "layer": args.layer,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "val_size": args.val_size,
            "seed": args.seed,
            "limit": args.limit,
            "pooling": args.pooling,
        },
        "tier1": {
            "frequency_type": args.tier1_frequency_type,
            "top_k": args.top_k,
            "min_freq": args.min_freq,
            "max_freq": args.max_freq,
            "stopword_mode": args.stopword_mode,
        },
        "analysis": {
            "result_dir": result_dir.as_posix(),
            "neurons": settings.NEURONS,
            "num_clusters": settings.NUM_CLUSTERS,
            "beam_size": settings.BEAM_SIZE,
            "max_formula_length": settings.MAX_FORMULA_LENGTH,
            "complexity_penalty": settings.COMPLEXITY_PENALTY,
        },
    }

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

def main():

  args = parse_args()  
  write_run_config(args)

  # should first extract activations from the model
  if args.pooling == "last_token":
      run_last_token_extraction(
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
  elif args.pooling == "mean_pool":
      run_mean_pool_extraction(
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
  else:
      raise ValueError(f"Invalid pooling method: {args.pooling}")
  
  if args.stopword_mode == "minimal":
      stopwords = DEFAULT_STOPWORDS
  elif args.stopword_mode == "nltk":
      stopwords = NLTK_STOPWORDS
  else:
      raise ValueError(f"Unsupported stopword mode: {args.stopword_mode}")

  # then should build the tier 1 concept matrix
  build_tier1_matrix(
        run_output_dir=args.output_dir,
        top_k=args.top_k,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        stopwords=stopwords,
        freq_type=args.tier1_frequency_type,
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
