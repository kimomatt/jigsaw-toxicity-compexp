import argparse
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

hf_token = os.environ.get("HF_TOKEN")


def parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    return int(value)

# parse args

# for args, we should be able to take in the path to the dataset, the path to the model, the layer we want to extract from, and the output path for the activations, also maybe some args for batch size and max sequence length for tokenization, and maybe a random seed for reproducibility when we recreate the dataset split

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract mean pooled activations for concept-expression analysis."
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
        default=32,
        help="Batch size for activation extraction",
    )
    parser.add_argument(
        "--max-len",
        type=parse_optional_int,
        default=512,
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
    return parser.parse_args()

# load dataset and rebuild df_val

# tightly coupled to jigsaw dataset, instead i think for each run u pass in custom load_and_prepare_df function that transforms the dataset into a simple two column structure with "input" and "labels", where "input" is the text to be fed into the model and "labels" is the multi-label target vector for that text, then we can reuse this code for other datasets in the future by just writing different load_and_prepare_df functions without having to change the rest of the code for tokenization and activation extraction
TOX_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
def load_and_prepare_df(dataset_dir: Path) -> pd.DataFrame:
    # Ensure numeric 0/1
    # Model input text column
    # Multi-label target vector
    # transforms dataset from its original structure to a simple two column structure
    # where the input is the comment_text
    # and the labels are multi-label target vectors like [1, 1, 0, 0, 1, 0] where
    # each number corresponds to if a category is considered true or not
    df = pd.read_csv(dataset_dir / "train.csv")
    # using float32 for labels because BCEWithLogitsLoss expects float targets
    df[TOX_COLS] = df[TOX_COLS].astype("float32")
    df["input"] = df["comment_text"]
    df["labels"] = df[TOX_COLS].values.tolist()
    return df[["input", "labels"]]

def split_multilabel(df: pd.DataFrame, val_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # we are doing a split with multi-label stratification to ensure label-balance, prevent
    # unstable/biased data
    y = np.array(df["labels"].tolist(), dtype=np.int64)
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, y))
    # iloc picks rows by position, reset index cleans the index after selection
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    return df_train, df_val

# load tokenizer and model

def load_model_and_tokenizer(model_name: str, model_path: Path):
    # load the model and tokenizer from Hugging Face, if model_path is provided, load from there instead of model_name
    # make sure to set output_hidden_states=True when loading the model so that we can access the hidden states later
    model_source = model_path or model_name

    # adds a bit of overhead, but ensures that architecture is the same as it was in the head-only training
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        token=hf_token,
        num_labels=len(TOX_COLS),
        problem_type="multi_label_classification",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_source, token=hf_token, add_prefix_space=True)
    # Setting the pad token id - determining what token id to pad with when doing batching, defined by the tokenizer so we use the eos token id
    # Also setting the token - be able to recognize the string representation
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model, tokenizer

# tokenize one batch
def tokenize_batch(texts: list[str], tokenizer: AutoTokenizer, max_len: int | None):
    # tokenize the input texts using the provided tokenizer. when max_len is set, truncate to that cap;
    # otherwise keep the full tokenized sequence length for each example in the batch.
    tokenizer_kwargs = {
        "padding": "longest",
        "return_tensors": "pt",
    }
    if max_len is None:
        tokenizer_kwargs["truncation"] = False
    else:
        tokenizer_kwargs["truncation"] = True
        tokenizer_kwargs["max_length"] = max_len

    tokenized = tokenizer(texts, **tokenizer_kwargs)
    return tokenized

# run one forward pass with output_hidden_states=True
# extract last non-pad token vectors from one chosen layer

def extract_activations(model: AutoModelForSequenceClassification, tokenized_inputs: dict[str, torch.Tensor], layer: int):
    # run a forward pass through the model with the tokenized inputs and extract the hidden states from the specified layer, then return the activations for the last non-pad token in each sequence in the batch
    # need the hidden state and the token indices to get the activations that we want
    # batch size is number of sequences/rows per batch
    with torch.no_grad(): # don't need gradients since we're just doing a forward pass / inference pass
        outputs = model(**tokenized_inputs, output_hidden_states=True) # unpacks the tokenized_inputs dictionary into keyword arguments for the model's forward method, which typically expects input_ids and attention_mask as arguments
        hidden_states = outputs.hidden_states  # tuple of (batch_size, seq_len, hidden_dim) for each layer
        # get the hidden states for the specified layer
        layer_hidden_states = hidden_states[layer]  # (batch_size, seq_len, hidden_dim)

        # find the indices of the last non-pad tokens in each sequence
        attention_mask = tokenized_inputs["attention_mask"]  # (batch_size, seq_len)
        seq_lengths = attention_mask.sum(dim=1)  # (batch_size,) gives us the length of each sequence before padding bc theres 1s for real tokens and 0s for pads

        # this is the point where we want to differ from the extract last token activations script and instead do mean pooling over all the non-pad tokens, so instead of just taking the activations at the last_token_indices, we want to take the mean of the activations for all tokens up to seq_lengths for each sequence in the batch, which means we need to create a mask that zeroes out the pad token activations and then take the mean over the seq_len dimension for each sequence

        mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1) to broadcast over hidden_dim, lets pytorch apply mask to every neuron dimension at each token 
        masked_hidden_states = layer_hidden_states * mask  # zero out pad token activations, (batch_size, seq_len, hidden_dim)
        sum_hidden_states = masked_hidden_states.sum(dim=1)  # sum over seq_len dimension, (batch_size, hidden_dim)
        mean_pooled_activations = sum_hidden_states / seq_lengths.unsqueeze(-1)  # divide by actual sequence lengths to get mean, (batch_size, hidden_dim)

    # move to cpu and convert to numpy for easier saving and downstream analysis, since we don't need to do any more PyTorch operations on the activations after this point, we can convert them to NumPy arrays which are more standard for data storage and analysis in Python, and also ensure that they are on the CPU so that we can save them without needing GPU resources, also needs to be on cpu to convert to numpy since numpy doesn't work with GPU tensors
    return mean_pooled_activations.cpu().float().numpy()


def run_extraction(
    dataset_dir: Path,
    model_name: str,
    model_path: Path | None,
    output_dir: Path,
    layer: int,
    batch_size: int,
    max_len: int | None,
    val_size: float,
    seed: int,
    limit: int | None = None,
) -> None:
    # this is the main function that will run the whole extraction process, including loading the dataset, preparing the dataframes, loading the model and tokenizer, running the forward passes to extract activations, and saving the results to the output directory
    df = load_and_prepare_df(dataset_dir)
    _df_train, df_val = split_multilabel(df, val_size, seed)
    if limit is not None:
        df_val = df_val.head(limit).reset_index(drop=True)
    model, tokenizer = load_model_and_tokenizer(model_name, model_path)
    model.eval()  # set model to evaluation mode since we're just doing inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # we will extract activations for the validation set since that's what we will be analyzing with the concepts later on
    all_activations = []
    total_batches = (len(df_val) + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, len(df_val), batch_size), start=1):
        batch_texts = df_val["input"].iloc[i : i + batch_size].tolist()
        tokenized_inputs = tokenize_batch(batch_texts, tokenizer, max_len)
        seq_len = int(tokenized_inputs["input_ids"].shape[1])
        if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == total_batches:
            print(
                f"Extraction progress: batch {batch_idx}/{total_batches}, "
                f"rows {i}-{min(i + batch_size, len(df_val)) - 1}, seq_len={seq_len}"
            )
        # move tokenized inputs to the same device as the model
        # this for each key-value pair in the tokenized_inputs dictionary, so for example if tokenized_inputs has keys "input_ids" and "attention_mask", it will move both of those tensors to the device (GPU or CPU) that the model is on, ensuring that the inputs are on the same device as the model for the forward pass
        tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
        batch_activations = extract_activations(model, tokenized_inputs, layer)
        # all_activations is a list of numpy arrays, where each array has shape (batch_size, hidden_dim) and contains the activations for the last non-pad token for each sequence in that batch
        all_activations.append(batch_activations)

    # concatenate all the batch activations together to get a full matrix of activations for the entire validation set
    # axis = 0 means we are stacking the rows
    all_activations = np.concatenate(all_activations, axis=0)  # (num_val_samples, hidden_dim)
    # verify shapes
    assert all_activations.shape[0] == len(df_val), f"Expected {len(df_val)} samples, got {all_activations.shape[0]}"
    print(f"Extracted activations shape: {all_activations.shape}")

    # save the activations and metadata to the output directory

    # parents=True means it will create any necessary parent directories if they don't exist, exist_ok=True means it won't raise an error if the directory already exists, necessary parent directories would be anything in the path that doesn't already exist, for example if output_dir is "compexp_toxicity/compexp/outputs" and "compexp_toxicity/compexp" already exists but "outputs" doesn't, it will create the "outputs" directory without raising an error about the directory already existing
    output_dir.mkdir(parents=True, exist_ok=True)

    # saving in npy format as "val_activations.npy" in the output directory
    np.save(output_dir / "val_activations.npy", all_activations)

    # saving the metadata (input texts and labels) for the validation set as "val_metadata.csv" in the output directory
    df_val[["input", "labels"]].to_csv(output_dir / "val_metadata.csv", index=False)

    # maybe also save layer, model name / path, max)len, val_size, seed
    metadata = {
        "dataset_dir": dataset_dir.as_posix(),
        "model_name_or_path": model_path.as_posix() if model_path else model_name,
        "layer": layer,
        "batch_size": batch_size,
        "max_len": max_len,
        "val_size": val_size,
        "seed": seed,
        "limit": limit,
        "pooling": "mean_pool",
    }

    # save the metadata as a json file in the output directory, named "extraction_metadata.json"
    with open(output_dir / "extraction_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def main():
    args = parse_args()
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
    

if __name__ == "__main__":
    main()

# if we can run this successfully then we can move onto connecting concept generation, and then the beam search
# over the concept space to find the concepts that are most predictive of toxicity according to the model, and then we can analyze those concepts to see if they align with human-understandable notions of toxicity or if they reveal any surprising insights about what the model is picking up on in the data.
