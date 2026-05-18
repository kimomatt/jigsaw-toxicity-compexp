import pandas as pd
import torch
import pytest
import numpy as np
import csv
from compexp_toxicity.compexp.extract_last_token_activations import load_and_prepare_df, extract_activations, tokenize_batch, split_multilabel
from compexp_toxicity.compexp.extract_mean_pool_activations import extract_activations as extract_mean_pool_activations

# tightly coupled to jigsaw dataset, instead i think for each run u pass in custom load_and_prepare_df function that transforms the dataset into a simple two column structure with "input" and "labels", where "input" is the text to be fed into the model and "labels" is the multi-label target vector for that text, then we can reuse this code for other datasets in the future by just writing different load_and_prepare_df functions without having to change the rest of the code for tokenization and activation extraction
# TOX_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# def load_and_prepare_df(dataset_dir: Path) -> pd.DataFrame:
#     # Ensure numeric 0/1
#     # Model input text column
#     # Multi-label target vector
#     # transforms dataset from its original structure to a simple two column structure
#     # where the input is the comment_text
#     # and the labels are multi-label target vectors like [1, 1, 0, 0, 1, 0] where
#     # each number corresponds to if a category is considered true or not
#     df = pd.read_csv(dataset_dir / "train.csv")
#     # using float32 for labels because BCEWithLogitsLoss expects float targets
#     df[TOX_COLS] = df[TOX_COLS].astype("float32")
#     df["input"] = df["comment_text"]
#     df["labels"] = df[TOX_COLS].values.tolist()
#     return df[["input", "labels"]]

# load_and_prepare_df
# happy path
# labels column shape/content
# output columns are exactly input and labels
def test_load_and_prepare_df_happy(tmp_path):
    csv_path = tmp_path / "train.csv"

    # content of csv
#     "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
# "0000997932d777bf","Explanation
# Why the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. And please don't remove the template from the talk page since I'm retired now.89.205.38.27",0,0,0,0,0,0
# "000103f0d9cfb60f","D'aww! He matches this background colour I'm seemingly stuck with. Thanks.  (talk) 21:51, January 11, 2016 (UTC)",0,0,0,0,0,0

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"])
        writer.writeheader()
        writer.writerow({"id": "s0", "comment_text": "I love this product.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0})
        writer.writerow({"id": "s1", "comment_text": "You are awful and I hate this.", "toxic": 1, "severe_toxic": 1, "obscene": 1, "threat": 1, "insult": 1, "identity_hate": 1})
        writer.writerow({"id": "s2", "comment_text": "Email me at example@email.com.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0})
    
    df = load_and_prepare_df(tmp_path)
    assert list(df.columns) == ["input", "labels"]
    assert len(df) == 3
    assert df["input"].tolist() == ["I love this product.", "You are awful and I hate this.", "Email me at example@email.com."]
    assert df["labels"].tolist() == [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]]


# def split_multilabel(df: pd.DataFrame, val_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
#     # we are doing a split with multi-label stratification to ensure label-balance, prevent
#     # unstable/biased data
#     y = np.array(df["labels"].tolist(), dtype=np.int64)
#     splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
#     train_idx, val_idx = next(splitter.split(df, y))
#     # iloc picks rows by position, reset index cleans the index after selection
#     df_train = df.iloc[train_idx].reset_index(drop=True)
#     df_val = df.iloc[val_idx].reset_index(drop=True)
#     return df_train, df_val

def test_split_multilabel_returns_train_and_val_with_expected_sizes():
    # create a simple dataframe with 10 examples and multi-label targets
    data = {
        "input": [f"comment_{i}" for i in range(10)],
        "labels": [[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0]],
    }
    df = pd.DataFrame(data)
    df_train, df_val = split_multilabel(df, val_size=0.2, seed=42)
    assert len(df_train) == 8
    assert len(df_val) == 2
    assert len(df_train) + len(df_val) == len(df)

    # check that both outputs have reset indices
    assert df_train.index.tolist() == list(range(8))
    assert df_val.index.tolist() == list(range(2))

def test_split_multilabel_is_reproducible_for_same_seed():
    data = {
        "input": [f"comment_{i}" for i in range(10)],
        "labels": [[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0]],
    }
    df = pd.DataFrame(data)
    df_train_1, df_val_1 = split_multilabel(df, val_size=0.2, seed=42)
    df_train_2, df_val_2 = split_multilabel(df, val_size=0.2, seed=42)
    assert df_train_1.equals(df_train_2)
    assert df_val_1.equals(df_val_2)

# def extract_activations(model: AutoModelForSequenceClassification, tokenized_inputs: dict[str, torch.Tensor], layer: int):
#     # run a forward pass through the model with the tokenized inputs and extract the hidden states from the specified layer, then return the activations for the last non-pad token in each sequence in the batch
#     # need the hidden state and the token indices to get the activations that we want
#     # batch size is number of sequences/rows per batch
#     with torch.no_grad(): # don't need gradients since we're just doing a forward pass / inference pass
#         outputs = model(**tokenized_inputs, output_hidden_states=True) # unpacks the tokenized_inputs dictionary into keyword arguments for the model's forward method, which typically expects input_ids and attention_mask as arguments
#         hidden_states = outputs.hidden_states  # tuple of (batch_size, seq_len, hidden_dim) for each layer
#         # get the hidden states for the specified layer
#         layer_hidden_states = hidden_states[layer]  # (batch_size, seq_len, hidden_dim)

#         # find the indices of the last non-pad tokens in each sequence
#         attention_mask = tokenized_inputs["attention_mask"]  # (batch_size, seq_len)
#         seq_lengths = attention_mask.sum(dim=1)  # (batch_size,) gives us the length of each sequence before padding bc theres 1s for real tokens and 0s for pads
#         last_token_indices = seq_lengths - 1  # (batch_size,) gives us the index of the last non-pad token for each sequence

#         # gather the activations for the last non-pad tokens using advanced indexing
#         batch_indices = torch.arange(int(layer_hidden_states.size(0)), device=layer_hidden_states.device)  # (batch_size,), gives us the batch indices [0, 1, 2, ..., batch_size-1] to index into the first dimension of layer_hidden_states, getting the row for each sequence in the batch

#         # getting all the neuron activations for the last non-pad token for each sequence by indexing into layer_hidden_states with batch_indices and last_token_indices, which gives us a tensor of shape (batch_size, hidden_dim) containing the activations for the last non-pad token in each sequence in the batch
#         activations = layer_hidden_states[batch_indices, last_token_indices]  # (batch_size, hidden_dim)

#     # move to cpu and convert to numpy for easier saving and downstream analysis, since we don't need to do any more PyTorch operations on the activations after this point, we can convert them to NumPy arrays which are more standard for data storage and analysis in Python, and also ensure that they are on the CPU so that we can save them without needing GPU resources, also needs to be on cpu to convert to numpy since numpy doesn't work with GPU tensors
#     return activations.cpu().float().numpy()

# extract_activations
# this is the big one
# use a fake model object returning fake hidden states
# verify it picks the last non-pad token correctly
# verify output shape and numpy conversion

def test_extract_activations_uses_last_non_pad_token():
    class FakeOutput:
        def __init__(self, hidden_states):
            self.hidden_states = hidden_states

    class FakeModel:
        def __call__(self, **kwargs):
            batch_size = kwargs["input_ids"].shape[0]
            seq_len = kwargs["input_ids"].shape[1]
            hidden_dim = 4

            hidden_states = []
            for layer in range(3):
                layer_tensor = torch.zeros(batch_size, seq_len, hidden_dim)
                for i in range(batch_size):
                    for j in range(seq_len):
                        layer_tensor[i, j] = (layer + 1) * 100 + (i + 1) * 10 + j
                hidden_states.append(layer_tensor)

            return FakeOutput(tuple(hidden_states))

    model = FakeModel()
    tokenized_inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
    }
    # for this set of inputs, you would have a batch size of 2 and a seq len of 4 and then 4 activations per token (and want teh last one)

    activations = extract_activations(model, tokenized_inputs, layer=1)

    assert isinstance(activations, np.ndarray)
    assert activations.shape == (2, 4)

    expected = np.array([
        [212, 212, 212, 212],  # seq 0, last non-pad index 2
        [221, 221, 221, 221],  # seq 1, last non-pad index 1
    ], dtype=np.float32)

    assert np.array_equal(activations, expected)

def test_extract_mean_pool_activations():
    class FakeOutput:
        def __init__(self, hidden_states):
            self.hidden_states = hidden_states

    class FakeModel:
        def __call__(self, **kwargs):
            batch_size = kwargs["input_ids"].shape[0]
            seq_len = kwargs["input_ids"].shape[1]
            hidden_dim = 4

            hidden_states = []
            for layer in range(3):
                layer_tensor = torch.zeros(batch_size, seq_len, hidden_dim)
                for i in range(batch_size):
                    for j in range(seq_len):
                        layer_tensor[i, j] = (layer + 1) * 100 + (i + 1) * 10 + j
                hidden_states.append(layer_tensor)

            # hidden states would look like this for layer 1:
            # [
            # seq0[
            #         [210, 210, 210, 210],
            #         [211, 211, 211, 211],
            #         [212, 212, 212, 212],
            #         [213, 213, 213, 213],
            #     ],
            # seq1[
            #         [220, 220, 220, 220],
            #         [221, 221, 221, 221],
            #         [222, 222, 222, 222],
            #         [223, 223, 223, 223],
            #     ],
            # ]

            return FakeOutput(tuple(hidden_states))

    model = FakeModel()
    tokenized_inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
    }
    # for this set of inputs, you would have a batch size of 2 and a seq len of 4 and then 4 activations per toke

    activations = extract_mean_pool_activations(model, tokenized_inputs, layer=1)

    assert isinstance(activations, np.ndarray)
    assert activations.shape == (2, 4)

    expected = np.array([
        [211, 211, 211, 211],  # seq 0, mean of indices 0, 1, 2 is 211
        [220.5, 220.5, 220.5, 220.5],  # seq 1, mean of indices 0, 1 is 220.5
    ], dtype=np.float32)

    assert np.allclose(activations, expected)
    
