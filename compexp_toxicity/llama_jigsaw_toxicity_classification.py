# Auto-generated from llama_jigsaw_toxicity_classification.ipynb
# This script preserves code cells in notebook order.

# %% [code] cell 1
%pip uninstall -y triton bitsandbytes peft
%pip install -U "numpy<2"
%pip install "torch==2.2.2" "torchvision==0.17.2" tensorboard
%pip install "triton==2.2.0"
%pip install --upgrade \
  "transformers==4.44.2" \
  "datasets==2.18.0" \
  "accelerate==0.33.0" \
  "evaluate==0.4.1" \
  "bitsandbytes==0.43.1" \
  "huggingface_hub==0.24.6" \
  "trl==0.8.6" \
  "peft==0.12.0"
%pip install iterative-stratification



# %% [code] cell 2
from google.colab import userdata

!rm -f ~/.cache/huggingface/token
!rm -rf ~/.cache/huggingface/hub

hugginface_token = userdata.get('huggingface')
!huggingface-cli login --token $hugginface_token

# %% [code] cell 3
import os
import random
import functools
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import evaluate

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score

from scipy.stats import pearsonr
from datasets import Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# %% [code] cell 4
creds = '{"username":"' + userdata.get('kaggle_user') + '","key":"' + userdata.get('kaggle_key') + '"}'

# %% [code] cell 5
!pip install kaggle

import os
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

iskaggle

from pathlib import Path

cred_path = Path('~/.kaggle/kaggle.json').expanduser()
cred_path.parent.mkdir(exist_ok=True)
cred_path.write_text(creds)
cred_path.chmod(0o600)

cred_path


# %% [code] cell 6
path = Path('jigsaw-toxic-comment-classification-challenge')


if not iskaggle and not path.exists():
    import zipfile,kaggle
    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f'{path}.zip').extractall(path)

print("path exists:", path.exists())
print("zip exists:", Path(f"{path}.zip").exists())
print("train exists:", (path / "train.csv").exists())
print(list(path.glob("*"))[:20])

# %% [code] cell 7
# unzipping the stuff that is still in zip format

import zipfile
from pathlib import Path

path = Path("jigsaw-toxic-comment-classification-challenge")

for z in ["train.csv.zip", "test.csv.zip", "test_labels.csv.zip", "sample_submission.csv.zip"]:
    zp = path / z
    if zp.exists():
        with zipfile.ZipFile(zp, "r") as f:
            f.extractall(path)

print((path / "train.csv").exists())

# %% [code] cell 8
from pathlib import Path
path = Path('jigsaw-toxic-comment-classification-challenge')

!ls {path}

df = pd.read_csv(path/'train.csv')


# %% [code] cell 9
tox_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Ensure numeric 0/1
df[tox_cols] = df[tox_cols].astype("float32")

# Model input text column
df["input"] = df["comment_text"]

# Multi-label target vector
df["labels"] = df[tox_cols].values.tolist()

df = df[["input", "labels"]]
df.head()

# transforms dataset from its original structure to a simple two column structure
# where the input is the comment_text
# and the labels are multi-label target vectors like [1, 1, 0, 0, 1, 0] where
# each number corresponds to if a category is considered true or not

# %% [code] cell 10

# we are doing a split with multi-label stratification to ensure label-balance, prevent
# unstable/biased data

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from datasets import Dataset, DatasetDict

tox_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# df is expected to already contain:
# - input (text)
# - labels (list of 6 floats/ints)
assert "input" in df.columns and "labels" in df.columns

# Build Y matrix for stratification
Y = np.array(df["labels"].tolist(), dtype=np.int64)

# 90/10 split with multi-label stratification
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, val_idx = next(msss.split(df, Y))

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val = df.iloc[val_idx].reset_index(drop=True)

# Safety checks
required_cols = ["input", "labels"]
for name, part in [("train", df_train), ("val", df_val)]:
    missing = [c for c in required_cols if c not in part.columns]
    assert not missing, f"{name} missing columns: {missing}"
    assert part["input"].notna().all(), f"{name} has null input"
    assert part["labels"].notna().all(), f"{name} has null labels"

# Convert to HF datasets
dataset_train = Dataset.from_pandas(df_train, preserve_index=False)
dataset_val = Dataset.from_pandas(df_val, preserve_index=False)

dataset = DatasetDict({
    "train": dataset_train,
    "val": dataset_val,
})

# Optional quick prevalence check
def prevalence(frame):
    y = np.array(frame["labels"].tolist(), dtype=np.float32)
    return pd.Series(y.mean(axis=0), index=tox_cols)

print("Train prevalence:\n", prevalence(df_train).round(4))
print("\nVal prevalence:\n", prevalence(df_val).round(4))
print("\nSizes:", len(df_train), len(df_val))

# %% [code] cell 11
tox_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# df_train["labels"] is expected to be list-like length 6
Y_train = np.array(df_train["labels"].tolist(), dtype=np.float32)  # shape [N, 6]

pos_counts = Y_train.sum(axis=0)                  # positives per label
neg_counts = Y_train.shape[0] - pos_counts        # negatives per label

# BCEWithLogitsLoss uses: pos_weight[c] > 1 to upweight rare positives
pos_weight = neg_counts / np.clip(pos_counts, 1.0, None)
pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

print("Positive counts per label:")
print(dict(zip(tox_cols, pos_counts.astype(int))))

print("\nComputed pos_weight per label:")
print(dict(zip(tox_cols, pos_weight.tolist())))

# labels with proportionally low pos_counts are weighted higher

pos_weight

# %% [code] cell 12
model_name = "meta-llama/Llama-3.1-8B"

# %% [code] cell 13
quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, # enable 4-bit quantization
    bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
)

# %% [code] cell 14
lora_config = LoraConfig(
    r = 16, # the dimension of the low-rank matrices
    lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
)

# %% [code] cell 15
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=len(tox_cols),
    problem_type="multi_label_classification",
    token=userdata.get('huggingface'),
    # adding problem_type to be explict, ensure safety
)

model

# %% [code] cell 16
model = prepare_model_for_kbit_training(model)
model

# %% [code] cell 17
model = get_peft_model(model, lora_config)
model

# %% [code] cell 18
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

# %% [code] cell 19
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

# %% [code] cell 20
MAX_LEN = 128
# change back to 512 when going to nautilus

def llama_preprocessing_function(examples):
    return tokenizer(examples["input"], truncation=True, max_length=MAX_LEN)

tokenized_datasets = dataset.map(llama_preprocessing_function, batched=True)

# keep multi-label targets as "labels" (already present in your df)
tokenized_datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# going from raw text data in the 'input' column to token IDs and attention masks that the model can understand, and then converting those into PyTorch tensors for efficient training.

# converting to tensors enables the core ML functionality to efficiently compute gradients and perform backpropagation during training, as well as to leverage GPU acceleration if available.
# a standardized process, doesn't matter what model or task you're working on, you typically want to convert your data into tensors before feeding it into the model for training or inference, because that's the format that deep learning frameworks like PyTorch and TensorFlow are optimized for.

# this section basically takes each dataset row, and tokenizes the 'input' text column into input IDs and attention masks, which are the formats the model expects. It also removes some unnecessary columns and renames the label column to 'label' for compatibility with the Trainer. Finally, it sets the format to PyTorch tensors for efficient training.
# the tokenizer isn't generating new token ids here, it is just obtaining them since the model has already been pretrained and the tokenizer is just converting the raw text into the token IDs that correspond to the model's vocabulary.
# preparing the dataset for training in a similar manner that you would for pre-training, except here the objective is supervised classification rather than next-token prediction, and we are configuring the model in such a way that during training it will only be making tweaks to the lora layers and the classification head, while the rest of the model's pretrained weights remain mostly frozen.

# %% [code] cell 21
collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

# takes the tokenized examples (of varying length) and at batch time pads them to the longest sequence in the batch, knows what id to insert for padding and how to update the attention mask accordingly, so that the model doesn't attend to the padding tokens. Once this is done we have uniform length tensors per batch
# batch time means when a group of examples are chosen to go through the model, before they do go through it they are padded to the same length so they can be processed together in a batch, this is more efficient than processing each example one at a time.

# %% [code] cell 22
# doing different compute_metrics bc we are doing multi-label, also pearson wouldn't have been good even if it was exclusively categorization

from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import torch

def compute_metrics(eval_pred):
    logits, labels = eval_pred  # logits: [N, 6], labels: [N, 6]

    probs = 1 / (1 + np.exp(-logits))      # sigmoid
    preds = (probs >= 0.5).astype(int)     # threshold

    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        #pools label decisions across all samples
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        # calculates the F1 score for each label independently and then averages them, treating all labels equally regardless of their frequency in the dataset. This is useful when you want to evaluate the performance on each label separately and give equal importance to all labels, even if some are rare.
        "samples_f1": f1_score(labels, preds, average="samples", zero_division=0),
        # how accurate is each example
        "subset_accuracy": accuracy_score(labels, preds),  # exact-match accuracy
        # how many examples got all labels correct
    }

# %% [code] cell 23
class CustomTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        if pos_weight is not None:
            # keep as float tensor; move to model device at loss time
            self.pos_weight = pos_weight.float()
        else:
            self.pos_weight = None

    def compute_loss(self, model, inputs, return_outputs=False):

        # extract labels and convert to float for BCEWithLogitsLoss, which expects float targets
        labels = inputs.pop("labels").float()   # [batch, num_labels]

        # forward pass through the model to get logits
        outputs = model(**inputs)

        # Extract logits assuming they are directly outputted by the model
        logits = outputs.get("logits")          # [batch, num_labels]

        if self.pos_weight is not None:
            loss_fn = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device)
            )
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

    # in BCE, essentially treating each label as a separate binary classification problem, and the pos_weight allows us to give more importance to the positive examples for each label, which can help the model learn better in cases of class imbalance. The compute_loss function is overridden to implement this custom loss calculation during training.
    # binary classify y to {0, 1}, if y = 1 then do -log(p) where p is the sigmoid prob, if y = 0 then -log(1-p), then avergae that across all labels
    # when u apply pos_weight, only modify the positive term, so if y = 1 then do -pos_weight*log(p) which gives more penalty to getting positive examples wrong, which can help the model learn better in cases where the positive class is underrepresented in the dataset, negative term remains unchanged
    # bce has equal penalty for missing positive and negative labels when pos_weight is not used,
    # applying pos_weight amplifies positive miss errors even more

# %% [code] cell 24
training_args = TrainingArguments(
    output_dir = 'sequence_classification',
    learning_rate = 1e-4,
    # learning rate here pretty much represents how much the model's weights (specifically the adapter weights of the lora matrices) are updated during training in response to the computed loss, it is different from lora_alpha which represents how strong the adapter signal is rahter than how much the weights are updated in response to that signal
    per_device_train_batch_size = 8,
    # training happens in batches of size 8, which means that the model will process 8 examples at a time before updating the weights. This is a common practice to balance memory usage and training speed, as larger batch sizes can lead to faster training but require more memory, while smaller batch sizes are more memory efficient but can lead to noisier gradient estimates and slower training.
    per_device_eval_batch_size = 8,
    # when evaluating (which happens once at the end of each epoch in this case), examples will be evaluated in batches of 8, which can speed up the evaluation process while still fitting within memory constraints. The same applies to training, where the model will process 8 examples at a time before updating the weights.
    num_train_epochs = 2,
    weight_decay = 0.01,
    # discourages large parameter values by directly bringing down the value of the weight by a certain amount during each optimizer step, which can help prevent overfitting and improve generalization to unseen data.
    eval_strategy = 'epoch',
    save_strategy = 'epoch',
    # running evaluation and saving checkpoints at the end of each epoch, which is a common practice to monitor training progress and keep track of the best model based on evaluation metrics.
    load_best_model_at_end = True,

    metric_for_best_model="eval_macro_f1",
    # tells trainer that metric to base off of is eval macro f1
    # chose this bc task is imbalanced multi-label so we care about rare labels
    # w this sparse labels matter
    greater_is_better=True,
    # confirms that higher metric values are better, which if we are using f1 then this is true
    logging_steps=50,
    # logs training stats every 50 optimizer steps to monitor progress
    report_to="none",
    # no external logger integrations
)

# high level picture: batches respectively go through the trainer model and when a batch goes thru the weights are updated based on the computed loss given its predictions, and then once all of the batches in the epoch have gone through, evaluation happens based on the most recent model weights and it is done on a held-out data (not the same data that the model was trained on),

# Since you’re using custom metrics, set metric_for_best_model (for example balanced_accuracy or macro_f1) so “best model” is chosen by the metric you care about.

# %% [code] cell 25
trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets['train'],
    eval_dataset = tokenized_datasets['val'],
    tokenizer = tokenizer,
    data_collator = collate_fn,
    compute_metrics = compute_metrics,
    pos_weight=pos_weight,
    # using pos_weight instead of class weight
)

# puts all the pieces we have been defining together into the Trainer, which will handle the training loop, evaluation,
# and everything in between. We pass in our model, training arguments, datasets, tokenizer, data collator, custom compute_metrics
# function, and class weights for loss computation. Once this Trainer is set up, we can call trainer.train() to start the training
# process, and it will use all the configurations we've defined to train the model and evaluate it at the end of each epoch.

# %% [code] cell 26
# 1) tiny debug subsets
debug_train = tokenized_datasets["train"].select(range(1000))
debug_val = tokenized_datasets["val"].select(range(200))

# 2) debug training args
debug_args = TrainingArguments(
    output_dir="sequence_classification_debug",
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    max_steps=20,                 # smoke test cap
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=20,
    load_best_model_at_end=False,  # keep simple for smoke test
    logging_steps=10,
    report_to="none",
)

# 3) debug trainer
debug_trainer = CustomTrainer(
    model=model,
    args=debug_args,
    train_dataset=debug_train,
    eval_dataset=debug_val,
    tokenizer=tokenizer,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    pos_weight=pos_weight,
)

# 4) run smoke train
debug_result = debug_trainer.train()
print(debug_result.metrics)

# 5) quick eval
debug_eval = debug_trainer.evaluate()
print(debug_eval)

# %% [code] cell 27
train_result = trainer.train()

# %% [code] cell 28
def make_predictions(model, df, threshold=0.5, batch_size=32):

    # put model in eval mode to disable dropout and other training-specific behavior, ensuring that the model's predictions are deterministic and consistent during inference, which is important for getting reliable predictions on new data.
    model.eval()

    # use the same devices as the model parameters, avoid device mismatch errors
    device = next(model.parameters()).device

    # getting the list of sentences back from the dataframe
    sentences = df["input"].tolist()

    # get logits from each mini-batch and store them in a list
    all_logits = []

    # batch inference loop, go through all the sentences in batches
    for i in range(0, len(sentences), batch_size):
        # batch of sentences to process together, this is more efficient than processing each sentence one at a time, especially when using a GPU, as it allows for parallel computation and better utilization of resources.
        batch_sentences = sentences[i:i + batch_size]

        # tokenize batch to what the model expects, and move tensors to the same device as the model to avoid device mismatch errors
        inputs = tokenizer(
            batch_sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # forward pass without gradient calculation since we are just doing inference
        with torch.no_grad():
            # forward pass through the model to get logits, which are the raw outputs before applying sigmoid for multi-label classification
            outputs = model(**inputs)

            # add the logits for this batch to our list, detaching from the computation graph (internal record of operations for backpropagation) and moving to CPU to avoid memory issues, let GPU memory focus on forward passes
            all_logits.append(outputs.logits.detach().cpu())

    # stacks tensors from all batches into a single tensor
    logits = torch.cat(all_logits, dim=0)                      # [N, 6]
    # applies sigmoid to convert logits to probabilities
    probs = torch.sigmoid(logits).numpy()                      # [N, 6]
    # applies the threshold to get binary predictions for each label
    preds = (probs >= threshold).astype(int)                   # [N, 6]

    tox_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    # return copy of original dataframe with new columns for predicted probabilities and labels, as well as convenience columns for each toxicity type, don't want to modify original dataframe in case we want to compare predictions to true labels or do other analyses, so we create a copy and add the new columns there.

    df = df.copy()
    df["pred_probs"] = probs.tolist()
    df["pred_labels"] = preds.tolist()

    # Optional convenience columns per label
    for j, col in enumerate(tox_cols):
        df[f"pred_{col}"] = preds[:, j]
        df[f"prob_{col}"] = probs[:, j]

    return df

# %% [code] cell 29
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np
import pandas as pd

def get_performance_metrics(df_pred):
    """
    Expects dataframe with:
      - true labels in column: 'labels' (list of 0/1 length 6)
      - predicted labels in column: 'pred_labels' (list of 0/1 length 6)
    """

    tox_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    # Convert list-columns to 2D numpy arrays: shape [N, 6]
    y_true = np.array(df_pred["labels"].tolist(), dtype=int)
    y_pred = np.array(df_pred["pred_labels"].tolist(), dtype=int)

    # Global multi-label metrics
    metrics = {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "subset_accuracy": accuracy_score(y_true, y_pred),  # exact match of all 6 labels
    }

    print("Overall metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Per-label summary table - one row per toxicity label, with
    # performance for that label treated as its own binary task
    per_label_rows = []
    for j, col in enumerate(tox_cols):
        report_j = classification_report(
            y_true[:, j],
            y_pred[:, j],
            output_dict=True,
            zero_division=0
        )
        per_label_rows.append({
            "label": col,
            "precision": report_j["1"]["precision"],
            "recall": report_j["1"]["recall"],
            "f1": report_j["1"]["f1-score"],
            "support_pos": int(report_j["1"]["support"]),
        })

    per_label_df = pd.DataFrame(per_label_rows).sort_values("label")
    print("\nPer-label metrics (positive class):")
    print(per_label_df.to_string(index=False))

    return metrics, per_label_df

# %% [code] cell 30
# DEBUG VERSION (fast smoke test)
# Switch back later: use full df_val instead of .head(500)
df_val_pred = make_predictions(model, df_val.head(500), threshold=0.5, batch_size=32)

# Evaluate predictions on that same debug dataframe
metrics, per_label_df = get_performance_metrics(df_val_pred)

# Inspect prediction dataframe
df_val_pred.head()


# # FULL VERSION
# df_val_pred = make_predictions(model, df_val, threshold=0.5, batch_size=32)
# metrics, per_label_df = get_performance_metrics(df_val_pred)
# df_val_pred.head()

# %% [code] cell 31
# DEBUG VERSION
metrics = debug_result.metrics
max_train_samples = len(debug_train)
metrics["train_samples"] = min(max_train_samples, len(debug_train))

debug_trainer.log_metrics("train", metrics)
debug_trainer.save_metrics("train", metrics)
debug_trainer.save_state()

# # FULL VERSION
# metrics = train_result.metrics
# max_train_samples = len(dataset_train)
# metrics["train_samples"] = min(max_train_samples, len(dataset_train))

# trainer.log_metrics("train", metrics)
# trainer.save_metrics("train", metrics)
# trainer.save_state()

# %% [code] cell 32
# DEBUG VERSION
debug_trainer.save_model("saved_model_debug")

# # FULL VERSION
# trainer.save_model("saved_model")

# %% [code] cell 33
from google.colab import drive
drive.mount('/content/drive')

# %% [code] cell 34
!cp -r sequence_classification /content/drive/MyDrive/Colab

# %% [code] cell 35
!cp -r saved_model /content/drive/MyDrive/Colab

