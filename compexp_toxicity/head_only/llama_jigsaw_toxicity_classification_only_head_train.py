#!/usr/bin/env python3
"""
Multi-label Jigsaw toxicity head-only fine-tuning extracted from notebook flow.
Designed for non-interactive execution on Nautilus/cluster jobs.
"""

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


TOX_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# handles dataset download (if needed) and all required unzipping so train.csv is ready
def maybe_download_kaggle(competition: str, dataset_dir: Path) -> None:
    # unzipping the stuff that is still in zip format
    if (dataset_dir / "train.csv").exists():
        return
    outer_zip = dataset_dir / f"{competition}.zip"
    if not outer_zip.exists():
        import kaggle

        dataset_dir.mkdir(parents=True, exist_ok=True)
        kaggle.api.competition_download_cli(competition, path=str(dataset_dir))
    with zipfile.ZipFile(outer_zip) as zf:
        zf.extractall(dataset_dir)

    # Jigsaw competition ships nested zips.
    for name in ["train.csv.zip", "test.csv.zip", "test_labels.csv.zip", "sample_submission.csv.zip"]:
        nested = dataset_dir / name
        if nested.exists():
            with zipfile.ZipFile(nested) as zf:
                zf.extractall(dataset_dir)

    if not (dataset_dir / "train.csv").exists():
        raise FileNotFoundError(f"train.csv not found in {dataset_dir}")


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


def compute_pos_weight(df_train: pd.DataFrame) -> torch.Tensor:
    # BCEWithLogitsLoss uses pos_weight[c] > 1 to upweight rare positives.
    y_train = np.array(df_train["labels"].tolist(), dtype=np.float32)
    pos = y_train.sum(axis=0)
    neg = y_train.shape[0] - pos

    pw = neg / np.clip(pos, 1.0, None)
    pw = np.minimum(pw, 30.0)  # cap extreme imbalance for numerical stability

    return torch.tensor(pw, dtype=torch.float32)



def build_tokenized_dataset(
    df_train: pd.DataFrame, df_val: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int
) -> DatasetDict:
    # going from raw text data in the 'input' column to token IDs and attention masks that the model can understand, and then converting those into PyTorch tensors for efficient training.
    # converting to tensors enables the core ML functionality to efficiently compute gradients and perform backpropagation during training, as well as to leverage GPU acceleration if available.
    # a standardized process, doesn't matter what model or task you're working on, you typically want to convert your data into tensors before feeding it into the model for training or inference, because that's the format that deep learning frameworks like PyTorch and TensorFlow are optimized for.
    # the tokenizer isn't generating new token ids here, it is just obtaining them since the model has already been pretrained and the tokenizer is just converting the raw text into the token IDs that correspond to the model's vocabulary.
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(df_train, preserve_index=False),
            "val": Dataset.from_pandas(df_val, preserve_index=False),
        }
    )

    def preprocess(batch):
        return tokenizer(batch["input"], truncation=True, max_length=max_len)

    tokenized = dataset.map(preprocess, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def compute_metrics(eval_pred):
    # doing different compute_metrics bc we are doing multi-label, also pearson wouldn't have been good even if it was exclusively categorization
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "samples_f1": f1_score(labels, preds, average="samples", zero_division=0),
        "subset_accuracy": accuracy_score(labels, preds),
    }


class CustomTrainer(Trainer):
    def __init__(self, *args, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight.float() if pos_weight is not None else None
        self._debug_count = 0
        self._debug_max_steps = 20

    def compute_loss(self, model, inputs, return_outputs=False):
        # extract labels and convert to float for BCEWithLogitsLoss, which expects float targets
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.get("logits").float()

        if self.pos_weight is not None:
            # in BCE, essentially treating each label as a separate binary classification problem, and the pos_weight allows us to give more importance to the positive examples for each label, which can help the model learn better in cases of class imbalance.
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device), reduction="none")
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss_mat = loss_fn(logits, labels)
        loss = loss_mat.mean()

        if self._debug_count < self._debug_max_steps:
            print(f"DEBUG step-call={self._debug_count}")
            print("  weighted_loss:", self.pos_weight is not None)
            print("  labels min/max:", float(labels.min()), float(labels.max()))
            print("  labels any nan/inf:", bool(torch.isnan(labels).any()), bool(torch.isinf(labels).any()))
            print("  logits min/max:", float(torch.nan_to_num(logits).min()), float(torch.nan_to_num(logits).max()))
            print("  logits any nan/inf:", bool(torch.isnan(logits).any()), bool(torch.isinf(logits).any()))
            print(
                "  loss_mat min/max:",
                float(torch.nan_to_num(loss_mat).min()),
                float(torch.nan_to_num(loss_mat).max()),
            )
            print("  loss_mat any nan/inf:", bool(torch.isnan(loss_mat).any()), bool(torch.isinf(loss_mat).any()))
            print("  scalar loss:", float(loss.detach().cpu()))
            self._debug_count += 1

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected")

        return (loss, outputs) if return_outputs else loss


def infer_probabilities(model, tokenizer, df: pd.DataFrame, max_len: int, batch_size: int) -> np.ndarray:
    # at a high level basically doing evaluation again, but this time manually without the Trainer's built in eval loop
    model.eval()
    device = next(model.parameters()).device
    print(f"Running export inference on device: {device}")
    sentences = df["input"].tolist()
    all_logits = []
    total_batches = (len(sentences) + batch_size - 1) // batch_size

    print(
        f"Starting export inference over {len(sentences)} examples "
        f"in {total_batches} batches (batch_size={batch_size})"
    )

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            # detach from graph and move to cpu to avoid unnecessary gpu memory usage during inference accumulation
            outputs = model(**inputs)
            all_logits.append(outputs.logits.detach().cpu())
        batch_idx = (i // batch_size) + 1
        if batch_idx == 1 or batch_idx == total_batches or batch_idx % 50 == 0:
            print(f"Completed export batch {batch_idx}/{total_batches}")

    logits = torch.cat(all_logits, dim=0)
    probs = torch.sigmoid(logits.float()).numpy()
    return probs


def optimize_thresholds(y_true: np.ndarray, probs: np.ndarray, per_label: bool = True) -> np.ndarray | float:
    # threshold sweep because 0.5 may not be optimal for imbalanced multi-label data.
    candidates = np.arange(0.1, 0.91, 0.05)

    if per_label:
        best = np.full(probs.shape[1], 0.5, dtype=np.float32)
        for j in range(probs.shape[1]):
            best_score = -1.0
            for t in candidates:
                pred_j = (probs[:, j] >= t).astype(int)
                score = f1_score(y_true[:, j], pred_j, average="binary", zero_division=0)
                if score > best_score:
                    best_score = score
                    best[j] = t
        return best

    best_t = 0.5
    best_score = -1.0
    for t in candidates:
        preds = (probs >= t).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_t = t
    return float(best_t)


def make_predictions_from_probs(df: pd.DataFrame, probs: np.ndarray, threshold: float | np.ndarray) -> pd.DataFrame:
    # scalar threshold applies to all labels; vector threshold applies per label.
    if np.isscalar(threshold):
        preds = (probs >= float(threshold)).astype(int)
    else:
        thr = np.array(threshold, dtype=np.float32).reshape(1, -1)
        preds = (probs >= thr).astype(int)

    out = df.copy()
    out["pred_probs"] = probs.tolist()
    out["pred_labels"] = preds.tolist()
    for j, col in enumerate(TOX_COLS):
        out[f"pred_{col}"] = preds[:, j]
        out[f"prob_{col}"] = probs[:, j]
    return out


def make_predictions(model, tokenizer, df: pd.DataFrame, max_len: int, threshold: float, batch_size: int) -> pd.DataFrame:
    probs = infer_probabilities(model, tokenizer, df, max_len=max_len, batch_size=batch_size)
    return make_predictions_from_probs(df, probs=probs, threshold=threshold)


def get_performance_metrics(df_pred: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    # evaluate predictions from dataframe and generate overall + per-label breakdown for analysis
    y_true = np.array(df_pred["labels"].tolist(), dtype=int)
    y_pred = np.array(df_pred["pred_labels"].tolist(), dtype=int)

    metrics = {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "subset_accuracy": accuracy_score(y_true, y_pred),
    }

    rows = []
    for j, col in enumerate(TOX_COLS):
        report = classification_report(y_true[:, j], y_pred[:, j], output_dict=True, zero_division=0)
        rows.append(
            {
                "label": col,
                "precision": report["1"]["precision"],
                "recall": report["1"]["recall"],
                "f1": report["1"]["f1-score"],
                "support_pos": int(report["1"]["support"]),
            }
        )
    return metrics, pd.DataFrame(rows).sort_values("label")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("jigsaw-toxic-comment-classification-challenge"))
    parser.add_argument("--competition", type=str, default="jigsaw-toxic-comment-classification-challenge")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--output-dir", type=Path, default=Path("sequence_classification"))
    parser.add_argument("--saved-model-dir", type=Path, default=None)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep-thresholds", action="store_true")
    parser.add_argument("--sweep-per-label", action="store_true")
    parser.add_argument("--use-pos-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-best-model-at-end", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--export-limit", type=int, default=None)
    parser.add_argument("--export-batch-size", type=int, default=32)
    parser.add_argument("--smoke-test", action="store_true")
    # Smoke test mode is for fast pipeline validation before long runs.
    parser.add_argument("--smoke-train-size", type=int, default=1000)
    parser.add_argument("--smoke-val-size", type=int, default=200)
    parser.add_argument("--smoke-max-steps", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # HF auth token is required for gated models
    hf_token = os.environ.get("HF_TOKEN")

    maybe_download_kaggle(args.competition, args.dataset_dir)
    df = load_and_prepare_df(args.dataset_dir)
    df_train, df_val = split_multilabel(df, val_size=args.val_size, seed=args.seed)
    pos_weight = compute_pos_weight(df_train) if args.use_pos_weight else None
    if pos_weight is None:
        print("pos_weight: disabled")
    else:
        print("pos_weight:", {c: float(w) for c, w in zip(TOX_COLS, pos_weight)})
        print("pos_weight min/max:", float(pos_weight.min()), float(pos_weight.max()))

    model_source = args.saved_model_dir or args.model_name
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        token=hf_token,
        num_labels=len(TOX_COLS),
        problem_type="multi_label_classification",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    if not args.export_only:
        # freezing the llama backbone
        # requires_grad tells PyTorch whether to compute gradients for those parameters during backpropagation, which is necessary for training
        # gradients tell the optimizer how to update teh weights, the direction to move the weight and how strongly to move it based on the loss
        for p in model.model.parameters():
            p.requires_grad = False

        # keeps only the classification head trainable, which is a small fraction of the total parameters
        # model.score learns weight matrix + bias to map hidden states to toxicity logits
        # hidden states are the internal vectors that represent the input text after being processed by the model, and the score layer learns to interpret those vectors for our specific classification task
        # only looking at the final layer's hidden state but there is a hidden state for each layer
        for p in model.score.parameters():
            p.requires_grad = True

        # sanity-check: verify exactly which parameters are trainable
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable_count:,} / {total_count:,} ({100 * trainable_count / total_count:.6f}%)")
        print("Trainable parameter names:")
        for name in trainable_names:
            print(f"  - {name}")


    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token, add_prefix_space=True)
    # Setting the pad token id - determining what token id to pad with when doing batching, defined by the tokenizer so we use the eos token id
    # Also setting the token - be able to recognize the string representation
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.export_only:
        export_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(export_device)
        print(f"Loaded export-only model onto device: {export_device}")

    if args.smoke_test:
        # tiny debug subsets / debug training args flow
        max_steps = args.smoke_max_steps
        num_epochs = 1
        eval_strategy = "steps"
        eval_steps = max(1, max_steps // 2)
        save_strategy = "steps"
        save_steps = max_steps
        logging_steps = 10
    else:
        # full training path
        max_steps = -1
        num_epochs = args.num_epochs
        eval_strategy = "epoch"
        eval_steps = None
        save_strategy = "epoch"
        save_steps = None
        logging_steps = 20

    if not args.export_only:
        tokenized = build_tokenized_dataset(df_train, df_val, tokenizer, max_len=args.max_len)
        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

        if args.smoke_test:
            train_ds = tokenized["train"].select(range(min(args.smoke_train_size, len(tokenized["train"]))))
            val_ds = tokenized["val"].select(range(min(args.smoke_val_size, len(tokenized["val"]))))
        else:
            train_ds = tokenized["train"]
            val_ds = tokenized["val"]

        training_args = TrainingArguments(
            output_dir=str(args.output_dir),
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            weight_decay=args.weight_decay,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            load_best_model_at_end=(args.load_best_model_at_end and (not args.smoke_test)),
            # macro_f1 as best model metric
            metric_for_best_model="eval_macro_f1",
            greater_is_better=True,
            logging_steps=logging_steps,
            report_to="none",
            max_grad_norm=1.0,
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            # using pos_weight instead of class weight
            pos_weight=pos_weight,
        )

        train_result = trainer.train()
        eval_metrics = trainer.evaluate()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_metrics("eval", eval_metrics)
        trainer.save_state()
        trainer.save_model(str(args.output_dir / "saved_model"))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Optional prediction report on val subset/full val.
    df_eval_source = df_val.head(500) if args.smoke_test else df_val
    if args.export_limit is not None:
        df_eval_source = df_eval_source.head(args.export_limit).reset_index(drop=True)
        print(f"Export limit enabled: {len(df_eval_source)} examples")
    else:
        df_eval_source = df_eval_source.reset_index(drop=True)

    probs = infer_probabilities(
        model=model,
        tokenizer=tokenizer,
        df=df_eval_source,
        max_len=args.max_len,
        batch_size=args.export_batch_size,
    )
    y_true_eval = np.array(df_eval_source["labels"].tolist(), dtype=int)
    if args.sweep_thresholds:
        threshold_used = optimize_thresholds(y_true_eval, probs, per_label=args.sweep_per_label)
    else:
        threshold_used = args.threshold

    df_pred = make_predictions_from_probs(df_eval_source, probs=probs, threshold=threshold_used)
    overall, per_label = get_performance_metrics(df_pred)
    (args.output_dir / "predictions").mkdir(exist_ok=True)
    df_pred.to_csv(args.output_dir / "predictions" / "val_predictions.csv", index=False)
    per_label.to_csv(args.output_dir / "predictions" / "per_label_metrics.csv", index=False)
    with open(args.output_dir / "predictions" / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    with open(args.output_dir / "predictions" / "thresholds.json", "w") as f:
        if np.isscalar(threshold_used):
            json.dump({"type": "global", "threshold": float(threshold_used)}, f, indent=2)
        else:
            json.dump(
                {
                    "type": "per_label",
                    "thresholds": {label: float(t) for label, t in zip(TOX_COLS, threshold_used)},
                },
                f,
                indent=2,
            )

    print("Training complete.")
    print("Saved model:", args.output_dir / "saved_model")
    print("Saved metrics:", args.output_dir / "predictions")


if __name__ == "__main__":
    main()
