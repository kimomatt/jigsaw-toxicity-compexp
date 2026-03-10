#!/usr/bin/env python3
"""
Multi-label Jigsaw toxicity fine-tuning (QLoRA) extracted from notebook flow.
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


TOX_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


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
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    return df_train, df_val


def compute_pos_weight(df_train: pd.DataFrame) -> torch.Tensor:
    # BCEWithLogitsLoss uses: pos_weight[c] > 1 to upweight rare positives
    # labels with proportionally low pos_counts are weighted higher
    y_train = np.array(df_train["labels"].tolist(), dtype=np.float32)
    pos = y_train.sum(axis=0)
    neg = y_train.shape[0] - pos
    return torch.tensor(neg / np.clip(pos, 1.0, None), dtype=torch.float32)


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

    def compute_loss(self, model, inputs, return_outputs=False):
        # extract labels and convert to float for BCEWithLogitsLoss, which expects float targets
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.pos_weight is not None:
            # in BCE, essentially treating each label as a separate binary classification problem, and the pos_weight allows us to give more importance to the positive examples for each label, which can help the model learn better in cases of class imbalance.
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def infer_probabilities(model, tokenizer, df: pd.DataFrame, max_len: int, batch_size: int) -> np.ndarray:
    # at a high level basically doing evaluation again, but this time manually without the Trainer's built in eval loop
    model.eval()
    device = next(model.parameters()).device
    sentences = df["input"].tolist()
    all_logits = []

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

    logits = torch.cat(all_logits, dim=0)
    probs = torch.sigmoid(logits).numpy()
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
    pos_weight = compute_pos_weight(df_train)

    quantization_config = BitsAndBytesConfig(
        # enable 4-bit quantization for qlora
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        # lora config for seq cls
        r=16,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        token=hf_token,
        quantization_config=quantization_config,
        num_labels=len(TOX_COLS),
        problem_type="multi_label_classification",
    )
    # prepare model for k-bit training then wrap with lora adapters
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token, add_prefix_space=True)
    # Setting the pad token id - determining what token id to pad with when doing batching, defined by the tokenizer so we use the eos token id
    # Also setting the token - be able to recognize the string representation
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    tokenized = build_tokenized_dataset(df_train, df_val, tokenizer, max_len=args.max_len)
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.smoke_test:
        # tiny debug subsets / debug training args flow
        train_ds = tokenized["train"].select(range(min(args.smoke_train_size, len(tokenized["train"]))))
        val_ds = tokenized["val"].select(range(min(args.smoke_val_size, len(tokenized["val"]))))
        max_steps = args.smoke_max_steps
        num_epochs = 1
        eval_strategy = "steps"
        eval_steps = max(1, max_steps // 2)
        save_strategy = "steps"
        save_steps = max_steps
    else:
        # full training path
        train_ds = tokenized["train"]
        val_ds = tokenized["val"]
        max_steps = -1
        num_epochs = args.num_epochs
        eval_strategy = "epoch"
        eval_steps = None
        save_strategy = "epoch"
        save_steps = None

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
        load_best_model_at_end=(not args.smoke_test),
        # macro_f1 as best model metric
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=10 if args.smoke_test else 100,
        report_to="none",
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

    # Optional prediction report on val subset/full val.
    df_eval_source = df_val.head(500) if args.smoke_test else df_val
    probs = infer_probabilities(
        model=model,
        tokenizer=tokenizer,
        df=df_eval_source,
        max_len=args.max_len,
        batch_size=32,
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
