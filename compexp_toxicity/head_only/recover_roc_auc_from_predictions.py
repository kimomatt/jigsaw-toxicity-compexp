#!/usr/bin/env python3
"""
Recover per-label and aggregate ROC AUC metrics from saved validation predictions.
"""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


TOX_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # what does it mean to add argument here
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("sequence_classification/predictions/val_predictions.csv"),
    )
    parser.add_argument("--output-json", type=Path, default=None)

    # is this parse_args function separate from this main function it is inside
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions_csv)
    # turn each string of labels like "[0, 1, 0, 0, 1, 0]" into a list of integers and then convert that whole column into a list of list, then convert the list of lists into a dataframe with columns named after the toxicity labels, and finally convert all values to integers
    y_true = pd.DataFrame(df["labels"].apply(ast.literal_eval).tolist(), columns=TOX_COLS).astype(int)
    # grab the probability columns for each label, rename them to match the toxicity labels
    y_score = df[[f"prob_{label}" for label in TOX_COLS]].copy()
    y_score.columns = TOX_COLS

    per_label_auc = {label: float(roc_auc_score(y_true[label], y_score[label])) for label in TOX_COLS}
    macro_auc = float(sum(per_label_auc.values()) / len(per_label_auc))
    micro_auc = float(roc_auc_score(y_true.values, y_score.values, average="micro"))

    payload = {
        "macro_roc_auc": macro_auc,
        "micro_roc_auc": micro_auc,
        "per_label_roc_auc": per_label_auc,
    }

    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
