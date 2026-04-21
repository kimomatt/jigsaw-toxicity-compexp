#!/usr/bin/env python3
"""
Optimize per-label thresholds from saved validation probabilities and report
thresholded metrics plus always-negative accuracy baselines.
"""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


TOX_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("sequence_classification/predictions/val_predictions.csv"),
    )
    parser.add_argument("--grid-start", type=float, default=0.05)
    parser.add_argument("--grid-stop", type=float, default=0.95)
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def build_grid(start: float, stop: float, step: float) -> list[float]:
    grid = []
    current = start
    while current <= stop + 1e-9:
        grid.append(round(current, 10))
        current += step
    return grid


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions_csv)
    y_true = pd.DataFrame(df["labels"].apply(ast.literal_eval).tolist(), columns=TOX_COLS).astype(int)
    probs = df[[f"prob_{label}" for label in TOX_COLS]].copy()
    probs.columns = TOX_COLS

    grid = build_grid(args.grid_start, args.grid_stop, args.grid_step)
    best_rows = []
    tuned_pred = {}
    n = len(df)

    for label in TOX_COLS:
        y = y_true[label]
        p = probs[label]
        best_row = None

        for threshold in grid:
            pred = (p >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, pred, average="binary", zero_division=0
            )
            row = {
                "label": label,
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(y.sum()),
            }
            if best_row is None or row["f1"] > best_row["f1"]:
                best_row = row
                tuned_pred[label] = pred

        acc_default = float(accuracy_score(y, df[f"pred_{label}"].astype(int)))
        acc_tuned = float(accuracy_score(y, tuned_pred[label]))
        always_neg_acc = float((n - int(y.sum())) / n)

        best_row["pos_pct"] = float(y.mean())
        best_row["neg_pct"] = float(1.0 - y.mean())
        best_row["always_neg_acc"] = always_neg_acc
        best_row["acc_0.5"] = acc_default
        best_row["acc_tuned"] = acc_tuned
        best_rows.append(best_row)

    tuned_pred_df = pd.DataFrame(tuned_pred)[TOX_COLS].astype(int)
    overall_metrics = {
        "micro_f1": float(f1_score(y_true.values, tuned_pred_df.values, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true.values, tuned_pred_df.values, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true.values, tuned_pred_df.values, average="samples", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true.values, tuned_pred_df.values)),
    }

    all_negative = int((y_true.sum(axis=1) == 0).sum())
    any_positive = int((y_true.sum(axis=1) > 0).sum())

    payload = {
        "type": "per_label",
        "thresholds": {row["label"]: row["threshold"] for row in best_rows},
        "overall_metrics": overall_metrics,
        "label_metrics": best_rows,
        "label_set_balance": {
            "total_examples": int(n),
            "all_negative_examples": all_negative,
            "all_negative_fraction": float(all_negative / n),
            "any_positive_examples": any_positive,
            "any_positive_fraction": float(any_positive / n),
        },
    }

    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
