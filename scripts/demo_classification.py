from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aigc_detector.data import load_labeled_paths, stratified_train_val_test_split
from aigc_detector.model import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Show example classifications and full clean-test metrics")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mixed_5k"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/mixed_5k_detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/mixed_5k_features.pt"))
    parser.add_argument("--examples-per-class", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    rows = load_labeled_paths(args.data_dir)
    _, _, test_rows = stratified_train_val_test_split(
        rows, args.data_dir, args.validation_fraction, args.test_fraction, args.seed
    )
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    features = cache["test_features"]
    labels = cache["test_labels"].to(torch.int64)
    expected_labels = torch.tensor([label for _, label in test_rows], dtype=torch.int64)
    if len(features) != len(test_rows) or not torch.equal(labels, expected_labels):
        raise ValueError("Feature cache does not match this dataset split; rebuild it with the same data and seed")

    head, _, temperature, _ = load_checkpoint(args.checkpoint, torch.device("cpu"))
    with torch.inference_mode():
        probabilities = torch.sigmoid(head(features) / temperature)
    predictions = (probabilities >= args.threshold).to(torch.int64)

    names = {0: "non-AI", 1: "AI-generated"}
    print(f"\nCorrectly classified examples (threshold={args.threshold:.2f})")
    print("true class    predicted class  AIGC probability  image")
    print("------------  ---------------  ----------------  -----")
    for label in (1, 0):
        shown = 0
        for index, ((path, _), truth, prediction, probability) in enumerate(
            zip(test_rows, labels.tolist(), predictions.tolist(), probabilities.tolist())
        ):
            if truth == label and prediction == truth:
                print(f"{names[truth]:12}  {names[prediction]:15}  {probability:16.4f}  {path}")
                shown += 1
                if shown >= args.examples_per_class:
                    break
        if shown < args.examples_per_class:
            print(f"Only {shown} correct {names[label]} examples were available")

    y_true = labels.numpy()
    y_pred = predictions.numpy()
    y_score = probabilities.numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print(f"\nMetrics over all {len(labels)} untouched clean test images")
    print(f"accuracy:          {accuracy_score(y_true, y_pred):.4f}")
    print(f"precision (AIGC):  {precision_score(y_true, y_pred):.4f}")
    print(f"recall (AIGC):     {recall_score(y_true, y_pred):.4f}")
    print(f"F1 (AIGC):         {f1_score(y_true, y_pred):.4f}")
    print(f"specificity:       {tn / (tn + fp):.4f}")
    print(f"ROC-AUC:           {roc_auc_score(y_true, y_score):.4f}")
    print(f"average precision: {average_precision_score(y_true, y_score):.4f}")
    print(f"confusion matrix:  TN={tn} FP={fp} FN={fn} TP={tp}")
    print("\nFull classification report")
    print(classification_report(y_true, y_pred, target_names=[names[0], names[1]], digits=4))


if __name__ == "__main__":
    main()
