from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def _source(path: str | Path, data_dir: Path) -> str:
    parts = Path(path).resolve().relative_to(data_dir.resolve()).parts
    if len(parts) < 3 or parts[0].lower() not in {"real", "fake", "ai", "aigc", "synthetic", "generated"}:
        raise ValueError(f"Expected path below a labeled class/<source> directory: {path}")
    return parts[1]


def _limit_per_class(
    rows: list[tuple[Path, int]], maximum: int | None
) -> list[tuple[Path, int]]:
    if maximum is None:
        return rows
    if maximum < 1:
        raise ValueError("--max-images-per-class must be positive")
    limited: list[tuple[Path, int]] = []
    for label in (0, 1):
        limited.extend([row for row in rows if row[1] == label][:maximum])
    return limited


def _metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | None]:
    predictions = (probabilities >= threshold).astype(int)
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    positive_recall = true_positive / max(1, true_positive + false_negative)
    negative_recall = true_negative / max(1, true_negative + false_positive)
    precision = true_positive / max(1, true_positive + false_positive)
    if set(np.unique(labels)) == {0, 1}:
        balanced_accuracy = (positive_recall + negative_recall) / 2
    elif np.all(labels == 1):
        balanced_accuracy = positive_recall
    else:
        balanced_accuracy = negative_recall

    auc = None
    average_precision = None
    if set(np.unique(labels)) == {0, 1}:
        order = np.argsort(probabilities, kind="mergesort")
        ranks = np.empty(len(probabilities), dtype=float)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and probabilities[order[end]] == probabilities[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
        positives = labels == 1
        positive_count = int(positives.sum())
        negative_count = len(labels) - positive_count
        auc = float(
            (ranks[positives].sum() - positive_count * (positive_count + 1) / 2)
            / (positive_count * negative_count)
        )
        average_precision = float(average_precision_score(labels, probabilities))
    return {
        "roc_auc": auc,
        "average_precision": average_precision,
        "balanced_accuracy": float(balanced_accuracy),
        "accuracy": float((true_positive + true_negative) / len(labels)),
        "precision": float(precision),
        "recall": float(positive_recall),
        "f1": float(2 * precision * positive_recall / max(1e-15, precision + positive_recall)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved detector on an external real/fake directory"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        help="Optional smoke-test limit; omit for the complete external dataset",
    )
    args = parser.parse_args()

    import torch

    from ..data import IMAGE_SUFFIXES, load_labeled_paths
    from ..features import extract_features
    from ..model import FrozenEncoders, load_checkpoint

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    rows = _limit_per_class(load_labeled_paths(args.data_dir), args.max_images_per_class)
    if any(path.suffix.lower() not in IMAGE_SUFFIXES for path, _ in rows):
        raise ValueError("Manifest contains an unsupported image suffix")

    device = torch.device(args.device)
    head, config, temperature, metadata = load_checkpoint(args.checkpoint, device)
    encoders = FrozenEncoders(config, device)
    features, labels_tensor, paths = extract_features(
        rows, encoders, args.batch_size, transform_mode="clean"
    )
    model_features = (
        features
        if config.quality_dim
        else features[:, : config.clip_dim + config.forensic_dim]
    )
    with torch.inference_mode():
        probabilities = torch.sigmoid(head(model_features.to(device)) / temperature).cpu().numpy()

    labels = labels_tensor.numpy().astype(int)
    threshold = float(metadata.get("threshold", 0.5))
    predictions = (probabilities >= threshold).astype(int)
    sources = [_source(path, args.data_dir) for path in paths]

    per_source = {}
    for source in sorted(set(sources)):
        mask = np.asarray([value == source for value in sources])
        per_source[source] = {
            "count": int(mask.sum()),
            "metrics": _metrics(labels[mask], probabilities[mask], threshold),
        }
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "split": "external_test",
        "image_count": len(paths),
        "threshold": threshold,
        "overall": _metrics(labels, probabilities, threshold),
        "per_source": per_source,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv"
    summary_path = args.output_dir / "summary.json"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "image_path",
                "split",
                "source",
                "true_label",
                "predicted_probability",
                "predicted_label",
            )
        )
        writer.writerows(
            (path, "external_test", source, int(label), float(probability), int(prediction))
            for path, source, label, probability, prediction in zip(
                paths, sources, labels, probabilities, predictions, strict=True
            )
        )
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Predictions: {predictions_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
