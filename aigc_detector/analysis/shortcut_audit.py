from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from ..metrics import classification_metrics, expected_calibration_error
from ..model import load_checkpoint
from ..train import choose_device


def _records(manifest: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["split"]].append(row)
    return dict(result)


def _leakage_report(records: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    path_counts: Counter[str] = Counter()
    for split, rows in records.items():
        for row in rows:
            group_splits[row["duplicate_group"]].add(split)
            hash_splits[row["content_sha256"]].add(split)
            path_counts[row["path"]] += 1
    cross_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    cross_hashes = sorted(value for value, splits in hash_splits.items() if len(splits) > 1)
    repeated_paths = sorted(path for path, count in path_counts.items() if count > 1)
    return {
        "rows": sum(len(rows) for rows in records.values()),
        "duplicate_groups": len(group_splits),
        "cross_split_duplicate_groups": len(cross_groups),
        "cross_split_exact_hashes": len(cross_hashes),
        "repeated_manifest_paths": len(repeated_paths),
        "examples": {
            "cross_split_duplicate_groups": cross_groups[:20],
            "cross_split_exact_hashes": cross_hashes[:20],
            "repeated_manifest_paths": repeated_paths[:20],
        },
        "passed": not cross_groups and not cross_hashes and not repeated_paths,
    }


def _source_probe(
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    train_sources: list[str],
    val_sources: list[str],
    seed: int,
) -> dict[str, object]:
    classes = sorted(set(train_sources))
    if classes != sorted(set(val_sources)) or len(classes) != 2:
        raise ValueError(f"Source probe currently requires the same two sources in both splits; got {classes}")
    mapping = {source: index for index, source in enumerate(classes)}
    train_y = np.array([mapping[source] for source in train_sources])
    val_y = np.array([mapping[source] for source in val_sources])
    classifier = SGDClassifier(
        loss="log_loss", alpha=1e-4, max_iter=1000, tol=1e-4,
        class_weight="balanced", random_state=seed, average=True,
    )
    classifier.fit(train_x.numpy(), train_y)
    probabilities = classifier.predict_proba(val_x.numpy())[:, 1]
    predictions = probabilities >= 0.5
    return {
        "positive_source": classes[1],
        "accuracy": float(accuracy_score(val_y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(val_y, predictions)),
        "roc_auc": float(roc_auc_score(val_y, probabilities)),
        "train_rows": len(train_y),
        "model_selection_rows": len(val_y),
    }


def _resolution_bucket(row: dict[str, str]) -> str:
    longest = max(int(row["width"]), int(row["height"]))
    if longest <= 64:
        return "low_le64"
    if longest <= 255:
        return "medium_65_255"
    return "high_ge256"


def _error_cases(
    records: dict[str, list[dict[str, str]]],
    probabilities: torch.Tensor,
    threshold: float,
    limit: int = 20,
) -> dict[str, list[dict[str, object]]]:
    train_rows = records["train"]
    train_hashes = [int(row["dhash64"], 16) for row in train_rows]
    val_rows = records["model_selection"]

    def describe(index: int) -> dict[str, object]:
        row = val_rows[index]
        query_hash = int(row["dhash64"], 16)
        nearest_index, distance = min(
            enumerate((query_hash ^ value).bit_count() for value in train_hashes),
            key=lambda item: item[1],
        )
        nearest = train_rows[nearest_index]
        return {
            "path": row["path"],
            "label": int(row["label"]),
            "ai_probability": float(probabilities[index]),
            "source": row["source"],
            "width": int(row["width"]),
            "height": int(row["height"]),
            "duplicate_group": row["duplicate_group"],
            "nearest_train_path": nearest["path"],
            "nearest_train_dhash_distance": distance,
            "nearest_train_label": int(nearest["label"]),
            "nearest_train_source": nearest["source"],
        }

    false_positive_indices = [
        index for index, row in enumerate(val_rows)
        if int(row["label"]) == 0 and probabilities[index] >= threshold
    ]
    false_negative_indices = [
        index for index, row in enumerate(val_rows)
        if int(row["label"]) == 1 and probabilities[index] < threshold
    ]
    false_positive_indices.sort(key=lambda index: float(probabilities[index]), reverse=True)
    false_negative_indices.sort(key=lambda index: float(probabilities[index]))
    return {
        "false_positives": [describe(index) for index in false_positive_indices[:limit]],
        "false_negatives": [describe(index) for index in false_negative_indices[:limit]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit duplicate leakage, source shortcuts, and resolution dependence")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional baseline checkpoint for resolution-bucket detector metrics")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = _records(args.split_manifest)
    required = {"train", "model_selection", "calibration", "test"}
    if set(records) != required:
        raise ValueError(f"Manifest splits must be {sorted(required)}")
    leakage = _leakage_report(records)
    if not leakage["passed"]:
        raise RuntimeError(f"Cross-split leakage audit failed: {leakage}")

    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=True)
    train_originals = len(records["train"])
    train_x = cache["train_features"][:train_originals]
    train_y = cache["train_labels"][:train_originals]
    val_x, val_y = cache["val_features"], cache["val_labels"]
    if len(val_x) != len(records["model_selection"]) or len(train_x) != train_originals:
        raise ValueError("Feature-cache row counts do not align with manifest splits")
    manifest_train_y = torch.tensor([int(row["label"]) for row in records["train"]], dtype=torch.float32)
    manifest_val_y = torch.tensor([int(row["label"]) for row in records["model_selection"]], dtype=torch.float32)
    if not torch.equal(train_y, manifest_train_y) or not torch.equal(val_y, manifest_val_y):
        raise ValueError("Feature-cache labels do not align with manifest order")

    train_sources = [row["source"].lower() for row in records["train"]]
    val_sources = [row["source"].lower() for row in records["model_selection"]]
    width = train_x.shape[1]
    modalities: dict[str, tuple[slice, slice] | slice] = {
        "clip": slice(0, 512),
        "laplacian": slice(512, 1792),
        "clip_laplacian": slice(0, 1792),
    }
    if width >= 3072:
        modalities.update({
            "fft": slice(1792, 3072),
            "fused": slice(0, 3072),
        })
    probes = {
        name: _source_probe(train_x[:, columns], val_x[:, columns], train_sources, val_sources, args.seed)
        for name, columns in modalities.items()
    }

    resolution_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for split, rows in records.items():
        for row in rows:
            resolution_counts[split][_resolution_bucket(row)] += 1
    report: dict[str, object] = {
        "_metadata": {
            "data_dir": str(args.data_dir),
            "split_manifest": str(args.split_manifest),
            "feature_cache": str(args.feature_cache),
            "seed": args.seed,
        },
        "leakage": leakage,
        "source_probes": probes,
        "resolution_counts": {split: dict(counts) for split, counts in resolution_counts.items()},
    }
    if args.checkpoint:
        device = choose_device(args.device)
        head, config, temperature, checkpoint_metadata = load_checkpoint(args.checkpoint, device)
        threshold = float(checkpoint_metadata.get("threshold", 0.5))
        expected_width = config.clip_dim + config.forensic_dim
        if val_x.shape[1] != expected_width:
            raise ValueError(f"Checkpoint expects {expected_width} features, cache has {val_x.shape[1]}")
        with torch.no_grad():
            probabilities = torch.sigmoid(head(val_x.to(device)) / temperature).cpu()
        groups = {
            "source": val_sources,
            "resolution": [_resolution_bucket(row) for row in records["model_selection"]],
        }
        detector_breakdown = {}
        for grouping, values in groups.items():
            detector_breakdown[grouping] = {}
            for value in sorted(set(values)):
                mask = torch.tensor([item == value for item in values], dtype=torch.bool)
                metrics = classification_metrics(val_y[mask], probabilities[mask], threshold)
                metrics["ece"] = expected_calibration_error(val_y[mask], probabilities[mask])
                metrics["images"] = int(mask.sum())
                detector_breakdown[grouping][value] = metrics
        report["detector_breakdown"] = detector_breakdown
        report["representative_errors"] = _error_cases(records, probabilities, threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
