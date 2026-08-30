from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .data import ExactSeverityTransform, SEVERITY_SPECS, load_split_manifest, severity_key
from .features import extract_features_with_factory
from .metrics import classification_metrics, expected_calibration_error
from .model import FrozenEncoders, load_checkpoint
from .train import choose_device


def _source(path: str, root: Path) -> str:
    parts = Path(path).resolve().relative_to(root.resolve()).parts
    return parts[1].lower() if len(parts) >= 3 else "default"


def _summary(cells: dict[str, dict[str, object]]) -> dict[str, object]:
    metrics = [cell["overall"] for cell in cells.values()]
    names = ("accuracy", "roc_auc", "average_precision", "brier", "ece")
    macro = {name: float(np.mean([row[name] for row in metrics])) for name in names}
    return {
        "macro": macro,
        "worst_accuracy": min(
            ({"condition": key, "value": cell["overall"]["accuracy"]} for key, cell in cells.items()),
            key=lambda row: row["value"],
        ),
        "worst_roc_auc": min(
            ({"condition": key, "value": cell["overall"]["roc_auc"]} for key, cell in cells.items()),
            key=lambda row: row["value"],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the exact challenge condition-by-severity matrix")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("model_selection", "calibration", "test"), default="model_selection")
    parser.add_argument("--allow-test", action="store_true", help="Required acknowledgement before reading reserved test images")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Skip cells already present in the output JSON")
    parser.add_argument("--only", action="append", default=[], help="Optional exact cell key; repeat to run a subset")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError("Reserved test evaluation requires --allow-test after the full pipeline is locked")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    head, config, temperature, metadata = load_checkpoint(args.checkpoint, device)
    encoders = FrozenEncoders(config, device)
    rows = load_split_manifest(args.data_dir, args.split_manifest)[args.split]
    selected = [(operation, value) for operation, value in SEVERITY_SPECS if not args.only or severity_key(operation, value) in args.only]
    known = {severity_key(operation, value) for operation, value in SEVERITY_SPECS}
    unknown = set(args.only) - known
    if unknown:
        raise ValueError(f"Unknown --only cells: {sorted(unknown)}; expected keys from {sorted(known)}")

    results: dict[str, object] = {
        "_metadata": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_training_metadata": metadata,
            "split": args.split,
            "split_images": len(rows),
            "split_manifest": str(args.split_manifest),
            "seed": args.seed,
            "temperature": temperature,
            "color_definition": "path-keyed independent +/- magnitude for brightness, contrast, saturation",
        },
        "cells": {},
    }
    if args.resume and args.output.is_file():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("_metadata", {}).get("checkpoint") != str(args.checkpoint) or prior.get("_metadata", {}).get("split") != args.split:
            raise ValueError("Resume output checkpoint/split does not match this run")
        results = prior
    cells = results["cells"]
    for operation, value in selected:
        key = severity_key(operation, value)
        if key in cells:
            print(f"Skipping completed cell: {key}")
            continue
        features, labels, paths = extract_features_with_factory(
            rows,
            encoders,
            args.batch_size,
            lambda path, _index, op=operation, val=value: ExactSeverityTransform(op, val, args.seed, str(path)),
            key,
        )
        with torch.no_grad():
            probabilities = torch.sigmoid(head(features.to(device)) / temperature).cpu()
        sources = [_source(path, args.data_dir) for path in paths]
        overall = classification_metrics(labels, probabilities)
        overall["ece"] = expected_calibration_error(labels, probabilities)
        by_source = {}
        for source in sorted(set(sources)):
            mask = torch.tensor([item == source for item in sources], dtype=torch.bool)
            source_metrics = classification_metrics(labels[mask], probabilities[mask])
            source_metrics["ece"] = expected_calibration_error(labels[mask], probabilities[mask])
            by_source[source] = source_metrics
        cells[key] = {
            "operation": operation,
            "value": value,
            "overall": overall,
            "by_source": by_source,
        }
        results["summary"] = _summary(cells)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: cells[key], "running_summary": results["summary"]}, indent=2))


if __name__ == "__main__":
    main()
