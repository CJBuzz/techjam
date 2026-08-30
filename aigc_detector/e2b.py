from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from .data import DeterministicTransform, ROBUSTNESS_CONDITIONS, RobustTransform
from .metrics import classification_metrics


ATOMIC_VIEWS = ("identity", "jpeg90", "resize0.75", "resize0.5", "blur0.5")
POLICIES = OrderedDict((
    ("identity", ("identity",)),
    ("identity+jpeg90", ("identity", "jpeg90")),
    ("identity+resize0.75", ("identity", "resize0.75")),
    ("identity+resize0.5", ("identity", "resize0.5")),
    ("identity+blur0.5", ("identity", "blur0.5")),
    ("identity+jpeg90+resize0.5", ("identity", "jpeg90", "resize0.5")),
    ("identity+jpeg90+resize0.75", ("identity", "jpeg90", "resize0.75")),
    ("identity+resize0.75+resize0.5", ("identity", "resize0.75", "resize0.5")),
    ("identity+jpeg90+blur0.5", ("identity", "jpeg90", "blur0.5")),
    ("identity+resize0.75+blur0.5", ("identity", "resize0.75", "blur0.5")),
    ("identity+jpeg90+resize0.75+blur0.5", ("identity", "jpeg90", "resize0.75", "blur0.5")),
    ("identity+jpeg90+resize0.75+resize0.5", ("identity", "jpeg90", "resize0.75", "resize0.5")),
))
AGGREGATIONS = ("mean", "median")


def atomic_view(image: Image.Image, name: str, seed: int, identity: str) -> Image.Image:
    """Apply one deterministic mild inference redistribution to an already-conditioned image."""
    image = image.convert("RGB")
    if name == "identity":
        return image.copy()
    condition = {
        "jpeg90": "jpeg_q90",
        "resize0.5": "resize_x0.5",
        "blur0.5": "blur_s0.5",
    }.get(name)
    if condition:
        return DeterministicTransform(condition, seed, identity, 0)(image.copy())
    if name == "resize0.75":
        return RobustTransform._apply_one(image.copy(), "resize", 0.75)
    raise ValueError(f"Unknown atomic TTA view: {name}")


def construct_policy_views(
    image: Image.Image, policy_name: str, seed: int, identity: str
) -> list[Image.Image]:
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown curated policy: {policy_name}")
    return [atomic_view(image, view, seed, identity) for view in POLICIES[policy_name]]


def aggregate_logits(logits: torch.Tensor, aggregation: str) -> torch.Tensor:
    if logits.ndim < 2 or logits.shape[-1] < 1:
        raise ValueError("Expected a non-empty atomic-view dimension")
    if aggregation == "mean":
        return logits.mean(dim=-1)
    if aggregation == "median":
        return torch.quantile(logits, 0.5, dim=-1)
    raise ValueError(f"Unknown logit aggregation: {aggregation}")


def require_validation_search(split: str) -> None:
    if split != "validation":
        raise ValueError("E2b policy search is validation-only; final-test search is forbidden")


def _rows_fingerprint(rows: list[tuple[Path, int]], root: Path) -> str:
    digest = hashlib.sha256()
    for path, label in sorted(rows, key=lambda row: str(row[0])):
        stat = path.stat()
        relative = path.resolve().relative_to(root.resolve())
        digest.update(f"{relative}\0{label}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def load_or_build_logit_cache(
    path: Path, manifest: dict, builder: Callable[[], dict]
) -> dict:
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if cached.get("manifest") != manifest:
            raise ValueError("Existing E2b atomic-logit cache is incompatible")
        return cached
    payload = builder()
    payload["manifest"] = manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def extract_atomic_logits(
    rows: list[tuple[Path, int]], conditions: tuple[str, ...], atomic_views: tuple[str, ...],
    encoders: torch.nn.Module, head: torch.nn.Module, model_config: object,
    batch_size: int, seed: int, device: torch.device,
) -> dict:
    condition_outputs = []
    labels_reference = torch.tensor([label for _, label in rows], dtype=torch.float32)
    for condition_index, condition in enumerate(conditions):
        batches = []
        for start in range(0, len(rows), batch_size):
            images = []
            batch_rows = rows[start : start + batch_size]
            for path, _ in batch_rows:
                official = DeterministicTransform(condition, seed, str(path), condition_index)
                with Image.open(path) as source:
                    conditioned = official(source.convert("RGB"))
                images.extend(
                    atomic_view(conditioned, view, seed, f"{path}:{condition}")
                    for view in atomic_views
                )
            features = encoders(images)
            model_features = features if model_config.quality_dim else features[
                :, : model_config.clip_dim + model_config.forensic_dim
            ]
            with torch.inference_mode():
                logits = head(model_features.to(device)).view(len(batch_rows), len(atomic_views)).cpu()
            batches.append(logits)
        condition_outputs.append(torch.cat(batches))
    return {
        "logits": torch.stack(condition_outputs),
        "labels": labels_reference,
        "conditions": list(conditions),
        "atomic_views": list(atomic_views),
    }


def score_policy(
    cache: dict, policy_name: str, aggregation: str, temperature: float, threshold: float,
    evaluation_split: str = "validation",
) -> dict:
    indices = [cache["atomic_views"].index(name) for name in POLICIES[policy_name]]
    selected = cache["logits"][:, :, indices]
    probabilities = torch.sigmoid(aggregate_logits(selected, aggregation) / temperature)
    labels = cache["labels"]
    per_condition = {
        condition: classification_metrics(labels, probabilities[index], threshold)
        for index, condition in enumerate(cache["conditions"])
    }
    transformed = [name for name in cache["conditions"] if name != "clean"]
    baccs = [per_condition[name]["balanced_accuracy"] for name in transformed]
    aucs = [per_condition[name]["roc_auc"] for name in transformed]
    fprs = [per_condition[name]["false_positive_rate"] for name in transformed]
    worst_index = int(np.argmin(baccs))
    clean = per_condition["clean"]
    return {
        "policy_name": policy_name,
        "atomic_views": list(POLICIES[policy_name]),
        "aggregation": aggregation,
        "number_of_views": len(POLICIES[policy_name]),
        "inference_multiplier": f"{len(POLICIES[policy_name])}x",
        "evaluation_split": evaluation_split,
        "selection_split": "validation" if evaluation_split == "validation" else None,
        "test_rows_used_for_selection": False,
        "clean_validation_balanced_accuracy": clean["balanced_accuracy"],
        "mean_transformed_validation_balanced_accuracy": float(np.mean(baccs)),
        "worst_transformed_validation_balanced_accuracy": float(baccs[worst_index]),
        "worst_condition": transformed[worst_index],
        "per_condition_balanced_accuracy": {
            name: metrics["balanced_accuracy"] for name, metrics in per_condition.items()
        },
        "mean_transformed_roc_auc": float(np.mean(aucs)),
        "worst_transformed_roc_auc": float(np.min(aucs)),
        "clean_false_positive_rate": clean["false_positive_rate"],
        "mean_transformed_false_positive_rate": float(np.mean(fprs)),
    }


def rank_policy_results(results: list[dict], split: str = "validation") -> list[dict]:
    require_validation_search(split)
    identity = next(
        row for row in results
        if row["policy_name"] == "identity" and row["aggregation"] == "mean"
    )
    clean_floor = identity["clean_validation_balanced_accuracy"] - 0.01
    eligible = [row for row in results if row["clean_validation_balanced_accuracy"] >= clean_floor]
    eligible.sort(key=lambda row: (
        row["worst_transformed_validation_balanced_accuracy"],
        row["mean_transformed_validation_balanced_accuracy"],
        -row["number_of_views"],
    ), reverse=True)
    ranks = {(row["policy_name"], row["aggregation"]): rank for rank, row in enumerate(eligible, 1)}
    return [{
        **row,
        "clean_constraint_pass": row["clean_validation_balanced_accuracy"] >= clean_floor,
        "rank": ranks.get((row["policy_name"], row["aggregation"])),
    } for row in results]


def write_search_outputs(results: list[dict], output_dir: Path, checkpoint: Path, seed: int) -> dict:
    ranked = rank_policy_results(results)
    winner = min((row for row in ranked if row["rank"] is not None), key=lambda row: row["rank"])
    output_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "selection_split": "validation", "test_rows_used": False,
        "checkpoint": str(checkpoint.resolve()),
        "ranking_policy": {
            "primary": "worst transformed validation balanced accuracy",
            "clean_constraint": "within 0.01 of identity validation clean balanced accuracy",
            "tie_breaks": ["mean transformed validation balanced accuracy", "fewer views"],
        },
        "results": ranked,
    }
    (output_dir / "tta_policy_summary.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    scalar_keys = [key for key, value in ranked[0].items() if not isinstance(value, (dict, list))]
    with (output_dir / "tta_policy_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys + ["atomic_views", "per_condition_balanced_accuracy"])
        writer.writeheader()
        for row in ranked:
            writer.writerow({key: row.get(key) for key in scalar_keys} | {
                "atomic_views": json.dumps(row["atomic_views"]),
                "per_condition_balanced_accuracy": json.dumps(row["per_condition_balanced_accuracy"], sort_keys=True),
            })
    locked = {
        "schema_version": 1,
        "selection_split": "validation",
        "test_rows_used_for_selection": False,
        "checkpoint": str(checkpoint.resolve()),
        "policy_name": winner["policy_name"],
        "atomic_views": winner["atomic_views"],
        "aggregation": winner["aggregation"],
        "seed": seed,
    }
    (output_dir / "winning_policy.json").write_text(json.dumps(locked, indent=2) + "\n", encoding="utf-8")
    return locked


def load_locked_policy(path: Path) -> dict:
    locked = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "selection_split", "test_rows_used_for_selection", "checkpoint",
        "policy_name", "atomic_views", "aggregation", "seed",
    }
    if set(locked) != required:
        raise ValueError("Locked policy must contain exactly one winner configuration")
    if locked["selection_split"] != "validation" or locked["test_rows_used_for_selection"] is not False:
        raise ValueError("Locked policy was not selected exclusively on validation")
    if locked["policy_name"] not in POLICIES or locked["atomic_views"] != list(POLICIES[locked["policy_name"]]):
        raise ValueError("Locked policy atomic views do not match the curated definition")
    if locked["aggregation"] not in AGGREGATIONS:
        raise ValueError("Locked policy aggregation is unsupported")
    return locked


def _load_rows(data_dir: Path, split: str, seed: int) -> list[tuple[Path, int]]:
    from .data import load_labeled_paths, stratified_train_val_test_split
    all_rows = load_labeled_paths(data_dir)
    _, validation, test = stratified_train_val_test_split(all_rows, data_dir, 0.15, 0.15, seed)
    return validation if split == "validation" else test


def search_command(args: argparse.Namespace) -> None:
    require_validation_search("validation")
    from .model import FrozenEncoders, load_checkpoint
    from .train import choose_device
    device = choose_device(args.device)
    head, config, temperature, metadata = load_checkpoint(args.checkpoint, device)
    rows = _load_rows(args.data_dir, "validation", args.seed)
    manifest = {
        "schema_version": 1, "split": "validation", "test_rows_used": False,
        "dataset_fingerprint": _rows_fingerprint(rows, args.data_dir),
        "checkpoint": str(args.checkpoint.resolve()), "conditions": list(ROBUSTNESS_CONDITIONS),
        "checkpoint_size": args.checkpoint.stat().st_size,
        "checkpoint_mtime_ns": args.checkpoint.stat().st_mtime_ns,
        "atomic_views": list(ATOMIC_VIEWS), "seed": args.seed,
    }
    def builder() -> dict:
        encoders = FrozenEncoders(config, device)
        payload = extract_atomic_logits(
            rows, ROBUSTNESS_CONDITIONS, ATOMIC_VIEWS, encoders, head, config,
            args.batch_size, args.seed, device,
        )
        del encoders
        return payload
    cache = load_or_build_logit_cache(args.output_dir / "validation_atomic_logits.pt", manifest, builder)
    threshold = float(metadata.get("threshold", 0.5))
    results = [
        score_policy(cache, policy, aggregation, temperature, threshold)
        for policy in POLICIES for aggregation in AGGREGATIONS
    ]
    write_search_outputs(results, args.output_dir, args.checkpoint, args.seed)


def locked_test_command(args: argparse.Namespace) -> None:
    locked = load_locked_policy(args.locked_policy)
    from .model import FrozenEncoders, load_checkpoint
    from .train import choose_device
    checkpoint = Path(locked["checkpoint"])
    device = choose_device(args.device)
    head, config, temperature, metadata = load_checkpoint(checkpoint, device)
    rows = _load_rows(args.data_dir, "test", int(locked["seed"]))
    encoders = FrozenEncoders(config, device)
    cache = extract_atomic_logits(
        rows, ROBUSTNESS_CONDITIONS, tuple(locked["atomic_views"]), encoders, head, config,
        args.batch_size, int(locked["seed"]), device,
    )
    del encoders
    result = score_policy(
        cache, locked["policy_name"], locked["aggregation"], temperature,
        float(metadata.get("threshold", 0.5)), evaluation_split="test",
    )
    output = {
        "evaluation_split": "test", "policy_search_performed": False,
        "locked_policy": locked, "result": result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "locked_test_metrics.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="E2b validation-only mild TTA policy search")
    subparsers = parser.add_subparsers(dest="command", required=True)
    search = subparsers.add_parser("search")
    search.add_argument("--data-dir", type=Path, required=True)
    search.add_argument("--checkpoint", type=Path, required=True)
    search.add_argument("--output-dir", type=Path, required=True)
    search.add_argument("--batch-size", type=int, default=8)
    search.add_argument("--seed", type=int, default=42)
    search.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    search.set_defaults(handler=search_command)
    locked = subparsers.add_parser("locked-test")
    locked.add_argument("--data-dir", type=Path, required=True)
    locked.add_argument("--locked-policy", type=Path, required=True)
    locked.add_argument("--output-dir", type=Path, required=True)
    locked.add_argument("--batch-size", type=int, default=8)
    locked.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    locked.set_defaults(handler=locked_test_command)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
