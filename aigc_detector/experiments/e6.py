from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..data import ROBUSTNESS_CONDITIONS, RobustTransform, load_labeled_paths, stratified_train_val_test_split
from .e4a import assemble_validation_matrix
from ..metrics import classification_metrics, fit_temperature, select_threshold
from ..model import FrozenEncoders, FusionHead, ModelConfig, load_checkpoint, save_checkpoint
from ..train import choose_device


SCALES = (1.0, 0.75, 0.50, 0.25)
DEFAULT_SCALE_WEIGHTS = (0.5, 1.0, 1.5)
DEFAULT_LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.40)
CONSISTENCY_MODES = ("logit_symmetric", "logit_asymmetric", "representation_asymmetric")


class ScaleTransform:
    def __init__(self, scale: float) -> None:
        if scale not in SCALES:
            raise ValueError(f"Unsupported E6 scale: {scale}")
        self.scale = scale

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        return image.copy() if self.scale == 1.0 else RobustTransform._apply_one(image, "resize", self.scale)


class ScaleDataset(Dataset):
    def __init__(self, rows: list[tuple[Path, int]], scale: float) -> None:
        self.rows, self.transform = rows, ScaleTransform(scale)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        path, label = self.rows[index]
        with Image.open(path) as source:
            return self.transform(source.convert("RGB")), label


def _collate(batch):
    images, labels = zip(*batch, strict=True)
    return list(images), torch.tensor(labels, dtype=torch.float32)


def representation(head: FusionHead, features: torch.Tensor) -> torch.Tensor:
    return head.network[:-1](features)


def scale_consistency_loss(
    scale_logits: torch.Tensor,
    mode: str,
    scale_weights: torch.Tensor,
    scale_representations: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale axis is first: clean teacher at index 0, three resized students thereafter."""
    if mode not in CONSISTENCY_MODES:
        raise ValueError(f"Unknown E6 consistency mode: {mode}")
    if scale_logits.ndim != 2 or scale_logits.shape[0] != 4 or scale_weights.shape != (3,):
        raise ValueError("Expected four scale logits and three non-clean scale weights")
    weights = scale_weights.to(scale_logits).view(3, 1)
    if mode == "logit_symmetric":
        losses = (scale_logits[1:] - scale_logits[0:1]) ** 2
    elif mode == "logit_asymmetric":
        losses = (scale_logits[1:] - scale_logits[0:1].detach()) ** 2
    else:
        if scale_representations is None or scale_representations.shape[:2] != scale_logits.shape:
            raise ValueError("representation_asymmetric requires aligned scale representations")
        normalized = F.normalize(scale_representations, dim=2)
        teacher = normalized[0:1].detach()
        losses = ((normalized[1:] - teacher) ** 2).sum(2)
    return (losses * weights).sum() / (weights.sum() * scale_logits.shape[1])


def add_scale_objective(base_loss: torch.Tensor, scale_loss: torch.Tensor, lambda_scale: float) -> torch.Tensor:
    if lambda_scale < 0:
        raise ValueError("lambda_scale must be non-negative")
    return base_loss + lambda_scale * scale_loss


def require_validation_selection(split: str) -> None:
    if split != "validation":
        raise ValueError("E6 configuration selection is validation-only; final-test selection is forbidden")


def scale_cache_manifest(base: dict, split: str, scale: float, row_count: int) -> dict:
    return {
        "schema_version": 1, "experiment": "E6", "split": split, "scale": scale,
        "row_count": row_count, "base_cache_manifest": base["manifest"],
        "interpolation": "official RobustTransform resize: bilinear downsample then bilinear restore",
        "raw_images_persisted": False,
    }


def scale_cache_valid(path: Path, expected_manifest: dict) -> bool:
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return payload.get("manifest") == expected_manifest and len(payload.get("features", ())) == expected_manifest["row_count"]


def _scale_name(scale: float) -> str:
    return f"scale_{scale:.2f}".replace(".", "p")


def cache_command(args: argparse.Namespace) -> None:
    base = torch.load(args.base_cache, map_location="cpu", weights_only=True, mmap=True)
    rows = load_labeled_paths(args.data_dir)
    train_rows, val_rows, _ = stratified_train_val_test_split(
        rows, args.data_dir, base["manifest"]["validation_fraction"],
        base["manifest"]["test_fraction"], base["manifest"]["seed"],
    )
    config = ModelConfig(**base["manifest"]["model_config"])
    device = choose_device(args.device)
    encoders = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in (("train", train_rows),):
        for scale in SCALES[1:]:
            path = args.output_dir / f"{split}_{_scale_name(scale)}.pt"
            manifest = scale_cache_manifest(base, split, scale, len(split_rows))
            if scale_cache_valid(path, manifest):
                print(f"Reusing E6 scale cache: {path}")
                continue
            if path.exists():
                raise ValueError(f"Incompatible E6 scale cache; refusing overwrite: {path}")
            if encoders is None:
                encoders = FrozenEncoders(config, device)
            loader = DataLoader(
                ScaleDataset(split_rows, scale), batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, collate_fn=_collate,
            )
            features, labels = [], []
            for images, batch_labels in loader:
                features.append(encoders(images))
                labels.append(batch_labels)
            payload = {"features": torch.cat(features), "labels": torch.cat(labels), "manifest": manifest}
            temporary = path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, path)
    if encoders is not None:
        del encoders
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1, "scales": list(SCALES), "base_cache": str(args.base_cache.resolve()),
        "complete_files": sorted(path.name for path in args.output_dir.glob("*.pt")),
    }, indent=2) + "\n", encoding="utf-8")


def load_scale_features(base: dict, directory: Path, split: str, originals: int) -> torch.Tensor:
    clean = base["train_features"][:originals] if split == "train" else base["val_features"]
    views = [clean]
    for scale in SCALES[1:]:
        path = directory / f"{split}_{_scale_name(scale)}.pt"
        expected = scale_cache_manifest(base, split, scale, originals)
        if not scale_cache_valid(path, expected):
            raise ValueError(f"Missing or incompatible E6 scale cache: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not torch.equal(payload["labels"], base["train_labels"][:originals] if split == "train" else base["val_labels"]):
            raise ValueError(f"Scale-cache labels do not align: {path}")
        views.append(payload["features"])
    return torch.stack(views)


def _validation_metrics(head, clean_x, clean_y, robust_x, robust_y, conditions, temperature, threshold):
    head.cpu().eval()
    with torch.no_grad():
        clean_p = torch.sigmoid(head(clean_x) / temperature)
        robust_p = torch.sigmoid(head(robust_x) / temperature)
    clean = classification_metrics(clean_y, clean_p, threshold)
    per_condition = {}
    for condition in ROBUSTNESS_CONDITIONS[1:]:
        mask = torch.tensor([value == condition for value in conditions])
        per_condition[condition] = classification_metrics(robust_y[mask], robust_p[mask], threshold)
    return clean, per_condition


def train_config(
    config_record: dict, base: dict, scale_train: torch.Tensor,
    robust_x: torch.Tensor, robust_y: torch.Tensor, robust_conditions: list[str],
    initialize: Path, output_dir: Path, device: torch.device,
    epochs: int, patience: int, batch_size: int, learning_rate: float, seed: int,
) -> dict:
    torch.manual_seed(seed)
    source, model_config, source_temperature, source_metadata = load_checkpoint(initialize, torch.device("cpu"))
    if asdict(model_config) != base["manifest"]["model_config"]:
        raise ValueError("E6 initialization checkpoint does not match frozen features")
    head = FusionHead(model_config).to(device)
    head.load_state_dict(source.state_dict())
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_x, train_y, groups = base["train_features"], base["train_labels"], base["train_groups"]
    originals = scale_train.shape[1]
    repeats = len(train_y) // originals
    loader = DataLoader(torch.arange(originals), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    group_names = sorted(set(groups))
    scale_weights = torch.tensor(config_record["scale_weights"], device=device)
    clean_x, clean_y = base["val_features"], base["val_labels"]
    with torch.no_grad():
        source_clean_probabilities = torch.sigmoid(source(clean_x) / source_temperature)
    source_clean = classification_metrics(
        clean_y, source_clean_probabilities, float(source_metadata.get("threshold", 0.5))
    )["balanced_accuracy"]
    clean_floor = source_clean - 0.01
    best_state, best_key, stale, logs = None, None, 0, []
    for epoch in range(1, epochs + 1):
        head.train()
        totals = {"classification": 0.0, "generic_consistency": 0.0, "worst_group": 0.0, "scale": 0.0, "batches": 0}
        for original_batch in loader:
            indices = torch.cat([original_batch + repeat * originals for repeat in range(repeats)])
            features, labels = train_x[indices].to(device), train_y[indices].to(device)
            batch_groups = [groups[index] for index in indices.tolist()]
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            classification = losses.mean()
            paired = logits.view(repeats, -1)
            generic = ((paired - paired.mean(0)) ** 2).mean()
            group_losses = [
                losses[torch.tensor([value == group for value in batch_groups], device=device)].mean()
                for group in group_names if group in batch_groups
            ]
            worst_group = torch.stack(group_losses).max()
            base_loss = classification + 0.05 * generic + 0.5 * worst_group
            if config_record["lambda_scale"] == 0:
                scale_loss = base_loss.new_zeros(())
            else:
                scale_batch = scale_train[:, original_batch].to(device)
                flat_scale = scale_batch.flatten(0, 1)
                scale_logits = head(flat_scale).view(4, len(original_batch))
                scale_reps = None
                if config_record["consistency_mode"] == "representation_asymmetric":
                    scale_reps = representation(head, flat_scale).view(4, len(original_batch), -1)
                scale_loss = scale_consistency_loss(
                    scale_logits, config_record["consistency_mode"], scale_weights, scale_reps
                )
            total = add_scale_objective(base_loss, scale_loss, config_record["lambda_scale"])
            total.backward()
            optimizer.step()
            for key, value in (("classification", classification), ("generic_consistency", generic),
                               ("worst_group", worst_group), ("scale", scale_loss)):
                totals[key] += float(value.detach())
            totals["batches"] += 1
        head.cpu().eval()
        with torch.no_grad():
            clean_logits, robust_logits = head(clean_x), head(robust_x)
        threshold = select_threshold(
            torch.cat((clean_y, robust_y)), torch.sigmoid(torch.cat((clean_logits, robust_logits))), "balanced"
        )
        clean_metric = classification_metrics(clean_y, torch.sigmoid(clean_logits), threshold)
        baccs = []
        for condition in ROBUSTNESS_CONDITIONS[1:]:
            mask = torch.tensor([value == condition for value in robust_conditions])
            baccs.append(classification_metrics(robust_y[mask], torch.sigmoid(robust_logits[mask]), threshold)["balanced_accuracy"])
        key = (
            clean_metric["balanced_accuracy"] >= clean_floor,
            min(baccs), float(np.mean(baccs)),
        )
        logs.append({"epoch": epoch, **{name: value / totals["batches"] for name, value in totals.items() if name != "batches"}})
        if best_key is None or key > best_key:
            best_key, stale = key, 0
            best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
        head.to(device)
    head.load_state_dict(best_state)
    head.cpu().eval()
    with torch.no_grad():
        clean_logits, robust_logits = head(clean_x), head(robust_x)
    calibration_logits, calibration_labels = torch.cat((clean_logits, robust_logits)), torch.cat((clean_y, robust_y))
    temperature = fit_temperature(calibration_logits, calibration_labels)
    threshold = select_threshold(calibration_labels, torch.sigmoid(calibration_logits / temperature), "balanced")
    clean, per_condition = _validation_metrics(
        head, clean_x, clean_y, robust_x, robust_y, robust_conditions, temperature, threshold
    )
    transformed = list(ROBUSTNESS_CONDITIONS[1:])
    baccs = [per_condition[name]["balanced_accuracy"] for name in transformed]
    aucs = [per_condition[name]["roc_auc"] for name in transformed]
    fprs = [per_condition[name]["false_positive_rate"] for name in transformed]
    checkpoint = output_dir / "model.pt"
    metadata = {"experiment": "E6", **config_record, "selection_split": "validation",
                "test_rows_used_for_selection": False, "threshold": threshold,
                "validation_metrics": clean, "robust_validation_metrics": per_condition}
    save_checkpoint(checkpoint, head, model_config, temperature, metadata)
    (output_dir / "training_log.json").write_text(json.dumps(logs, indent=2) + "\n", encoding="utf-8")
    worst_index = int(np.argmin(baccs))
    return {
        **config_record, "checkpoint": str(checkpoint), "status": "succeeded",
        "trainable_parameter_count": sum(p.numel() for p in head.parameters() if p.requires_grad),
        "selection_split": "validation", "test_rows_used_for_selection": False,
        "clean_validation_balanced_accuracy": clean["balanced_accuracy"],
        "mean_transformed_validation_balanced_accuracy": float(np.mean(baccs)),
        "worst_transformed_validation_balanced_accuracy": float(baccs[worst_index]),
        "resize_x0.5_balanced_accuracy": per_condition["resize_x0.5"]["balanced_accuracy"],
        "resize_x0.25_balanced_accuracy": per_condition["resize_x0.25"]["balanced_accuracy"],
        "blur_s2.0_balanced_accuracy": per_condition["blur_s2.0"]["balanced_accuracy"],
        "noise_s0.10_balanced_accuracy": per_condition["noise_s0.10"]["balanced_accuracy"],
        "worst_condition": transformed[worst_index],
        "clean_false_positive_rate": clean["false_positive_rate"],
        "mean_transformed_false_positive_rate": float(np.mean(fprs)),
        "mean_transformed_roc_auc": float(np.mean(aucs)), "worst_transformed_roc_auc": float(np.min(aucs)),
    }


def rank_results(rows: list[dict], split: str = "validation") -> list[dict]:
    require_validation_selection(split)
    baseline = next(row for row in rows if row.get("status") == "succeeded" and row["lambda_scale"] == 0)
    floor = baseline["clean_validation_balanced_accuracy"] - 0.01
    eligible = [row for row in rows if row.get("status") == "succeeded" and row["clean_validation_balanced_accuracy"] >= floor]
    eligible.sort(key=lambda row: (row["worst_transformed_validation_balanced_accuracy"],
                                   row["mean_transformed_validation_balanced_accuracy"]), reverse=True)
    ranks = {(row["consistency_mode"], row["lambda_scale"]): index for index, row in enumerate(eligible, 1)}
    return [{**row, "clean_constraint_pass": row.get("status") == "succeeded" and row["clean_validation_balanced_accuracy"] >= floor,
             "rank": ranks.get((row["consistency_mode"], row["lambda_scale"]))} for row in rows]


def sweep_command(args: argparse.Namespace) -> None:
    require_validation_selection("validation")
    base = torch.load(args.base_cache, map_location="cpu", weights_only=True, mmap=True)
    originals = int(base["train_original_indices"].max()) + 1
    scale_train = load_scale_features(base, args.scale_cache, "train", originals)
    extra = torch.load(args.validation_cache, map_location="cpu", weights_only=True, mmap=True)
    robust_x, robust_y, robust_conditions = assemble_validation_matrix(base, extra)
    device = choose_device(args.device)
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for lambda_scale in args.lambdas:
        record = {"consistency_mode": args.consistency_mode, "lambda_scale": lambda_scale,
                  "scale_weights": list(args.scale_weights)}
        name = f"{args.consistency_mode}_lambda_{lambda_scale:.2f}".replace(".", "p")
        directory = args.output_dir / "checkpoints" / name
        result_path = directory / "result.json"
        if result_path.is_file() and (directory / "model.pt").is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if all(result.get(key) == value for key, value in record.items()) and result.get("status") == "succeeded":
                rows.append(result)
                continue
        directory.mkdir(parents=True, exist_ok=True)
        try:
            result = train_config(record, base, scale_train, robust_x, robust_y, robust_conditions,
                                  args.initialize_from_checkpoint, directory, device, args.epochs,
                                  args.patience, args.batch_size, args.learning_rate, args.seed)
        except Exception as error:
            result = {**record, "status": "failed", "failure_reason": f"{type(error).__name__}: {error}"}
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        rows.append(result)
    ranked = rank_results(rows)
    config_dir = args.output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    for row in ranked:
        config_name = f"lambda_{row['lambda_scale']:.2f}".replace(".", "p") + ".json"
        (config_dir / config_name).write_text(
            json.dumps({key: row[key] for key in ("consistency_mode", "lambda_scale", "scale_weights")}, indent=2) + "\n"
        )
    document = {"selection_split": "validation", "test_rows_used_for_selection": False, "results": ranked}
    (args.output_dir / "validation_summary.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    scalar_keys = sorted({key for row in ranked for key, value in row.items() if not isinstance(value, (dict, list))})
    with (args.output_dir / "validation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys + ["scale_weights"])
        writer.writeheader()
        for row in ranked:
            writer.writerow({key: row.get(key) for key in scalar_keys} | {"scale_weights": json.dumps(row["scale_weights"])})
    winner = min((row for row in ranked if row["rank"] is not None), key=lambda row: row["rank"])
    (args.output_dir / "winning_config.json").write_text(json.dumps({
        "schema_version": 1, "selection_split": "validation", "test_rows_used_for_selection": False,
        "checkpoint": str(Path(winner["checkpoint"]).resolve()), "configuration": {
            key: winner[key] for key in ("consistency_mode", "lambda_scale", "scale_weights")
        }, "seed": args.seed,
    }, indent=2) + "\n", encoding="utf-8")


def locked_test_command(args: argparse.Namespace) -> None:
    locked = json.loads(args.winning_config.read_text(encoding="utf-8"))
    if locked.get("selection_split") != "validation" or locked.get("test_rows_used_for_selection") is not False:
        raise ValueError("E6 configuration was not locked using validation only")
    subprocess.run([
        sys.executable, "-m", "aigc_detector.evaluate", "--data-dir", str(args.data_dir),
        "--checkpoint", locked["checkpoint"], "--split", "test", "--profile", "full",
        "--tta", "none", "--seed", str(locked["seed"]), "--batch-size", str(args.batch_size),
        "--device", args.device, "--output-dir", str(args.output_dir),
    ], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="E6 targeted scale-consistency experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache = subparsers.add_parser("cache")
    cache.add_argument("--data-dir", type=Path, required=True)
    cache.add_argument("--base-cache", type=Path, required=True)
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument("--batch-size", type=int, default=8)
    cache.add_argument("--workers", type=int, default=4)
    cache.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    cache.set_defaults(handler=cache_command)
    sweep = subparsers.add_parser("sweep")
    sweep.add_argument("--base-cache", type=Path, required=True)
    sweep.add_argument("--validation-cache", type=Path, required=True)
    sweep.add_argument("--scale-cache", type=Path, required=True)
    sweep.add_argument("--initialize-from-checkpoint", type=Path, required=True)
    sweep.add_argument("--output-dir", type=Path, required=True)
    sweep.add_argument("--consistency-mode", choices=CONSISTENCY_MODES, default="logit_asymmetric")
    sweep.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS))
    sweep.add_argument("--scale-weights", type=float, nargs=3, default=list(DEFAULT_SCALE_WEIGHTS))
    sweep.add_argument("--epochs", type=int, default=30)
    sweep.add_argument("--patience", type=int, default=6)
    sweep.add_argument("--batch-size", type=int, default=32)
    sweep.add_argument("--learning-rate", type=float, default=1e-3)
    sweep.add_argument("--seed", type=int, default=42)
    sweep.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    sweep.set_defaults(handler=sweep_command)
    test = subparsers.add_parser("locked-test")
    test.add_argument("--data-dir", type=Path, required=True)
    test.add_argument("--winning-config", type=Path, required=True)
    test.add_argument("--output-dir", type=Path, required=True)
    test.add_argument("--batch-size", type=int, default=8)
    test.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    test.set_defaults(handler=locked_test_command)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
