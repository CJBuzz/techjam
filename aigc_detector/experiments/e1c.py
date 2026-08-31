from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..data import ROBUSTNESS_CONDITIONS
from ..features import extract_condition_tta_features
from ..metrics import classification_metrics, select_threshold
from ..model import FrozenEncoders, load_checkpoint
from ..train import choose_device


COARSE_ALPHAS = tuple(round(index / 10, 2) for index in range(11))
ENSEMBLE_MODES = {"raw": ("clean",), "mild3": ("clean", "jpeg_q90", "resize_x0.5")}


def blend_calibrated_logits(
    e0_raw_logits: torch.Tensor, e0_temperature: float,
    e1_raw_logits: torch.Tensor, e1_temperature: float, alpha: float,
) -> torch.Tensor:
    """Blend temperature-calibrated logits; alpha=1 is E0 and alpha=0 is E1."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return alpha * (e0_raw_logits / e0_temperature) + (1 - alpha) * (e1_raw_logits / e1_temperature)


def aggregate_checkpoint_views(raw_view_logits: torch.Tensor, mode: str) -> torch.Tensor:
    if mode not in ENSEMBLE_MODES:
        raise ValueError(f"Unknown controlled ensemble mode: {mode}")
    expected = 1 if mode == "raw" else 3
    if raw_view_logits.ndim != 2 or raw_view_logits.shape[1] < expected:
        raise ValueError(f"Mode {mode} requires at least {expected} ordered views")
    return raw_view_logits[:, 0] if mode == "raw" else raw_view_logits[:, :3].mean(1)


def require_validation_selection(split: str) -> None:
    if split != "validation":
        raise ValueError("E1c alpha selection is validation-only; final-test selection is forbidden")


def load_or_build_logit_cache(path: Path, manifest: dict, builder: Callable[[], dict]) -> dict:
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("manifest") != manifest:
            raise ValueError("Existing E1c validation-logit cache is incompatible")
        return payload
    payload = builder()
    payload["manifest"] = manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def _fingerprint(rows: list[tuple[Path, int]], root: Path) -> str:
    digest = hashlib.sha256()
    for path, label in sorted(rows, key=lambda row: str(row[0])):
        stat = path.stat()
        digest.update(
            f"{path.resolve().relative_to(root.resolve())}\0{label}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()


def extract_shared_logits(
    rows: list[tuple[Path, int]], conditions: tuple[str, ...], encoders: FrozenEncoders,
    models: list[tuple[torch.nn.Module, object]], batch_size: int, seed: int,
    device: torch.device,
) -> dict:
    """Encode mild3 views once, then run both heads on the shared frozen features."""
    outputs = [[] for _ in models]
    labels_reference = None
    for condition in conditions:
        features, labels, _ = extract_condition_tta_features(
            rows, encoders, batch_size, condition, seed, "mild3"
        )
        if labels_reference is None:
            labels_reference = labels
        elif not torch.equal(labels_reference, labels):
            raise ValueError("Condition labels do not align")
        for output, (head, config) in zip(outputs, models, strict=True):
            model_features = features if config.quality_dim else features[
                ..., : config.clip_dim + config.forensic_dim
            ]
            with torch.inference_mode():
                logits = head(model_features.flatten(0, 1).to(device)).view(model_features.shape[:2]).cpu()
            output.append(logits)
    return {
        "raw_view_logits_e0": torch.stack(outputs[0]),
        "raw_view_logits_e1": torch.stack(outputs[1]),
        "labels": labels_reference,
        "conditions": list(conditions),
        "views": list(ENSEMBLE_MODES["mild3"]),
    }


def score_alpha(
    cache: dict, mode: str, alpha: float, e0_temperature: float, e1_temperature: float
) -> dict:
    e0 = torch.stack([
        aggregate_checkpoint_views(condition_logits, mode)
        for condition_logits in cache["raw_view_logits_e0"]
    ])
    e1 = torch.stack([
        aggregate_checkpoint_views(condition_logits, mode)
        for condition_logits in cache["raw_view_logits_e1"]
    ])
    blended = blend_calibrated_logits(e0, e0_temperature, e1, e1_temperature, alpha)
    labels = cache["labels"]
    calibration_labels = labels.repeat(len(cache["conditions"]))
    threshold = select_threshold(calibration_labels, torch.sigmoid(blended.flatten()), "balanced")
    probabilities = torch.sigmoid(blended)
    per_condition = {
        condition: classification_metrics(labels, probabilities[index], threshold)
        for index, condition in enumerate(cache["conditions"])
    }
    transformed = [condition for condition in cache["conditions"] if condition != "clean"]
    baccs = [per_condition[name]["balanced_accuracy"] for name in transformed]
    aucs = [per_condition[name]["roc_auc"] for name in transformed]
    fprs = [per_condition[name]["false_positive_rate"] for name in transformed]
    worst_index = int(np.argmin(baccs))
    clean = per_condition["clean"]
    return {
        "mode": mode, "alpha": alpha, "ensemble_threshold": threshold,
        "selection_split": "validation", "test_rows_used_for_selection": False,
        "clean_validation_balanced_accuracy": clean["balanced_accuracy"],
        "mean_transformed_validation_balanced_accuracy": float(np.mean(baccs)),
        "worst_transformed_validation_balanced_accuracy": float(baccs[worst_index]),
        "worst_condition": transformed[worst_index],
        "clean_false_positive_rate": clean["false_positive_rate"],
        "mean_transformed_false_positive_rate": float(np.mean(fprs)),
        "mean_transformed_roc_auc": float(np.mean(aucs)),
        "worst_transformed_roc_auc": float(np.min(aucs)),
        "per_condition": per_condition,
        "external_roc_auc": None, "external_balanced_accuracy": None,
        "external_precision": None, "external_recall": None, "external_false_positive_rate": None,
    }


def rank_results(rows: list[dict], split: str = "validation") -> list[dict]:
    require_validation_selection(split)
    ranks = {}
    constraints = {}
    for mode in ENSEMBLE_MODES:
        family = [row for row in rows if row["mode"] == mode]
        if not family:
            continue
        e0 = next(row for row in family if float(row["alpha"]) == 1.0)
        floor = e0["clean_validation_balanced_accuracy"] - 0.01
        eligible = [row for row in family if row["clean_validation_balanced_accuracy"] >= floor]
        eligible.sort(key=lambda row: (
            row["worst_transformed_validation_balanced_accuracy"],
            row["mean_transformed_validation_balanced_accuracy"],
        ), reverse=True)
        ranks.update({(mode, float(row["alpha"])): rank for rank, row in enumerate(eligible, 1)})
        constraints[mode] = floor
    return [{
        **row,
        "clean_constraint_pass": row["clean_validation_balanced_accuracy"] >= constraints[row["mode"]],
        "rank": ranks.get((row["mode"], float(row["alpha"]))),
    } for row in rows]


def refined_alphas(rows: list[dict], step: float) -> dict[str, list[float]]:
    if step <= 0 or step > 0.1:
        raise ValueError("refinement step must be in (0, 0.1]")
    ranked = rank_results(rows)
    result = {}
    for mode in ENSEMBLE_MODES:
        best = min((row for row in ranked if row["mode"] == mode and row["rank"] is not None), key=lambda row: row["rank"])
        center = float(best["alpha"])
        values = set(COARSE_ALPHAS)
        count = round(0.1 / step)
        values.update(round(center + offset * step, 3) for offset in range(-count, count + 1)
                      if 0 <= center + offset * step <= 1)
        result[mode] = sorted(values)
    return result


def write_outputs(rows: list[dict], output_dir: Path, e0: Path, e1: Path, seed: int) -> dict:
    ranked = rank_results(rows)
    eligible = [row for row in ranked if row["rank"] is not None]
    eligible.sort(key=lambda row: (
        row["worst_transformed_validation_balanced_accuracy"],
        row["mean_transformed_validation_balanced_accuracy"],
    ), reverse=True)
    winner = eligible[0]
    document = {
        "selection_split": "validation", "test_rows_used_for_selection": False,
        "external_metrics_used_for_ranking": False,
        "family_ranking": "Each raw/mild3 family is ranked separately under its E0 clean constraint.",
        "results": ranked,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ensemble_summary.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    scalar_keys = [key for key, value in ranked[0].items() if not isinstance(value, dict)]
    with (output_dir / "ensemble_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in scalar_keys} for row in ranked)
    locked = {
        "schema_version": 1, "selection_split": "validation", "test_rows_used_for_selection": False,
        "mode": winner["mode"], "alpha": winner["alpha"],
        "ensemble_threshold": winner["ensemble_threshold"],
        "e0_checkpoint": str(e0.resolve()), "e1_checkpoint": str(e1.resolve()), "seed": seed,
    }
    (output_dir / "winning_ensemble.json").write_text(json.dumps(locked, indent=2) + "\n", encoding="utf-8")
    return locked


def load_locked(path: Path) -> dict:
    locked = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "selection_split", "test_rows_used_for_selection", "mode", "alpha",
                "ensemble_threshold", "e0_checkpoint", "e1_checkpoint", "seed"}
    if set(locked) != required:
        raise ValueError("Locked ensemble must contain exactly one validation-selected configuration")
    if locked["selection_split"] != "validation" or locked["test_rows_used_for_selection"] is not False:
        raise ValueError("Ensemble was not locked on validation only")
    if locked["mode"] not in ENSEMBLE_MODES:
        raise ValueError("Locked ensemble mode must be raw or existing mild3")
    return locked


def _split_rows(data_dir: Path, split: str, seed: int) -> list[tuple[Path, int]]:
    from ..data import load_labeled_paths, stratified_train_val_test_split
    rows = load_labeled_paths(data_dir)
    if split == "all":
        return rows
    _, validation, test = stratified_train_val_test_split(rows, data_dir, 0.15, 0.15, seed)
    return validation if split == "validation" else test


def _models(e0_path: Path, e1_path: Path, device: torch.device):
    e0_head, e0_config, e0_temperature, _ = load_checkpoint(e0_path, device)
    e1_head, e1_config, e1_temperature, _ = load_checkpoint(e1_path, device)
    fields = ("clip_model", "clip_dim", "forensic_dim", "forensic_mode", "quality_dim")
    if tuple(getattr(e0_config, key) for key in fields) != tuple(getattr(e1_config, key) for key in fields):
        raise ValueError("E0 and E1 encoder configurations differ")
    return e0_head, e1_head, e0_config, e1_config, e0_temperature, e1_temperature


def search_command(args: argparse.Namespace) -> None:
    require_validation_selection("validation")
    device = choose_device(args.device)
    e0_head, e1_head, e0_config, e1_config, e0_temperature, e1_temperature = _models(
        args.e0_checkpoint, args.e1_checkpoint, device
    )
    rows = _split_rows(args.data_dir, "validation", args.seed)
    manifest = {
        "schema_version": 1, "split": "validation", "test_rows_used": False,
        "dataset_fingerprint": _fingerprint(rows, args.data_dir),
        "conditions": list(ROBUSTNESS_CONDITIONS), "views": list(ENSEMBLE_MODES["mild3"]),
        "seed": args.seed,
        "checkpoints": [{"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                        for path in (args.e0_checkpoint, args.e1_checkpoint)],
    }
    def builder():
        encoders = FrozenEncoders(e0_config, device)
        payload = extract_shared_logits(
            rows, ROBUSTNESS_CONDITIONS, encoders,
            [(e0_head, e0_config), (e1_head, e1_config)], args.batch_size, args.seed, device,
        )
        del encoders
        return payload
    cache = load_or_build_logit_cache(args.output_dir / "validation_logits.pt", manifest, builder)
    coarse = [score_alpha(cache, mode, alpha, e0_temperature, e1_temperature)
              for mode in ENSEMBLE_MODES for alpha in COARSE_ALPHAS]
    grids = refined_alphas(coarse, args.refine_step)
    rows_scored = [score_alpha(cache, mode, alpha, e0_temperature, e1_temperature)
                   for mode in ENSEMBLE_MODES for alpha in grids[mode]]
    write_outputs(rows_scored, args.output_dir, args.e0_checkpoint, args.e1_checkpoint, args.seed)


def evaluate_locked(args: argparse.Namespace, split: str) -> dict:
    locked = load_locked(args.locked_ensemble)
    device = choose_device(args.device)
    e0_path, e1_path = Path(locked["e0_checkpoint"]), Path(locked["e1_checkpoint"])
    e0_head, e1_head, e0_config, e1_config, t0, t1 = _models(e0_path, e1_path, device)
    rows = _split_rows(args.data_dir, split, int(locked["seed"]))
    conditions = ("clean",) if split == "all" else ROBUSTNESS_CONDITIONS
    encoders = FrozenEncoders(e0_config, device)
    cache = extract_shared_logits(
        rows, conditions, encoders, [(e0_head, e0_config), (e1_head, e1_config)],
        args.batch_size, int(locked["seed"]), device,
    )
    del encoders
    mode, alpha = locked["mode"], float(locked["alpha"])
    e0 = torch.stack([aggregate_checkpoint_views(value, mode) for value in cache["raw_view_logits_e0"]])
    e1 = torch.stack([aggregate_checkpoint_views(value, mode) for value in cache["raw_view_logits_e1"]])
    probabilities = torch.sigmoid(blend_calibrated_logits(e0, t0, e1, t1, alpha))
    per_condition = {
        condition: classification_metrics(cache["labels"], probabilities[index], float(locked["ensemble_threshold"]))
        for index, condition in enumerate(conditions)
    }
    return {"evaluation_split": "external" if split == "all" else "test",
            "alpha_search_performed": False, "locked_ensemble": locked, "per_condition": per_condition}


def locked_test_command(args: argparse.Namespace) -> None:
    output = evaluate_locked(args, "test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "locked_test_metrics.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def external_command(args: argparse.Namespace) -> None:
    output = evaluate_locked(args, "all")
    metrics = output["per_condition"]["clean"]
    output["external_metrics"] = {
        "roc_auc": metrics["roc_auc"], "balanced_accuracy": metrics["balanced_accuracy"],
        "precision": metrics["precision"], "recall": metrics["recall"],
        "false_positive_rate": metrics["false_positive_rate"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "external_metrics.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="E1c validation-only calibrated E0/E1 ensemble")
    subparsers = parser.add_subparsers(dest="command", required=True)
    search = subparsers.add_parser("search")
    search.add_argument("--data-dir", type=Path, required=True)
    search.add_argument("--e0-checkpoint", type=Path, required=True)
    search.add_argument("--e1-checkpoint", type=Path, required=True)
    search.add_argument("--output-dir", type=Path, required=True)
    search.add_argument("--refine-step", type=float, default=0.025)
    search.add_argument("--batch-size", type=int, default=8)
    search.add_argument("--seed", type=int, default=42)
    search.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    search.set_defaults(handler=search_command)
    for name, handler in (("external", external_command), ("locked-test", locked_test_command)):
        command = subparsers.add_parser(name)
        command.add_argument("--data-dir", type=Path, required=True)
        command.add_argument("--locked-ensemble", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
        command.set_defaults(handler=handler)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
