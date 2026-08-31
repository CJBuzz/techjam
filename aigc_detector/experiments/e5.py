from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ..data import DeterministicTransform, ROBUSTNESS_CONDITIONS, test_time_views
from ..metrics import select_threshold
from ..model import FrozenEncoders, image_quality_statistics, load_checkpoint
from ..train import choose_device


QUALITY_FEATURE_NAMES = (
    "laplacian_energy", "fft_high_frequency_ratio", "spectral_slope",
    "block_boundary_energy", "contrast_span", "clipping_fraction",
    "tta_calibrated_logit_std",
)
CALIBRATION_MODES = ("global", "binned", "continuous")
PROHIBITED_FEATURE_TOKENS = ("condition", "label", "target", "source", "generator", "path", "filename")


def validate_quality_schema() -> None:
    if any(token in name for name in QUALITY_FEATURE_NAMES for token in PROHIBITED_FEATURE_TOKENS):
        raise ValueError("E5 quality schema contains prohibited information")


def require_validation_fit(split: str) -> None:
    if split != "validation":
        raise ValueError("E5 calibration fitting is validation-only; final-test fitting is forbidden")


def quality_vector(image: Image.Image, calibrated_view_logits: torch.Tensor) -> torch.Tensor:
    """Safe deterministic image statistics plus prediction dispersion; no metadata inputs."""
    validate_quality_schema()
    statistics = image_quality_statistics([image.convert("RGB")])[0]
    dispersion = calibrated_view_logits.float().std(unbiased=False).view(1)
    return torch.cat((statistics, dispersion))


def global_config(threshold: float) -> dict:
    return {"mode": "global", "parameter_count": 0, "threshold": float(threshold),
            "threshold_definition": "saved global checkpoint threshold"}


def fit_binned_thresholds(
    probabilities: torch.Tensor, quality: torch.Tensor, labels: torch.Tensor,
    base_threshold: float, bins: int = 4, split: str = "validation",
) -> dict:
    require_validation_fit(split)
    if bins < 2 or bins > 5:
        raise ValueError("Use 2-5 quality bins")
    mean, std = quality.mean(0), quality.std(0, unbiased=False).clamp_min(1e-6)
    scores = ((quality - mean) / std).mean(1)
    boundaries = torch.quantile(scores, torch.linspace(0, 1, bins + 1)[1:-1]).unique(sorted=True)
    bin_ids = torch.bucketize(scores, boundaries)
    thresholds = []
    for index in range(len(boundaries) + 1):
        mask = bin_ids == index
        thresholds.append(
            select_threshold(labels[mask], probabilities[mask], "balanced")
            if int(mask.sum()) >= 2 and len(labels[mask].unique()) == 2 else float(base_threshold)
        )
    return {
        "mode": "binned", "parameter_count": len(thresholds),
        "threshold_definition": "validation-quantile quality bins with one validation-selected threshold per bin",
        "base_threshold": float(base_threshold), "quality_mean": mean.tolist(),
        "quality_std": std.tolist(), "boundaries": boundaries.tolist(), "thresholds": thresholds,
        "fit_split": "validation",
    }


def fit_continuous_threshold(
    probabilities: torch.Tensor, quality: torch.Tensor, labels: torch.Tensor,
    base_threshold: float, max_delta: float = 0.10, l2: float = 0.25,
    split: str = "validation", seed: int = 42,
) -> dict:
    require_validation_fit(split)
    if not 0 < max_delta <= 0.2:
        raise ValueError("max_delta must be in (0, 0.2]")
    torch.manual_seed(seed)
    mean, std = quality.mean(0), quality.std(0, unbiased=False).clamp_min(1e-6)
    standardized = (quality - mean) / std
    beta = torch.zeros(quality.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([beta, bias], lr=0.03)
    for _ in range(250):
        optimizer.zero_grad(set_to_none=True)
        delta = max_delta * torch.tanh(standardized @ beta + bias)
        thresholds = (base_threshold + delta).clamp(0.01, 0.99)
        decision_logits = 10.0 * (probabilities - thresholds)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(decision_logits, labels)
        loss = loss + l2 * beta.square().mean() + l2 * bias.square().mean()
        loss.backward()
        optimizer.step()
    return {
        "mode": "continuous", "parameter_count": quality.shape[1] + 1,
        "threshold_definition": "t(x)=clamp(t0+max_delta*tanh(beta*q_standardized+bias),0.01,0.99)",
        "base_threshold": float(base_threshold), "max_delta": float(max_delta),
        "quality_mean": mean.tolist(), "quality_std": std.tolist(),
        "beta": beta.detach().tolist(), "bias": float(bias.detach()),
        "l2": float(l2), "fit_split": "validation", "seed": seed,
    }


def thresholds_for(quality: torch.Tensor, config: dict) -> torch.Tensor:
    mode = config["mode"]
    if mode == "global":
        return torch.full((len(quality),), float(config["threshold"]))
    mean = torch.tensor(config["quality_mean"], dtype=quality.dtype)
    std = torch.tensor(config["quality_std"], dtype=quality.dtype)
    standardized = (quality - mean) / std
    if mode == "binned":
        scores = standardized.mean(1)
        bins = torch.bucketize(scores, torch.tensor(config["boundaries"], dtype=quality.dtype))
        values = torch.tensor(config["thresholds"], dtype=quality.dtype)
        return values[bins]
    if mode == "continuous":
        beta = torch.tensor(config["beta"], dtype=quality.dtype)
        delta = float(config["max_delta"]) * torch.tanh(standardized @ beta + float(config["bias"]))
        return (float(config["base_threshold"]) + delta).clamp(0.01, 0.99)
    raise ValueError(f"Unknown calibration mode: {mode}")


def adaptive_metrics(labels: torch.Tensor, probabilities: torch.Tensor, thresholds: torch.Tensor) -> dict:
    y = labels.numpy().astype(int)
    p = probabilities.numpy()
    predictions = (probabilities >= thresholds).numpy().astype(int)
    tn = int(((y == 0) & (predictions == 0)).sum())
    fp = int(((y == 0) & (predictions == 1)).sum())
    fn = int(((y == 1) & (predictions == 0)).sum())
    tp = int(((y == 1) & (predictions == 1)).sum())
    specificity = tn / max(1, tn + fp)
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    return {
        "sample_count": len(y), "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (specificity + recall) / 2,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(1e-15, precision + recall),
        "specificity": specificity, "false_positive_rate": fp / max(1, tn + fp),
        "false_negative_rate": fn / max(1, tp + fn),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": "quality_conditioned", "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def score_config(observations: dict, config: dict) -> dict:
    per_condition = {}
    for index, condition in enumerate(observations["conditions"]):
        quality = observations["quality"][index]
        probabilities = observations["probabilities"][index]
        per_condition[condition] = adaptive_metrics(
            observations["labels"], probabilities, thresholds_for(quality, config)
        )
    transformed = list(observations["conditions"])[1:]
    baccs = [per_condition[name]["balanced_accuracy"] for name in transformed]
    aucs = [per_condition[name]["roc_auc"] for name in transformed]
    fprs = [per_condition[name]["false_positive_rate"] for name in transformed]
    fnrs = [per_condition[name]["false_negative_rate"] for name in transformed]
    worst_index = int(np.argmin(baccs))
    clean = per_condition["clean"]
    return {
        "mode": config["mode"], "parameter_count": config["parameter_count"],
        "threshold_definition": config["threshold_definition"],
        "selection_split": "validation", "test_rows_used_for_selection": False,
        "clean_validation_balanced_accuracy": clean["balanced_accuracy"],
        "mean_transformed_validation_balanced_accuracy": float(np.mean(baccs)),
        "worst_transformed_validation_balanced_accuracy": float(baccs[worst_index]),
        "worst_condition": transformed[worst_index],
        "clean_false_positive_rate": clean["false_positive_rate"],
        "clean_false_negative_rate": clean["false_negative_rate"],
        "mean_transformed_false_positive_rate": float(np.mean(fprs)),
        "mean_transformed_false_negative_rate": float(np.mean(fnrs)),
        "per_condition_balanced_accuracy": {name: value["balanced_accuracy"] for name, value in per_condition.items()},
        "mean_transformed_roc_auc": float(np.mean(aucs)), "worst_transformed_roc_auc": float(np.min(aucs)),
    }


def rank_results(rows: list[dict], split: str = "validation") -> list[dict]:
    require_validation_fit(split)
    baseline = next(row for row in rows if row["mode"] == "global")
    floor = baseline["clean_validation_balanced_accuracy"] - 0.01
    complexity = {"global": 0, "binned": 1, "continuous": 2}
    eligible = [row for row in rows if row["clean_validation_balanced_accuracy"] >= floor]
    eligible.sort(key=lambda row: (
        row["worst_transformed_validation_balanced_accuracy"],
        row["mean_transformed_validation_balanced_accuracy"],
        -complexity[row["mode"]],
    ), reverse=True)
    ranks = {row["mode"]: rank for rank, row in enumerate(eligible, 1)}
    return [{**row, "clean_constraint_pass": row["clean_validation_balanced_accuracy"] >= floor,
             "rank": ranks.get(row["mode"])} for row in rows]


def load_or_build_observations(path: Path, manifest: dict, builder: Callable[[], dict]) -> dict:
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("manifest") != manifest:
            raise ValueError("Existing E5 observation cache is incompatible")
        return payload
    payload = builder()
    payload["manifest"] = manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def extract_observations(
    rows: list[tuple[Path, int]], conditions: tuple[str, ...], mode: str,
    encoders: FrozenEncoders, head: torch.nn.Module, config: object,
    temperature: float, batch_size: int, seed: int, device: torch.device,
) -> dict:
    tta_mode = "none" if mode == "raw" else "mild3"
    probability_conditions, quality_conditions = [], []
    labels = torch.tensor([label for _, label in rows], dtype=torch.float32)
    for condition_index, condition in enumerate(conditions):
        probabilities, qualities = [], []
        for start in range(0, len(rows), batch_size):
            images, conditioned_images = [], []
            batch_rows = rows[start : start + batch_size]
            for path, _ in batch_rows:
                transform = DeterministicTransform(condition, seed, str(path), condition_index)
                with Image.open(path) as source:
                    conditioned = transform(source.convert("RGB"))
                conditioned_images.append(conditioned)
                images.extend(test_time_views(conditioned, tta_mode, seed, f"{path}:{condition}"))
            features = encoders(images)
            model_features = features if config.quality_dim else features[
                :, : config.clip_dim + config.forensic_dim
            ]
            view_count = 1 if mode == "raw" else 3
            with torch.inference_mode():
                view_logits = head(model_features.to(device)).view(len(batch_rows), view_count).cpu() / temperature
            probabilities.extend(torch.sigmoid(view_logits.mean(1)).tolist())
            qualities.extend(quality_vector(image, logits) for image, logits in zip(conditioned_images, view_logits, strict=True))
        probability_conditions.append(torch.tensor(probabilities))
        quality_conditions.append(torch.stack(qualities))
    return {"probabilities": torch.stack(probability_conditions), "quality": torch.stack(quality_conditions),
            "labels": labels, "conditions": list(conditions), "base_mode": mode}


def _rows(data_dir: Path, split: str, seed: int):
    from ..data import load_labeled_paths, stratified_train_val_test_split
    rows = load_labeled_paths(data_dir)
    _, validation, test = stratified_train_val_test_split(rows, data_dir, 0.15, 0.15, seed)
    return validation if split == "validation" else test


def _fingerprint(rows, root):
    digest = hashlib.sha256()
    for path, label in sorted(rows, key=lambda row: str(row[0])):
        stat = path.stat()
        digest.update(f"{path.resolve().relative_to(root.resolve())}\0{label}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def write_outputs(results: list[dict], configs: dict, observations: dict, output_dir: Path,
                  checkpoint: Path, base_mode: str, seed: int) -> None:
    ranked = rank_results(results)
    winner = min((row for row in ranked if row["rank"] is not None), key=lambda row: row["rank"])
    output_dir.mkdir(parents=True, exist_ok=True)
    document = {"selection_split": "validation", "test_rows_used_for_selection": False,
                "base_mode": base_mode, "quality_features": list(QUALITY_FEATURE_NAMES), "results": ranked}
    (output_dir / "calibration_summary.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    scalar_keys = [key for key, value in ranked[0].items() if not isinstance(value, dict)]
    with (output_dir / "calibration_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys + ["per_condition_balanced_accuracy"])
        writer.writeheader()
        for row in ranked:
            writer.writerow({key: row.get(key) for key in scalar_keys} | {
                "per_condition_balanced_accuracy": json.dumps(row["per_condition_balanced_accuracy"], sort_keys=True)
            })
    (output_dir / "quality_bins.json").write_text(json.dumps(configs["binned"], indent=2) + "\n", encoding="utf-8")
    locked = {"schema_version": 1, "selection_split": "validation", "test_rows_used_for_selection": False,
              "base_checkpoint": str(checkpoint.resolve()), "base_mode": base_mode,
              "quality_features": list(QUALITY_FEATURE_NAMES), "calibration": configs[winner["mode"]], "seed": seed}
    (output_dir / "winning_calibration.json").write_text(json.dumps(locked, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "quality_by_condition.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["condition"] + [f"{name}_{stat}" for name in QUALITY_FEATURE_NAMES for stat in ("mean", "std")]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, condition in enumerate(observations["conditions"]):
            row = {"condition": condition}
            for feature_index, name in enumerate(QUALITY_FEATURE_NAMES):
                values = observations["quality"][index, :, feature_index]
                row[f"{name}_mean"], row[f"{name}_std"] = float(values.mean()), float(values.std(unbiased=False))
            writer.writerow(row)


def validation_command(args: argparse.Namespace) -> None:
    require_validation_fit("validation")
    device = choose_device(args.device)
    head, config, temperature, metadata = load_checkpoint(args.checkpoint, device)
    rows = _rows(args.data_dir, "validation", args.seed)
    manifest = {"schema_version": 1, "split": "validation", "test_rows_used": False,
                "dataset_fingerprint": _fingerprint(rows, args.data_dir),
                "checkpoint": str(args.checkpoint.resolve()), "checkpoint_size": args.checkpoint.stat().st_size,
                "checkpoint_mtime_ns": args.checkpoint.stat().st_mtime_ns, "base_mode": args.base_mode,
                "conditions": list(ROBUSTNESS_CONDITIONS), "quality_features": list(QUALITY_FEATURE_NAMES), "seed": args.seed}
    def builder():
        encoders = FrozenEncoders(config, device)
        payload = extract_observations(rows, ROBUSTNESS_CONDITIONS, args.base_mode, encoders, head, config,
                                       temperature, args.batch_size, args.seed, device)
        del encoders
        return payload
    observations = load_or_build_observations(args.output_dir / "validation_observations.pt", manifest, builder)
    probabilities = observations["probabilities"].flatten()
    quality = observations["quality"].flatten(0, 1)
    labels = observations["labels"].repeat(len(observations["conditions"]))
    base_threshold = float(metadata.get("threshold", 0.5))
    configs = {
        "global": global_config(base_threshold),
        "binned": fit_binned_thresholds(probabilities, quality, labels, base_threshold, args.bins),
        "continuous": fit_continuous_threshold(probabilities, quality, labels, base_threshold,
                                                args.max_delta, args.l2, seed=args.seed),
    }
    results = [score_config(observations, configs[mode]) for mode in CALIBRATION_MODES]
    write_outputs(results, configs, observations, args.output_dir, args.checkpoint, args.base_mode, args.seed)


def load_locked(path: Path) -> dict:
    locked = json.loads(path.read_text(encoding="utf-8"))
    if locked.get("selection_split") != "validation" or locked.get("test_rows_used_for_selection") is not False:
        raise ValueError("Calibration was not locked using validation only")
    if locked.get("quality_features") != list(QUALITY_FEATURE_NAMES):
        raise ValueError("Locked quality schema is incompatible")
    return locked


def locked_test_command(args: argparse.Namespace) -> None:
    locked = load_locked(args.locked_calibration)
    device = choose_device(args.device)
    checkpoint = Path(locked["base_checkpoint"])
    head, config, temperature, _ = load_checkpoint(checkpoint, device)
    rows = _rows(args.data_dir, "test", int(locked["seed"]))
    encoders = FrozenEncoders(config, device)
    observations = extract_observations(rows, ROBUSTNESS_CONDITIONS, locked["base_mode"], encoders, head, config,
                                        temperature, args.batch_size, int(locked["seed"]), device)
    del encoders
    result = score_config(observations, locked["calibration"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "locked_test_metrics.json").write_text(json.dumps({
        "evaluation_split": "test", "calibration_fitting_performed": False,
        "locked_calibration": locked, "result": result,
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="E5 validation-only quality-conditioned calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validation = subparsers.add_parser("validation")
    validation.add_argument("--data-dir", type=Path, required=True)
    validation.add_argument("--checkpoint", type=Path, required=True)
    validation.add_argument("--output-dir", type=Path, required=True)
    validation.add_argument("--base-mode", choices=("raw", "mild3"), default="raw")
    validation.add_argument("--bins", type=int, default=4)
    validation.add_argument("--max-delta", type=float, default=0.10)
    validation.add_argument("--l2", type=float, default=0.25)
    validation.add_argument("--batch-size", type=int, default=8)
    validation.add_argument("--seed", type=int, default=42)
    validation.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    validation.set_defaults(handler=validation_command)
    test = subparsers.add_parser("locked-test")
    test.add_argument("--data-dir", type=Path, required=True)
    test.add_argument("--locked-calibration", type=Path, required=True)
    test.add_argument("--output-dir", type=Path, required=True)
    test.add_argument("--batch-size", type=int, default=8)
    test.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    test.set_defaults(handler=locked_test_command)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
