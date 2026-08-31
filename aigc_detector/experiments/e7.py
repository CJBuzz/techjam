from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..data import (
    BALANCED_TRANSFORM_GROUPS,
    ROBUSTNESS_CONDITIONS,
    DeterministicTransform,
    RobustTransform,
    load_labeled_paths,
    stratified_train_val_test_split,
)
from .e4a import FEATURE_BLOCKS, assemble_validation_matrix, prepare_missing_validation_features
from ..metrics import classification_metrics, fit_temperature, select_threshold
from ..model import FrozenEncoders, ModelConfig, load_checkpoint
from ..train import choose_device


SUPPORTED_BINS = (16, 32, 64)
MODEL_MODES = ("radial_only", "fused_radial", "clip_radial")
STABILITY_SCALES = (1.0, 0.75, 0.50, 0.25)


def radial_bin_indices(height: int, width: int, bins: int) -> np.ndarray:
    if min(height, width, bins) < 2:
        raise ValueError("Radial descriptor requires dimensions and bins >= 2")
    yy, xx = np.indices((height, width), dtype=np.float64)
    cy, cx = height // 2, width // 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius /= radius.max() + 1e-12
    return np.minimum((radius * bins).astype(np.int64), bins - 1)


def radial_fft_descriptor(image: Image.Image, bins: int = 32) -> torch.Tensor:
    """L1-normalized log radial power after mean removal and a 2D Hann window."""
    if bins not in SUPPORTED_BINS:
        raise ValueError(f"Supported radial bin counts: {SUPPORTED_BINS}")
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    luminance = rgb.mean(axis=2)
    centered = luminance - luminance.mean()
    window = np.outer(np.hanning(luminance.shape[0]), np.hanning(luminance.shape[1]))
    spectrum = np.fft.fftshift(np.fft.fft2(centered * window))
    power = np.abs(spectrum) ** 2
    power[luminance.shape[0] // 2, luminance.shape[1] // 2] = 0.0
    indices = radial_bin_indices(*luminance.shape, bins)
    energy = np.bincount(indices.ravel(), weights=power.ravel(), minlength=bins)[:bins]
    compressed = np.log1p(energy)
    total = compressed.sum()
    descriptor = compressed / total if total > 1e-12 else np.zeros(bins, dtype=np.float64)
    return torch.tensor(descriptor, dtype=torch.float32)


class DescriptorDataset(Dataset):
    def __init__(self, tasks: list[tuple[Path, str, int]], bins: tuple[int, ...], seed: int) -> None:
        self.tasks, self.bins, self.seed = tasks, bins, seed

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: int):
        path, transform_name, repeat = self.tasks[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if transform_name.startswith("scale:"):
            scale = float(transform_name.split(":", 1)[1])
            image = image if scale == 1 else RobustTransform._apply_one(image, "resize", scale)
        elif transform_name != "clean":
            image = DeterministicTransform(transform_name, self.seed, str(path), repeat)(image)
        return {bins: radial_fft_descriptor(image, bins) for bins in self.bins}


def _descriptor_collate(batch):
    # NumPy payloads are copied through the worker queue. Returning Torch
    # storage here invokes multiprocessing's local resource-sharing socket,
    # which is unavailable on some WSL/container filesystems.
    return {bins: np.stack([row[bins].numpy() for row in batch]) for bins in batch[0]}


def extract_descriptor_tasks(
    tasks: list[tuple[Path, str, int]], bins: tuple[int, ...], seed: int,
    batch_size: int, workers: int,
) -> dict[int, torch.Tensor]:
    loader = DataLoader(
        DescriptorDataset(tasks, bins, seed), batch_size=batch_size, shuffle=False,
        num_workers=workers, collate_fn=_descriptor_collate,
    )
    outputs = {count: [] for count in bins}
    for batch in loader:
        for count in bins:
            outputs[count].append(torch.from_numpy(batch[count]))
    return {count: torch.cat(values) for count, values in outputs.items()}


def descriptor_manifest(base: dict, bins: int, train_rows: int, val_rows: int, stability_rows: int) -> dict:
    return {
        "schema_version": 1, "bins": bins, "base_cache_manifest": base["manifest"],
        "train_originals": train_rows, "validation_originals": val_rows,
        "stability_originals": stability_rows, "train_repeats": base["manifest"]["augmentation_repeats"],
        "validation_conditions": list(ROBUSTNESS_CONDITIONS),
        "stability_scales": list(STABILITY_SCALES), "raw_images_persisted": False,
        "seed": base["manifest"]["seed"],
        "definition": "mean RGB luminance; mean removal; 2D Hann; centered FFT power; DC excluded; 32/16/64 normalized radial bins",
    }


def descriptor_cache_valid(path: Path, manifest: dict) -> bool:
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("manifest") != manifest or not all(
        key in payload for key in ("train", "validation", "stability")
    ):
        return False
    bins = manifest["bins"]
    expected = {
        "train": (manifest["train_originals"] * manifest["train_repeats"], bins),
        "validation": (manifest["validation_originals"] * len(manifest["validation_conditions"]), bins),
        "stability": (manifest["stability_originals"] * len(manifest["stability_scales"]), bins),
    }
    return all(tuple(payload[key].shape) == shape for key, shape in expected.items())


def cache_command(args: argparse.Namespace) -> None:
    requested = tuple(args.bins)
    if any(count not in SUPPORTED_BINS for count in requested):
        raise ValueError(f"Supported bins are {SUPPORTED_BINS}")
    base = torch.load(args.base_cache, map_location="cpu", weights_only=True, mmap=True)
    seed = int(base["manifest"]["seed"])
    if args.seed != seed:
        raise ValueError(f"E7 cache seed {args.seed} must match base-cache seed {seed}")
    rows = load_labeled_paths(args.data_dir)
    train_rows, val_rows, _ = stratified_train_val_test_split(
        rows, args.data_dir, base["manifest"]["validation_fraction"],
        base["manifest"]["test_fraction"], base["manifest"]["seed"],
    )
    repeats = base["manifest"]["augmentation_repeats"]
    train_tasks = []
    for repeat in range(repeats):
        for index, (path, _) in enumerate(train_rows):
            group = "clean" if repeat == 0 else BALANCED_TRANSFORM_GROUPS[
                (index * (repeats - 1) + repeat - 1) % len(BALANCED_TRANSFORM_GROUPS)
            ]
            train_tasks.append((path, group, repeat))
    validation_tasks = [
        (path, condition, condition_index)
        for condition_index, condition in enumerate(ROBUSTNESS_CONDITIONS)
        for path, _ in val_rows
    ]
    stability_rows = train_rows[: min(args.stability_images, len(train_rows))]
    stability_tasks = [
        (path, f"scale:{scale}", scale_index)
        for scale_index, scale in enumerate(STABILITY_SCALES)
        for path, _ in stability_rows
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for count in requested:
        manifest = descriptor_manifest(base, count, len(train_rows), len(val_rows), len(stability_rows))
        path = args.output_dir / f"radial_{count}.pt"
        if descriptor_cache_valid(path, manifest):
            print(f"Reusing radial descriptor cache: {path}")
        elif path.exists():
            raise ValueError(f"Incompatible radial cache; refusing overwrite: {path}")
        else:
            missing.append(count)
    if missing:
        counts = tuple(missing)
        train = extract_descriptor_tasks(train_tasks, counts, seed, args.batch_size, args.workers)
        validation = extract_descriptor_tasks(validation_tasks, counts, seed, args.batch_size, args.workers)
        stability = extract_descriptor_tasks(stability_tasks, counts, seed, args.batch_size, args.workers)
        for count in counts:
            manifest = descriptor_manifest(base, count, len(train_rows), len(val_rows), len(stability_rows))
            path = args.output_dir / f"radial_{count}.pt"
            temporary = path.with_suffix(".pt.tmp")
            torch.save({
                "train": train[count], "validation": validation[count], "stability": stability[count],
                "train_labels": base["train_labels"], "validation_labels": base["val_labels"],
                "manifest": manifest,
            }, temporary)
            os.replace(temporary, path)
    (args.output_dir.parent / "feature_config.json").write_text(json.dumps({
        "supported_bins": list(SUPPORTED_BINS), "default_bins": 32,
        "luminance": "mean RGB", "mean_subtraction": True, "hann_window": True,
        "spectrum": "centered FFT power", "dc_handling": "excluded",
        "aggregation": "concentric resolution-normalized radial energy bins",
        "normalization": "log1p bin energy then L1 normalization",
        "num_workers": args.workers,
    }, indent=2) + "\n", encoding="utf-8")


def select_features(fused: torch.Tensor, radial: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "radial_only":
        return radial
    if mode == "clip_radial":
        return torch.cat((fused[:, FEATURE_BLOCKS["clip"]], radial), dim=1)
    if mode == "fused_radial":
        return torch.cat((fused, radial), dim=1)
    raise ValueError(f"Unknown E7 model mode: {mode}")


def cosine_stability(reference: torch.Tensor, transformed: torch.Tensor) -> tuple[float, float, float, float]:
    similarity = F.cosine_similarity(reference, transformed, dim=1, eps=1e-8)
    distance = (reference - transformed).norm(dim=1)
    return float(similarity.mean()), float(similarity.std(unbiased=False)), float(distance.mean()), float(distance.std(unbiased=False))


def radial_stability_rows(payloads: dict[int, dict]) -> list[dict]:
    rows = []
    for bins, payload in payloads.items():
        count = payload["manifest"]["stability_originals"]
        scales = payload["stability"].view(len(STABILITY_SCALES), count, bins)
        for index, scale in enumerate(STABILITY_SCALES[1:], 1):
            sim_mean, sim_std, dist_mean, dist_std = cosine_stability(scales[0], scales[index])
            rows.append({"representation": f"radial_{bins}", "scale": scale,
                         "mean_cosine_similarity": sim_mean, "std_cosine_similarity": sim_std,
                         "mean_l2_distance": dist_mean, "std_l2_distance": dist_std})
    return rows


def require_validation_selection(split: str) -> None:
    if split != "validation":
        raise ValueError("E7 configuration selection is validation-only; final-test selection is forbidden")


def train_one(mode, bins, base, radial, robust_x, robust_y, robust_conditions,
              output_dir, device, epochs, patience, batch_size, learning_rate, seed,
              baseline_clean):
    torch.manual_seed(seed)
    train_fused, train_y = base["train_features"], base["train_labels"]
    clean_fused, clean_y = base["val_features"], base["val_labels"]
    val_count = len(clean_y)
    radial_validation = radial["validation"].view(len(ROBUSTNESS_CONDITIONS), val_count, bins)
    train_x = select_features(train_fused, radial["train"], mode)
    clean_x = select_features(clean_fused, radial_validation[0], mode)
    condition_order = list(dict.fromkeys(robust_conditions))
    official_indices = {name: index for index, name in enumerate(ROBUSTNESS_CONDITIONS)}
    robust_radial = torch.cat([radial_validation[official_indices[name]] for name in condition_order])
    robust_selected = select_features(robust_x, robust_radial, mode)
    head = nn.Linear(train_x.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    originals = int(base["train_original_indices"].max()) + 1
    repeats = len(train_y) // originals
    loader = DataLoader(torch.arange(originals), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    groups, group_names = base["train_groups"], sorted(set(base["train_groups"]))
    best_state, best_key, stale = None, None, 0
    for _ in range(epochs):
        head.train()
        for original_batch in loader:
            indices = torch.cat([original_batch + repeat * originals for repeat in range(repeats)])
            features, labels = train_x[indices].to(device), train_y[indices].to(device)
            batch_groups = [groups[index] for index in indices.tolist()]
            optimizer.zero_grad(set_to_none=True)
            logits = head(features).squeeze(1)
            losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            generic = ((logits.view(repeats, -1) - logits.view(repeats, -1).mean(0)) ** 2).mean()
            group_losses = [losses[torch.tensor([value == group for value in batch_groups], device=device)].mean()
                            for group in group_names if group in batch_groups]
            loss = losses.mean() + 0.05 * generic + 0.5 * torch.stack(group_losses).max()
            loss.backward()
            optimizer.step()
        head.cpu().eval()
        with torch.no_grad():
            clean_logits, robust_logits = head(clean_x).squeeze(1), head(robust_selected).squeeze(1)
        threshold = select_threshold(torch.cat((clean_y, robust_y)),
                                     torch.sigmoid(torch.cat((clean_logits, robust_logits))), "balanced")
        clean_metric = classification_metrics(clean_y, torch.sigmoid(clean_logits), threshold)
        baccs = []
        for condition in ROBUSTNESS_CONDITIONS[1:]:
            mask = torch.tensor([value == condition for value in robust_conditions])
            baccs.append(classification_metrics(robust_y[mask], torch.sigmoid(robust_logits[mask]), threshold)["balanced_accuracy"])
        clean_pass = clean_metric["balanced_accuracy"] >= baseline_clean - 0.01
        key = (clean_pass, min(baccs), float(np.mean(baccs)))
        if best_key is None or key > best_key:
            best_key, stale = key, 0
            best_state = {name: value.detach().clone() for name, value in head.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
        head.to(device)
    head.load_state_dict(best_state)
    head.cpu().eval()
    with torch.no_grad():
        clean_logits, robust_logits = head(clean_x).squeeze(1), head(robust_selected).squeeze(1)
    calibration_logits, calibration_labels = torch.cat((clean_logits, robust_logits)), torch.cat((clean_y, robust_y))
    temperature = fit_temperature(calibration_logits, calibration_labels)
    threshold = select_threshold(calibration_labels, torch.sigmoid(calibration_logits / temperature), "balanced")
    clean = classification_metrics(clean_y, torch.sigmoid(clean_logits / temperature), threshold)
    probabilities = torch.sigmoid(robust_logits / temperature)
    per_condition = {}
    for condition in ROBUSTNESS_CONDITIONS[1:]:
        mask = torch.tensor([value == condition for value in robust_conditions])
        per_condition[condition] = classification_metrics(robust_y[mask], probabilities[mask], threshold)
    baccs = [per_condition[name]["balanced_accuracy"] for name in ROBUSTNESS_CONDITIONS[1:]]
    aucs = [per_condition[name]["roc_auc"] for name in ROBUSTNESS_CONDITIONS[1:]]
    fprs = [per_condition[name]["false_positive_rate"] for name in ROBUSTNESS_CONDITIONS[1:]]
    checkpoint = output_dir / "model.pt"
    torch.save({"state_dict": head.state_dict(), "mode": mode, "bins": bins,
                "input_dim": train_x.shape[1], "temperature": temperature, "threshold": threshold,
                "metadata": {"selection_split": "validation", "test_rows_used_for_selection": False}}, checkpoint)
    worst_index = int(np.argmin(baccs))
    return {"model_mode": mode, "radial_bins": bins, "feature_dimension": train_x.shape[1],
            "trainable_parameter_count": sum(p.numel() for p in head.parameters()), "checkpoint": str(checkpoint),
            "status": "succeeded", "selection_split": "validation", "test_rows_used_for_selection": False,
            "clean_validation_balanced_accuracy": clean["balanced_accuracy"],
            "mean_transformed_validation_balanced_accuracy": float(np.mean(baccs)),
            "worst_transformed_validation_balanced_accuracy": float(baccs[worst_index]),
            "resize_x0.5_balanced_accuracy": per_condition["resize_x0.5"]["balanced_accuracy"],
            "resize_x0.25_balanced_accuracy": per_condition["resize_x0.25"]["balanced_accuracy"],
            "worst_condition": ROBUSTNESS_CONDITIONS[1:][worst_index],
            "mean_transformed_false_positive_rate": float(np.mean(fprs)),
            "mean_transformed_roc_auc": float(np.mean(aucs)), "worst_transformed_roc_auc": float(np.min(aucs))}


def rank_results(rows, baseline_clean, split="validation"):
    require_validation_selection(split)
    floor = baseline_clean - 0.01
    eligible = [row for row in rows if row.get("status") == "succeeded" and row["clean_validation_balanced_accuracy"] >= floor]
    eligible.sort(key=lambda row: (row["worst_transformed_validation_balanced_accuracy"],
                                   row["mean_transformed_validation_balanced_accuracy"]), reverse=True)
    ranks = {(row["model_mode"], row["radial_bins"]): rank for rank, row in enumerate(eligible, 1)}
    return [{**row, "clean_constraint_pass": row.get("status") == "succeeded" and row["clean_validation_balanced_accuracy"] >= floor,
             "rank": ranks.get((row["model_mode"], row["radial_bins"]))} for row in rows]


def select_eligible_winner(ranked: list[dict]) -> dict | None:
    eligible = [row for row in ranked if row.get("rank") is not None]
    return min(eligible, key=lambda row: row["rank"]) if eligible else None


def eligibility_summary(ranked: list[dict], baseline_clean: float) -> dict:
    winner = select_eligible_winner(ranked)
    reason = None if winner else (
        "No succeeded E7 candidate satisfied clean validation balanced accuracy >= "
        f"baseline ({baseline_clean:.12g}) - 0.01."
    )
    return {"eligible_winner": winner, "no_eligible_candidate": winner is None, "reason": reason}


def sweep_command(args):
    require_validation_selection("validation")
    base = torch.load(args.base_cache, map_location="cpu", weights_only=True, mmap=True)
    extra = prepare_missing_validation_features(
        base, args.base_cache, args.validation_cache, args.data_dir,
        args.device, args.feature_batch_size,
    )
    if extra.get("manifest", {}).get("source_manifest") != base.get("manifest"):
        raise ValueError("Validation feature cache is incompatible with the local base cache")
    robust_x, robust_y, robust_conditions = assemble_validation_matrix(base, extra)
    baseline_head, _, baseline_temperature, baseline_metadata = load_checkpoint(args.baseline_checkpoint, torch.device("cpu"))
    with torch.no_grad():
        baseline_probabilities = torch.sigmoid(baseline_head(base["val_features"]) / baseline_temperature)
    baseline_clean = classification_metrics(base["val_labels"], baseline_probabilities,
                                            float(baseline_metadata.get("threshold", 0.5)))["balanced_accuracy"]
    payloads = {}
    for bins in args.bins:
        path = args.feature_cache / f"radial_{bins}.pt"
        payloads[bins] = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if payloads[bins].get("manifest", {}).get("base_cache_manifest") != base.get("manifest"):
            raise ValueError(f"Radial cache is incompatible with base cache: {path}")
    device = choose_device(args.device)
    rows = []
    for mode in args.modes:
        for bins in args.bins:
            directory = args.output_dir / "checkpoints" / f"{mode}_bins_{bins}"
            result_path = directory / "result.json"
            if result_path.is_file() and (directory / "model.pt").is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("status") == "succeeded" and result.get("model_mode") == mode and result.get("radial_bins") == bins:
                    rows.append(result); continue
            directory.mkdir(parents=True, exist_ok=True)
            try:
                result = train_one(mode, bins, base, payloads[bins], robust_x, robust_y, robust_conditions,
                                   directory, device, args.epochs, args.patience, args.batch_size,
                                   args.learning_rate, args.seed, baseline_clean)
            except Exception as error:
                result = {"model_mode": mode, "radial_bins": bins, "status": "failed",
                          "failure_reason": f"{type(error).__name__}: {error}"}
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            rows.append(result)
    ranked = rank_results(rows, baseline_clean)
    eligibility = eligibility_summary(ranked, baseline_clean)
    winner = eligibility["eligible_winner"]
    no_eligible_reason = eligibility["reason"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_summary.json").write_text(json.dumps({
        "selection_split": "validation", "baseline_clean_balanced_accuracy": baseline_clean,
        **eligibility,
        "results": ranked,
    }, indent=2) + "\n", encoding="utf-8")
    scalar_keys = sorted({key for row in ranked for key, value in row.items() if not isinstance(value, (dict, list))})
    with (args.output_dir / "validation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys); writer.writeheader(); writer.writerows(ranked)
    stability = radial_stability_rows(payloads)
    if args.forensic_scale_cache:
        originals = payloads[args.bins[0]]["manifest"]["stability_originals"]
        clean = base["train_features"][:originals]
        for representation_name, block in (("laplacian", FEATURE_BLOCKS["laplacian"]), ("fft", FEATURE_BLOCKS["fft"])):
            for scale in STABILITY_SCALES[1:]:
                scale_path = args.forensic_scale_cache / (f"train_scale_{scale:.2f}".replace(".", "p") + ".pt")
                if scale_path.is_file():
                    transformed = torch.load(scale_path, map_location="cpu", weights_only=True, mmap=True)["features"][:originals]
                    values = cosine_stability(clean[:, block], transformed[:, block])
                    stability.append({"representation": representation_name, "scale": scale,
                                      "mean_cosine_similarity": values[0], "std_cosine_similarity": values[1],
                                      "mean_l2_distance": values[2], "std_l2_distance": values[3]})
    with (args.output_dir / "stability_by_scale.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stability[0])); writer.writeheader(); writer.writerows(stability)
    winning_path = args.output_dir / "winning_config.json"
    if winner is None:
        winning_path.unlink(missing_ok=True)
        print(f"E7 completed with no eligible deployment candidate: {no_eligible_reason}")
    else:
        winning_path.write_text(json.dumps({
            "schema_version": 1, "selection_split": "validation", "test_rows_used_for_selection": False,
            "checkpoint": str(Path(winner["checkpoint"]).resolve()), "model_mode": winner["model_mode"],
            "radial_bins": winner["radial_bins"], "seed": args.seed,
        }, indent=2) + "\n", encoding="utf-8")


def locked_test_command(args):
    locked = json.loads(args.winning_config.read_text(encoding="utf-8"))
    if locked.get("selection_split") != "validation" or locked.get("test_rows_used_for_selection") is not False:
        raise ValueError("E7 configuration was not locked on validation")
    # Dedicated final evaluation is intentionally explicit because E7 adds radial inputs.
    checkpoint = torch.load(locked["checkpoint"], map_location="cpu", weights_only=True)
    rows = load_labeled_paths(args.data_dir)
    _, _, test_rows = stratified_train_val_test_split(rows, args.data_dir, 0.15, 0.15, locked["seed"])
    device = choose_device(args.device)
    config = ModelConfig(forensic_mode="laplacian_fft", forensic_dim=2560)
    encoders = FrozenEncoders(config, device)
    head = nn.Linear(checkpoint["input_dim"], 1); head.load_state_dict(checkpoint["state_dict"]); head.eval()
    results = {}
    for index, condition in enumerate(ROBUSTNESS_CONDITIONS):
        from ..features import extract_condition_features
        fused, labels, _, _ = extract_condition_features(
            test_rows, encoders, args.batch_size, (condition,), locked["seed"]
        )
        tasks = [(path, condition, 0) for path, _ in test_rows]
        radial = extract_descriptor_tasks(tasks, (locked["radial_bins"],), locked["seed"], args.batch_size, args.workers)[locked["radial_bins"]]
        selected = select_features(fused, radial, locked["model_mode"])
        with torch.no_grad(): probabilities = torch.sigmoid(head(selected).squeeze(1) / checkpoint["temperature"])
        results[condition] = classification_metrics(labels, probabilities, checkpoint["threshold"])
    del encoders
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "locked_test_metrics.json").write_text(json.dumps({
        "evaluation_split": "test", "configuration_selection_performed": False,
        "locked_configuration": locked, "per_condition": results,
    }, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="E7 scale-stable radial frequency features")
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--data-dir", type=Path, required=True); cache.add_argument("--base-cache", type=Path, required=True)
    cache.add_argument("--output-dir", type=Path, required=True); cache.add_argument("--bins", type=int, nargs="+", default=list(SUPPORTED_BINS))
    cache.add_argument("--stability-images", type=int, default=500); cache.add_argument("--batch-size", type=int, default=32)
    cache.add_argument("--workers", type=int, default=int(os.getenv("NUM_WORKERS", "4"))); cache.add_argument("--seed", type=int, default=42)
    cache.set_defaults(handler=cache_command)
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--data-dir", type=Path, required=True); sweep.add_argument("--base-cache", type=Path, required=True)
    sweep.add_argument("--validation-cache", type=Path, required=True)
    sweep.add_argument("--feature-cache", type=Path, required=True); sweep.add_argument("--baseline-checkpoint", type=Path, required=True)
    sweep.add_argument("--output-dir", type=Path, required=True); sweep.add_argument("--forensic-scale-cache", type=Path)
    sweep.add_argument("--bins", type=int, nargs="+", default=list(SUPPORTED_BINS)); sweep.add_argument("--modes", nargs="+", choices=MODEL_MODES, default=list(MODEL_MODES))
    sweep.add_argument("--epochs", type=int, default=30); sweep.add_argument("--patience", type=int, default=6)
    sweep.add_argument("--feature-batch-size", type=int, default=8)
    sweep.add_argument("--batch-size", type=int, default=64); sweep.add_argument("--learning-rate", type=float, default=1e-3)
    sweep.add_argument("--seed", type=int, default=42); sweep.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    sweep.set_defaults(handler=sweep_command)
    test = sub.add_parser("locked-test")
    test.add_argument("--data-dir", type=Path, required=True); test.add_argument("--winning-config", type=Path, required=True)
    test.add_argument("--output-dir", type=Path, required=True); test.add_argument("--batch-size", type=int, default=8)
    test.add_argument("--workers", type=int, default=int(os.getenv("NUM_WORKERS", "4"))); test.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    test.set_defaults(handler=locked_test_command)
    args = parser.parse_args(); args.handler(args)


if __name__ == "__main__":
    main()
