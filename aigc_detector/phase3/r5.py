from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aigc_detector.data import DeterministicTransform, ROBUSTNESS_CONDITIONS
from aigc_detector.metrics import select_threshold

from .artifacts import atomic_json, validate_completion
from .config import load_config, require_validation_selection
from .data import ManifestRecord, exact_track5_transform, load_manifest
from .metrics import summarize_validation
from .r1 import BACKBONES, validate_offline_asset_path
from .r2 import validate_no_split_leakage
from .r3 import CONSISTENCY_CONFIGS, train_paired
from .r4 import select_training_records
from .ranking import rank_candidates
from .runtime import initialize_process_group, resolve_distributed, seed_everything


EXPERTS = {"low": {"resolution": 256, "inference_multiplier": 1.0},
           "high": {"resolution": 384, "inference_multiplier": 2.25}}
ALPHAS = tuple(round(value / 10, 1) for value in range(11))
PARAMETER_LIMIT = 2_000_000_000


def assert_checkpoint_compatible(model: torch.nn.Module, checkpoint: dict) -> None:
    state = checkpoint.get("state_dict", checkpoint)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state)); unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(key for key in set(expected) & set(state) if expected[key].shape != state[key].shape)
    if missing or unexpected or mismatched:
        raise ValueError(f"Incompatible warm-start checkpoint: missing={missing[:3]}, "
                         f"unexpected={unexpected[:3]}, shape_mismatch={mismatched[:3]}")


def specialist_chain(expert: str, seed: int, identity: str, epoch: int) -> tuple[str, ...]:
    if expert not in EXPERTS: raise ValueError(f"Unknown R5 expert: {expert}")
    digest = hashlib.sha256(f"r5\0{expert}\0{seed}\0{identity}\0{epoch}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    if expert == "low":
        primary = rng.choice(("resize_x0.25", "resize_x0.25", "resize_x0.5", "blur_s2.0",
                              "blur_s1.0", "noise_s0.10", "noise_s0.05"))
        secondary = rng.choice((None, "jpeg_q70", "jpeg_q50", "noise_s0.05"))
        return (primary,) if secondary is None else (primary, secondary)
    # Fine texture is retained most of the time; destructive views are occasional.
    primary = rng.choice(("clean", "clean", "jpeg_q90", "jpeg_q70", "blur_s0.5",
                          "noise_s0.02", "color_0.8", "color_1.2", "resize_x0.5"))
    return (primary,)


class SpecialistPairedDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[ManifestRecord], processor, resolution: int, seed: int,
                 expert: str, epochs: int) -> None:
        self.records, self.processor, self.resolution = records, processor, resolution
        self.seed, self.expert, self.epochs, self.epoch = seed, expert, max(epochs, 1), 0

    def set_epoch(self, epoch: int) -> None: self.epoch = epoch
    def __len__(self) -> int: return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as source: original = source.convert("RGB")
        identity = record.unique_id or record.path
        clean = exact_track5_transform("clean", self.seed, identity, self.epoch)(original.copy())
        chain = specialist_chain(self.expert, self.seed, identity, self.epoch)
        corruption = "+".join(item for item in chain if item != "clean") or "clean"
        corrupt = DeterministicTransform(corruption, self.seed, identity, self.epoch)(original.copy())
        size = {"height": self.resolution, "width": self.resolution}
        clean_pixels = self.processor(images=clean, size=size, return_tensors="pt")["pixel_values"][0]
        corrupt_pixels = self.processor(images=corrupt, size=size, return_tensors="pt")["pixel_values"][0]
        return clean_pixels, corrupt_pixels, torch.tensor(record.label, dtype=torch.float32), record.path


def blend_logits(low: np.ndarray, high: np.ndarray, alpha: float) -> np.ndarray:
    if not 0 <= alpha <= 1: raise ValueError("alpha must be in [0, 1]")
    if low.shape != high.shape: raise ValueError("Expert logit arrays must have identical shapes")
    return alpha * low + (1 - alpha) * high


def deployment_parameters(low_parameters: int, high_parameters: int) -> int:
    total = int(low_parameters) + int(high_parameters)
    if total >= PARAMETER_LIMIT:
        raise ValueError(f"R5 ensemble has {total:,} parameters and violates the <2B limit")
    return total


def load_r4_candidate(recommendation: Path, output: Path) -> tuple[dict, Path]:
    document = json.loads(recommendation.read_text(encoding="utf-8"))
    if document.get("selection_split") != "validation" or document.get("final_test_evaluated") is not False:
        raise ValueError("R4 recommendation is not validation-only")
    candidate = document.get("candidate")
    if not candidate or not candidate.get("clean_constraint_pass"): raise ValueError("R4 has no eligible champion")
    checkpoint = output / candidate.get("checkpoint_relative_path", "candidate/best_model.pt")
    if not checkpoint.is_file(): raise FileNotFoundError(f"R4 checkpoint missing: {checkpoint}")
    return candidate, checkpoint


def discover_output(experiment: str, input_root: Path = Path("/kaggle/input")) -> Path:
    matches = []
    paths = set(input_root.glob("*/recommended_candidate.json")) | set(input_root.glob("*/*/recommended_candidate.json"))
    for path in paths:
        try: document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): continue
        if document.get("experiment") == experiment and document.get("selection_split") == "validation": matches.append(path.parent)
    if len(matches) != 1: raise ValueError(f"Expected exactly one attached {experiment} output, found {len(matches)}")
    return matches[0]


def run_expert(config_path: Path, manifest_path: Path, r4_recommendation: Path,
               r4_output: Path, output: Path) -> None:
    config = load_config(config_path); require_validation_selection(config.selection_split)
    expert = str(config.training["expert"])
    if expert not in EXPERTS or config.input_resolution != EXPERTS[expert]["resolution"]:
        raise ValueError("R5 expert resolution does not match its centralized definition")
    candidate, checkpoint = load_r4_candidate(r4_recommendation, r4_output)
    config = replace(config, backbone=candidate["model_backbone"])
    records = load_manifest(manifest_path); validate_no_split_leakage(records)
    selected, distribution = select_training_records(records, int(config.training["max_train_examples"]),
                                                     config.seed, candidate["bias_policy"])
    asset = validate_offline_asset_path(config.model.get("asset_paths", {}).get(config.backbone, ""),
                                        BACKBONES[config.backbone]["optional"])
    if asset is None: raise FileNotFoundError("R4 champion backbone assets are not attached")
    setting = candidate.get("consistency_setting", "baseline")
    if setting not in CONSISTENCY_CONFIGS: raise ValueError(f"Unknown inherited consistency setting: {setting}")
    context = resolve_distributed(); initialize_process_group(context, config.distributed.backend)
    seed_everything(config.seed, context.rank)
    factory = lambda rows, processor, cfg: SpecialistPairedDataset(
        rows, processor, cfg.input_resolution, cfg.seed, expert, int(cfg.training.get("epochs", 2)))
    metrics = train_paired(
        config, records, asset, checkpoint, candidate, setting, output / "candidate", context,
        selected_records=selected, training_distribution=distribution, experiment="R5",
        extra_metadata={"expert": expert, "candidate_id": f"r5:{expert}", "warm_started_from_r4": True,
                        "inference_multiplier": EXPERTS[expert]["inference_multiplier"],
                        "bias_policy": candidate["bias_policy"]},
        training_dataset_factory=factory, calibrate_logits=True,
        checkpoint_validator=assert_checkpoint_compatible,
    )
    if context.is_primary:
        ranked = rank_candidates([metrics], config.baseline_clean_balanced_accuracy, effective_tie=0.002)
        recommended = ranked[0] if ranked[0]["validation_rank"] else None
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "expert_summary.json", {"experiment": "R5", "expert": expert,
                    "selection_split": "validation", "final_test_evaluated": False, "results": ranked})
        atomic_json(output / "recommended_candidate.json", {"experiment": f"R5-{expert}",
                    "candidate": recommended, "selection_split": "validation", "final_test_evaluated": False})


def _load_expert(output: Path, expected: str) -> tuple[dict, np.ndarray, np.ndarray]:
    contract = validate_completion(output / "candidate"); metrics = contract["metrics"]
    if metrics.get("expert") != expected or not metrics.get("logits_calibrated"):
        raise ValueError(f"Attached {expected} expert is incompatible or uncalibrated")
    arrays = np.load(output / "candidate/val_logits.npz")
    return metrics, arrays["logits"], arrays["labels"]


def ensemble_candidates(low_metrics: dict, high_metrics: dict, low_logits: np.ndarray,
                        high_logits: np.ndarray, labels: np.ndarray, baseline_clean: float) -> tuple[list[dict], dict[str, np.ndarray]]:
    if low_logits.shape != high_logits.shape or low_logits.shape[0] != len(ROBUSTNESS_CONDITIONS):
        raise ValueError("R5 expert validation logits are misaligned")
    total = deployment_parameters(low_metrics["total_deployment_parameter_count"],
                                  high_metrics["total_deployment_parameter_count"])
    rows, logits_by_id = [], {}
    for name, metrics, logits in (("low", low_metrics, low_logits), ("high", high_metrics, high_logits)):
        row = {**metrics, "candidate_id": f"r5:{name}", "candidate_type": "single_expert",
               "expert": name, "alpha": 1.0 if name == "low" else 0.0}
        rows.append(row); logits_by_id[row["candidate_id"]] = logits
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    for alpha in ALPHAS[1:-1]:
        logits = blend_logits(low_logits, high_logits, alpha)
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        threshold = select_threshold(label_tensor.repeat(len(ROBUSTNESS_CONDITIONS)),
                                     torch.sigmoid(logits_tensor.flatten()), "balanced")
        probabilities = {condition: torch.sigmoid(logits_tensor[index])
                         for index, condition in enumerate(ROBUSTNESS_CONDITIONS)}
        candidate_id = f"r5:ensemble:{alpha:.1f}"
        metadata = {"candidate_id": candidate_id, "candidate_type": "two_expert_ensemble", "alpha": alpha,
                    "model_backbone": f"{low_metrics['model_backbone']}+{high_metrics['model_backbone']}",
                    "trainable_parameter_count": (low_metrics["trainable_parameter_count"] +
                                                   high_metrics["trainable_parameter_count"]),
                    "total_deployment_parameter_count": total, "input_resolution": "256+384",
                    "training_data_counts": {"low": low_metrics["training_data_counts"],
                                             "high": high_metrics["training_data_counts"]},
                    "inference_multiplier": 3.25, "status": "succeeded",
                    "parameter_limit": PARAMETER_LIMIT, "parameter_limit_pass": True,
                    "approximate_pixel_compute_multiplier": 3.25}
        summary, _ = summarize_validation(label_tensor, probabilities, threshold, metadata)
        rows.append(summary); logits_by_id[candidate_id] = logits
    return rank_candidates(rows, baseline_clean, effective_tie=0.002), logits_by_id


def run_ensemble(low_output: Path, high_output: Path, output: Path, baseline_clean: float) -> None:
    low_metrics, low_logits, low_labels = _load_expert(low_output, "low")
    high_metrics, high_logits, high_labels = _load_expert(high_output, "high")
    if not np.array_equal(low_labels, high_labels): raise ValueError("Expert validation labels are not aligned")
    ranked, logits_by_id = ensemble_candidates(low_metrics, high_metrics, low_logits, high_logits,
                                                low_labels, baseline_clean)
    winner = next((row for row in ranked if row["validation_rank"] == 1), None)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "r5_summary.json", {"experiment": "R5", "selection_split": "validation",
                "final_test_evaluated": False, "parameter_limit": PARAMETER_LIMIT,
                "results": ranked, "eligible_winner": winner})
    fields = sorted({key for row in ranked for key, value in row.items() if not isinstance(value, (dict, list))})
    with (output / "r5_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in ranked)
    recommendation = None
    if winner:
        recommendation = {**winner, "low_checkpoint_kernel_path": "low/candidate/best_model.pt",
                          "high_checkpoint_kernel_path": "high/candidate/best_model.pt"}
        np.savez_compressed(output / "val_logits.npz", logits=logits_by_id[winner["candidate_id"]], labels=low_labels)
    atomic_json(output / "recommended_candidate.json", {"experiment": "R5", "candidate": recommendation,
                "selection_split": "validation", "final_test_evaluated": False})


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-3 R5 multi-resolution specialists")
    subparsers = parser.add_subparsers(dest="command", required=True)
    expert = subparsers.add_parser("expert"); expert.add_argument("--config", type=Path, required=True)
    expert.add_argument("--manifest", type=Path, required=True); expert.add_argument("--r4-recommendation", type=Path, required=True)
    expert.add_argument("--r4-output", type=Path, required=True); expert.add_argument("--output", type=Path, required=True)
    ensemble = subparsers.add_parser("ensemble"); ensemble.add_argument("--low-output", type=Path, required=True)
    ensemble.add_argument("--high-output", type=Path, required=True); ensemble.add_argument("--output", type=Path, required=True)
    ensemble.add_argument("--baseline-clean", type=float, default=0.9681)
    args = parser.parse_args()
    if args.command == "expert": run_expert(args.config, args.manifest, args.r4_recommendation, args.r4_output, args.output)
    else: run_ensemble(args.low_output, args.high_output, args.output, args.baseline_clean)


if __name__ == "__main__": main()
