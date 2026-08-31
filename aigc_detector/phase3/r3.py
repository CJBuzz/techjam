from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler

from aigc_detector.data import DeterministicTransform, ROBUSTNESS_CONDITIONS
from aigc_detector.metrics import select_threshold

from .artifacts import atomic_json, write_artifact_contract
from .config import load_config, require_validation_selection
from .data import ManifestRecord, exact_track5_transform, load_manifest, manifest_counts
from .metrics import summarize_validation
from .r1 import (
    BACKBONES,
    R1Dataset,
    VisionDetector,
    _collate,
    configure_trainable_layers,
    load_offline_model,
    validate_offline_asset_path,
)
from .r2 import balanced_training_records, curriculum_chain, validate_no_split_leakage
from .ranking import rank_candidates
from .runtime import (
    WallClockGuard,
    enable_gradient_checkpointing,
    fp16_autocast,
    initialize_process_group,
    make_grad_scaler,
    optimizer_step_due,
    resolve_distributed,
    seed_everything,
    wrap_ddp,
)


CONSISTENCY_CONFIGS = {
    "baseline": {"lambda_pred": 0.0, "lambda_feat": 0.0},
    "mild": {"lambda_pred": 0.05, "lambda_feat": 0.02},
    "medium": {"lambda_pred": 0.15, "lambda_feat": 0.05},
    "strong": {"lambda_pred": 0.35, "lambda_feat": 0.15},
}


def paired_forward(model, clean: torch.Tensor, corrupt: torch.Tensor) -> tuple[torch.Tensor, ...]:
    batch = clean.shape[0]
    logits, features = model(torch.cat((clean, corrupt), dim=0), return_features=True)
    return logits[:batch], logits[batch:], features[:batch], features[batch:]


def asymmetric_prediction_divergence(clean_logits: torch.Tensor, corrupt_logits: torch.Tensor) -> torch.Tensor:
    teacher = torch.sigmoid(clean_logits).detach().clamp(1e-6, 1 - 1e-6)
    student = torch.sigmoid(corrupt_logits).clamp(1e-6, 1 - 1e-6)
    return (teacher * (teacher.log() - student.log()) +
            (1 - teacher) * ((1 - teacher).log() - (1 - student).log())).mean()


def asymmetric_feature_distance(clean_features: torch.Tensor, corrupt_features: torch.Tensor) -> torch.Tensor:
    teacher = F.normalize(clean_features, dim=1).detach()
    student = F.normalize(corrupt_features, dim=1)
    return (1 - (teacher * student).sum(dim=1)).mean()


def paired_loss(clean_logits: torch.Tensor, corrupt_logits: torch.Tensor,
                clean_features: torch.Tensor, corrupt_features: torch.Tensor,
                labels: torch.Tensor, lambda_pred: float, lambda_feat: float) -> dict[str, torch.Tensor]:
    classification = (F.binary_cross_entropy_with_logits(clean_logits, labels) +
                      F.binary_cross_entropy_with_logits(corrupt_logits, labels))
    prediction = (asymmetric_prediction_divergence(clean_logits, corrupt_logits)
                  if lambda_pred else classification.new_zeros(()))
    feature = (asymmetric_feature_distance(clean_features, corrupt_features)
               if lambda_feat else classification.new_zeros(()))
    total = classification + lambda_pred * prediction + lambda_feat * feature
    return {"total": total, "classification": classification, "prediction": prediction, "feature": feature}


class PairedDataset(torch.utils.data.Dataset):
    """Returns pixels and label only; corruption identities never enter the classifier."""
    def __init__(self, records: list[ManifestRecord], processor, resolution: int, seed: int,
                 regime: str, epochs: int) -> None:
        self.records, self.processor, self.resolution, self.seed = records, processor, resolution, seed
        self.regime, self.epochs, self.epoch = regime, max(epochs, 1), 0

    def set_epoch(self, epoch: int) -> None: self.epoch = epoch
    def __len__(self): return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as source: original = source.convert("RGB")
        identity = record.unique_id or record.path
        clean = exact_track5_transform("clean", self.seed, identity, self.epoch)(original.copy())
        progress = self.epoch / max(self.epochs - 1, 1)
        chain = curriculum_chain(self.regime, progress, self.seed, f"{identity}:{self.epoch}")
        group = "+".join(item for item in chain if item != "clean") or "clean"
        corrupt = DeterministicTransform(group, self.seed, identity, self.epoch)(original.copy())
        clean_values = self.processor(images=clean, size={"height": self.resolution, "width": self.resolution}, return_tensors="pt")
        corrupt_values = self.processor(images=corrupt, size={"height": self.resolution, "width": self.resolution}, return_tensors="pt")
        return clean_values["pixel_values"][0], corrupt_values["pixel_values"][0], torch.tensor(record.label, dtype=torch.float32), record.path


def _paired_collate(batch):
    clean, corrupt, labels, paths = zip(*batch, strict=True)
    return torch.stack(clean), torch.stack(corrupt), torch.stack(labels), list(paths)


def infer_with_features(model, records, processor, config, condition, context):
    dataset = R1Dataset(records, processor, config.input_resolution, config.seed, False, condition)
    loader = DataLoader(dataset, batch_size=int(config.training.get("validation_batch_size", 16)),
                        shuffle=False, num_workers=config.dataloader_workers, collate_fn=_collate, pin_memory=True)
    logits, features, labels = [], [], []
    model.eval()
    with torch.no_grad():
        for pixels, target, _ in loader:
            with fp16_autocast(context): output, representation = model(
                pixels.to(context.device, non_blocking=True), return_features=True
            )
            logits.append(output.float().cpu()); features.append(representation.float().cpu()); labels.append(target)
    return torch.cat(logits), torch.cat(features), torch.cat(labels)


def load_r2_candidate(recommendation_path: Path, r2_output: Path) -> tuple[dict, Path]:
    document = json.loads(recommendation_path.read_text(encoding="utf-8"))
    if document.get("selection_split") != "validation" or document.get("final_test_evaluated") is not False:
        raise ValueError("R2 recommendation is not validation-only")
    candidate = document.get("candidate")
    if not candidate or not candidate.get("clean_constraint_pass"):
        raise ValueError("R2 has no clean-eligible recommended candidate")
    relative = candidate.get("checkpoint_relative_path")
    checkpoint = r2_output / relative if relative else Path(candidate.get("checkpoint", ""))
    if not checkpoint.is_file(): raise FileNotFoundError(f"R2 checkpoint missing: {checkpoint}")
    return candidate, checkpoint


def discover_r2_output(input_root: Path = Path("/kaggle/input")) -> Path:
    matches = []
    for path in input_root.glob("*/recommended_candidate.json"):
        try: document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): continue
        if document.get("experiment") == "R2" and document.get("selection_split") == "validation": matches.append(path.parent)
    if len(matches) != 1: raise ValueError(f"Expected exactly one attached R2 champion output, found {len(matches)}")
    return matches[0]


def train_paired(config, records, asset, checkpoint, candidate, setting_name, output, context,
                 selected_records=None, training_distribution=None, experiment="R3",
                 extra_metadata=None, post_validation_callback=None, training_dataset_factory=None,
                 calibrate_logits=False, checkpoint_validator=None, model_factory=None,
                 model_diagnostics_callback=None):
    setting = CONSISTENCY_CONFIGS[setting_name]
    backbone, processor, hidden = load_offline_model(asset)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if model_factory is None:
        model = VisionDetector(backbone, hidden)
        if checkpoint_validator is not None: checkpoint_validator(model, state)
        model.load_state_dict(state.get("state_dict", state), strict=True)
    else:
        model = model_factory(backbone, hidden, state)
    counts = configure_trainable_layers(model.backbone, model.classifier, candidate["training_mode"])
    if config.gradient_checkpointing and candidate["training_mode"] != "linear_head":
        enable_gradient_checkpointing(model.backbone, True)
    model = wrap_ddp(model, context)
    if selected_records is None:
        selected, distribution = balanced_training_records(
            records, int(config.training.get("max_train_examples", 50000)), config.seed,
            config.training.get("max_fake_per_generator")
        )
    else:
        selected = list(selected_records)
        distribution = training_distribution or manifest_counts(selected)
    validation = [row for row in records if row.split == "validation"]
    dataset = (training_dataset_factory(selected, processor, config) if training_dataset_factory else
               PairedDataset(selected, processor, config.input_resolution, config.seed,
                             candidate.get("regime", "compound_curriculum"), int(config.training.get("epochs", 2))))
    sampler = DistributedSampler(dataset, shuffle=True, seed=config.seed) if context.distributed else None
    loader = DataLoader(dataset, batch_size=int(config.training.get("batch_size_per_gpu", 4)), sampler=sampler,
                        shuffle=sampler is None, num_workers=config.dataloader_workers,
                        collate_fn=_paired_collate, pin_memory=True, drop_last=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=float(config.training.get("learning_rate", 1e-5)), weight_decay=0.05)
    scaler = make_grad_scaler(context); guard = WallClockGuard(config.max_wall_minutes)
    accumulation = config.gradient_accumulation_steps; optimizer.zero_grad(set_to_none=True)
    sums = {"classification": 0.0, "prediction": 0.0, "feature": 0.0}; batches = steps = 0; stopped = False
    for epoch in range(min(int(config.training.get("epochs", 2)), 2)):
        dataset.set_epoch(epoch)
        if sampler is not None: sampler.set_epoch(epoch)
        model.train()
        for micro_step, (clean, corrupt, labels, _) in enumerate(loader):
            labels = labels.to(context.device)
            with fp16_autocast(context):
                clean_logits, corrupt_logits, clean_features, corrupt_features = paired_forward(
                    model, clean.to(context.device, non_blocking=True), corrupt.to(context.device, non_blocking=True)
                )
                losses = paired_loss(clean_logits, corrupt_logits, clean_features, corrupt_features, labels,
                                     setting["lambda_pred"], setting["lambda_feat"])
                scaled_loss = losses["total"] / accumulation
            scaler.scale(scaled_loss).backward(); batches += 1
            for name in sums: sums[name] += float(losses[name].detach())
            if optimizer_step_due(micro_step, accumulation):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); steps += 1
                local_stop = guard.should_stop()
                if context.distributed:
                    flag = torch.tensor([int(local_stop)], device=context.device)
                    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX); local_stop = bool(flag.item())
                if local_stop: stopped = True; break
        if stopped: break
    if context.distributed: torch.distributed.barrier()
    condition_logits, condition_features, labels = {}, {}, None
    for condition in ROBUSTNESS_CONDITIONS:
        logits, features, current_labels = infer_with_features(model, validation, processor, config, condition, context)
        condition_logits[condition], condition_features[condition] = logits, features
        if labels is None: labels = current_labels
        elif not torch.equal(labels, current_labels): raise ValueError("Validation labels changed between conditions")
    stacked = torch.stack([condition_logits[name] for name in ROBUSTNESS_CONDITIONS])
    temperature = 1.0
    if calibrate_logits:
        log_temperature = torch.zeros((), requires_grad=True)
        repeated_labels = labels.repeat(len(ROBUSTNESS_CONDITIONS))
        optimizer_temperature = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=40)
        def closure():
            optimizer_temperature.zero_grad()
            loss = F.binary_cross_entropy_with_logits(stacked.flatten() / log_temperature.exp(), repeated_labels)
            loss.backward(); return loss
        optimizer_temperature.step(closure)
        temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
        condition_logits = {name: values / temperature for name, values in condition_logits.items()}
        stacked = torch.stack([condition_logits[name] for name in ROBUSTNESS_CONDITIONS])
    threshold = select_threshold(labels.repeat(len(ROBUSTNESS_CONDITIONS)), torch.sigmoid(stacked.flatten()), "balanced")
    probabilities = {name: torch.sigmoid(condition_logits[name]) for name in ROBUSTNESS_CONDITIONS}
    metadata = {"model_backbone": config.backbone, **counts, "input_resolution": config.input_resolution,
                "training_data_counts": distribution, "inference_multiplier": 1, "clean_constraint_pass": False,
                "status": "succeeded", "candidate_id": f"{config.backbone}:{candidate['training_mode']}:{setting_name}",
                "training_mode": candidate["training_mode"], "consistency_setting": setting_name, **setting,
                "classification_loss": sums["classification"] / max(batches, 1),
                "prediction_consistency_loss": sums["prediction"] / max(batches, 1),
                "feature_consistency_loss": sums["feature"] / max(batches, 1),
                "optimizer_steps": steps, "wall_clock_stopped": guard.should_stop(),
                "temperature": temperature, "logits_calibrated": bool(calibrate_logits)}
    if extra_metadata: metadata.update(extra_metadata)
    metrics, condition_rows = summarize_validation(labels, probabilities, threshold, metadata)
    metrics["clean_constraint_pass"] = metrics["clean_validation_balanced_accuracy"] >= config.baseline_clean_balanced_accuracy - 0.01
    metrics["checkpoint"] = str(output / "best_model.pt")
    clean_reference = F.normalize(condition_features["clean"], dim=1)
    stability = []
    for condition in ROBUSTNESS_CONDITIONS:
        transformed = F.normalize(condition_features[condition], dim=1)
        distance = 1 - (clean_reference * transformed).sum(dim=1)
        stability.append({"condition": condition, "mean_cosine_distance": float(distance.mean()),
                          "std_cosine_distance": float(distance.std(unbiased=False))})
    if context.is_primary:
        if post_validation_callback is not None:
            post_validation_callback(validation, condition_features, output)
        state_model = model.module if hasattr(model, "module") else model
        if model_diagnostics_callback is not None:
            model_diagnostics_callback(state_model, validation, processor, config, context, output)
        write_artifact_contract(output, config.to_dict(), metrics, condition_rows, stacked.numpy(), labels.numpy(),
                                state_model.state_dict(), {"experiment": experiment, "candidate_id": metrics["candidate_id"],
                                "selection_split": "validation", "final_test_evaluated": False})
        with (output / "pair_stability_by_condition.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stability[0])); writer.writeheader(); writer.writerows(stability)
    if context.distributed: torch.distributed.barrier()
    return metrics if context.is_primary else {"status": "worker_complete"}


def write_r3_summary(rows, output, baseline_clean):
    ranked = rank_candidates(rows, baseline_clean, effective_tie=0.002)
    eligible = [row for row in ranked if row["validation_rank"] is not None]
    winner = min(eligible, key=lambda row: row["validation_rank"]) if eligible else None
    output.mkdir(parents=True, exist_ok=True)
    recommended = None if winner is None else {**winner, "checkpoint_relative_path": "candidate/best_model.pt"}
    atomic_json(output / "r3_summary.json", {"experiment": "R3", "selection_split": "validation",
                "final_test_evaluated": False, "eligible_winner": winner, "results": ranked})
    fields = sorted({key for row in ranked for key, value in row.items() if not isinstance(value, (dict, list))})
    with (output / "r3_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in ranked)
    atomic_json(output / "recommended_candidate.json", {"experiment": "R3", "candidate": recommended,
                "selection_split": "validation", "final_test_evaluated": False})
    source = output / "candidate/pair_stability_by_condition.csv"
    if source.is_file(): (output / "pair_stability_by_condition.csv").write_bytes(source.read_bytes())


def select_promotion_setting(summaries: list[dict], baseline_clean: float) -> dict:
    rows = []
    for summary in summaries:
        if summary.get("selection_split") != "validation" or summary.get("final_test_evaluated") is not False:
            raise ValueError("R3 promotion selection is validation-only")
        rows.extend(summary["results"])
    ranked = rank_candidates(rows, baseline_clean, effective_tie=0.002)
    eligible = [row for row in ranked if row["validation_rank"] is not None]
    winner = min(eligible, key=lambda row: row["validation_rank"]) if eligible else None
    if winner is None: raise ValueError("No eligible R3 consistency setting can be promoted")
    return {"selected_setting": winner["consistency_setting"], "selection_split": "validation",
            "final_test_evaluated": False, "candidate_id": winner["candidate_id"]}


def run(config_path: Path, manifest_path: Path, r2_recommendation: Path, r2_output: Path,
        output: Path, promotion_config: Path | None = None) -> None:
    config = load_config(config_path); require_validation_selection(config.selection_split)
    candidate, checkpoint = load_r2_candidate(r2_recommendation, r2_output)
    config = replace(config, backbone=candidate["model_backbone"])
    setting = str(config.training["consistency_setting"])
    if promotion_config:
        promotion = json.loads(promotion_config.read_text(encoding="utf-8"))
        if promotion.get("selection_split") != "validation" or promotion.get("final_test_evaluated") is not False:
            raise ValueError("R3 promotion config is not validation-only")
        setting = promotion["selected_setting"]
    if setting not in CONSISTENCY_CONFIGS: raise ValueError(f"Unknown consistency setting: {setting}")
    records = load_manifest(manifest_path); validate_no_split_leakage(records)
    asset = validate_offline_asset_path(config.model.get("asset_paths", {}).get(config.backbone, ""),
                                        BACKBONES[config.backbone]["optional"])
    if asset is None: raise FileNotFoundError("R2 champion backbone assets are not attached")
    context = resolve_distributed(); initialize_process_group(context, config.distributed.backend); seed_everything(config.seed, context.rank)
    metrics = train_paired(config, records, asset, checkpoint, candidate, setting, output / "candidate", context)
    if context.is_primary: write_r3_summary([metrics], output, config.baseline_clean_balanced_accuracy)


def main():
    parser = argparse.ArgumentParser(description="Phase-3 R3 paired clean/degraded consistency")
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--r2-recommendation", type=Path, required=True); parser.add_argument("--r2-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--promotion-config", type=Path)
    args = parser.parse_args(); run(args.config, args.manifest, args.r2_recommendation, args.r2_output,
                                    args.output, args.promotion_config)


if __name__ == "__main__": main()
