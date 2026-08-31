from __future__ import annotations

import argparse
import csv
import json
import inspect
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from aigc_detector.data import ROBUSTNESS_CONDITIONS
from aigc_detector.metrics import select_threshold

from .artifacts import atomic_json, validate_completion, write_artifact_contract
from .config import Phase3Config, load_config, require_validation_selection
from .data import ManifestRecord, exact_track5_transform, load_manifest, manifest_counts
from .metrics import summarize_validation
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


BACKBONES = {
    "dinov3_vitl16": {"model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m", "resolution": 256, "optional": False},
    "siglip2_large_256": {"model_id": "google/siglip2-large-patch16-256", "resolution": 256, "optional": False},
    "siglip2_so400m_256": {"model_id": "google/siglip2-so400m-patch16-256", "resolution": 256, "optional": True},
}
TRAINING_MODES = ("linear_head", "last2", "last4")


def validate_offline_asset_path(path: str | Path, optional: bool = False) -> Path | None:
    path = Path(path)
    if path.is_dir() and any((path / name).is_file() for name in ("config.json", "model.safetensors", "pytorch_model.bin")):
        return path
    if optional:
        return None
    raise FileNotFoundError(f"Offline model asset is absent or incomplete: {path}")


def memory_preflight(minimum_gb: float, optional: bool, cuda_totals: list[float] | None = None) -> tuple[bool, str | None]:
    if cuda_totals is None:
        cuda_totals = [torch.cuda.get_device_properties(index).total_memory / 2**30
                       for index in range(torch.cuda.device_count())]
    if cuda_totals and min(cuda_totals) < minimum_gb:
        reason = f"minimum GPU memory {min(cuda_totals):.1f} GB is below required {minimum_gb:.1f} GB"
        if optional: return False, reason
        raise RuntimeError(reason)
    return True, None


def locate_transformer_blocks(backbone: nn.Module) -> list[nn.Module]:
    candidates = (
        "encoder.layer", "encoder.layers", "vision_model.encoder.layer", "vision_model.encoder.layers",
        "vision_model.head.attention", "layer", "layers", "blocks",
    )
    for dotted in candidates:
        value = backbone
        try:
            for part in dotted.split("."):
                value = getattr(value, part)
        except AttributeError:
            continue
        if isinstance(value, (nn.ModuleList, nn.Sequential, list, tuple)) and len(value):
            return list(value)
    raise ValueError("Could not locate transformer blocks for partial fine-tuning")


def configure_trainable_layers(backbone: nn.Module, classifier: nn.Module, mode: str) -> dict[str, int]:
    if mode not in TRAINING_MODES:
        raise ValueError(f"Unknown R1 training mode: {mode}")
    for parameter in backbone.parameters(): parameter.requires_grad = False
    for parameter in classifier.parameters(): parameter.requires_grad = True
    if mode != "linear_head":
        block_count = int(mode.removeprefix("last"))
        blocks = locate_transformer_blocks(backbone)
        if len(blocks) < block_count:
            raise ValueError(f"Backbone has {len(blocks)} blocks, cannot unfreeze final {block_count}")
        for block in blocks[-block_count:]:
            for parameter in block.parameters(): parameter.requires_grad = True
        for name, module in backbone.named_modules():
            if "norm" in name.lower() or isinstance(module, nn.LayerNorm):
                for parameter in module.parameters(recurse=False): parameter.requires_grad = True
    trainable = sum(p.numel() for p in list(backbone.parameters()) + list(classifier.parameters()) if p.requires_grad)
    total = sum(p.numel() for p in list(backbone.parameters()) + list(classifier.parameters()))
    return {"trainable_parameter_count": trainable, "total_deployment_parameter_count": total}


def pooled_features(output) -> torch.Tensor:
    if getattr(output, "pooler_output", None) is not None:
        return output.pooler_output
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        if isinstance(output, (tuple, list)) and output: hidden = output[0]
        else: raise ValueError("Backbone output has no pooled or hidden representation")
    return hidden[:, 0] if hidden.ndim == 3 else hidden


class VisionDetector(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__(); self.backbone = backbone; self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, pixel_values: torch.Tensor, return_features: bool = False):
        parameters = inspect.signature(self.backbone.forward).parameters
        kwargs = {"pixel_values": pixel_values}
        if "interpolate_pos_encoding" in parameters:
            kwargs["interpolate_pos_encoding"] = True
        features = pooled_features(self.backbone(**kwargs))
        logits = self.classifier(features).squeeze(1)
        return (logits, features) if return_features else logits


def load_offline_model(asset_path: Path) -> tuple[nn.Module, object, int]:
    from transformers import AutoConfig, AutoImageProcessor, AutoModel
    configuration = AutoConfig.from_pretrained(asset_path, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(asset_path, local_files_only=True)
    backbone = AutoModel.from_pretrained(asset_path, local_files_only=True)
    hidden = getattr(configuration, "hidden_size", None) or getattr(configuration, "vision_config", None).hidden_size
    return backbone, processor, int(hidden)


def source_balanced_subset(records: list[ManifestRecord], maximum: int, seed: int) -> list[ManifestRecord]:
    if maximum <= 0 or len(records) <= maximum:
        return records
    groups: dict[tuple, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        key = (record.label, record.generator if record.label else record.source)
        groups[key].append(record)
    rng = random.Random(seed)
    for values in groups.values(): rng.shuffle(values)
    selected = []
    keys = sorted(groups, key=str)
    while len(selected) < maximum:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < maximum:
                selected.append(groups[key].pop()); progressed = True
        if not progressed: break
    return selected


class R1Dataset(Dataset):
    def __init__(self, records: list[ManifestRecord], processor, resolution: int, seed: int,
                 training: bool, condition: str = "clean") -> None:
        self.records, self.processor, self.resolution = records, processor, resolution
        self.seed, self.training, self.condition, self.epoch = seed, training, condition, 0

    def set_epoch(self, epoch: int) -> None: self.epoch = epoch

    def __len__(self) -> int: return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.image_path) as source: image = source.convert("RGB")
        condition = self.condition
        if self.training:
            condition = ROBUSTNESS_CONDITIONS[(index + self.epoch * len(self.records)) % len(ROBUSTNESS_CONDITIONS)]
        image = exact_track5_transform(condition, self.seed, record.base_id or record.image_path, self.epoch)(image)
        values = self.processor(images=image, size={"height": self.resolution, "width": self.resolution}, return_tensors="pt")
        return values["pixel_values"][0], torch.tensor(record.label, dtype=torch.float32), record.image_path


def _collate(batch):
    pixels, labels, paths = zip(*batch, strict=True)
    return torch.stack(pixels), torch.stack(labels), list(paths)


def infer_condition(model, records, processor, config, condition, context) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = R1Dataset(records, processor, config.input_resolution, config.seed, False, condition)
    loader = DataLoader(dataset, batch_size=int(config.training.get("validation_batch_size", 16)),
                        shuffle=False, num_workers=config.dataloader_workers, collate_fn=_collate, pin_memory=True)
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for pixels, target, _ in loader:
            with fp16_autocast(context): output = model(pixels.to(context.device, non_blocking=True))
            logits.append(output.float().cpu()); labels.append(target)
    return torch.cat(logits), torch.cat(labels)


def save_safe_checkpoint(path: Path, model: nn.Module, optimizer, epoch: int, metadata: dict) -> None:
    state_model = model.module if hasattr(model, "module") else model
    temporary = path.with_suffix(".tmp")
    torch.save({"state_dict": state_model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "metadata": metadata}, temporary)
    os.replace(temporary, path)


def train_candidate(config: Phase3Config, records: list[ManifestRecord], asset: Path, mode: str,
                    output: Path, context, initial_checkpoint: Path | None = None,
                    training_dataset_factory=None) -> dict:
    if (output / "COMPLETED.json").is_file():
        return validate_completion(output)["metrics"]
    backbone, processor, hidden = load_offline_model(asset)
    model = VisionDetector(backbone, hidden)
    if initial_checkpoint is not None:
        state = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("state_dict", state), strict=True)
    counts = configure_trainable_layers(model.backbone, model.classifier, mode)
    if config.gradient_checkpointing and mode != "linear_head": enable_gradient_checkpointing(model.backbone, True)
    model = wrap_ddp(model, context)
    train_records = source_balanced_subset([row for row in records if row.split == "train"],
                                           int(config.training.get("max_train_examples", 25000)), config.seed)
    validation = [row for row in records if row.split == "validation"]
    if not train_records or not validation: raise ValueError("R1 requires non-empty train and validation manifests")
    dataset = (training_dataset_factory(train_records, processor, config)
               if training_dataset_factory else R1Dataset(train_records, processor, config.input_resolution, config.seed, True))
    sampler = DistributedSampler(dataset, shuffle=True, seed=config.seed) if context.distributed else None
    loader = DataLoader(dataset, batch_size=int(config.training.get("batch_size_per_gpu", 8)),
                        shuffle=sampler is None, sampler=sampler, num_workers=config.dataloader_workers,
                        collate_fn=_collate, pin_memory=True, drop_last=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(config.training.get("learning_rate", 2e-5)),
                                  weight_decay=float(config.training.get("weight_decay", 0.05)))
    scaler = make_grad_scaler(context); guard = WallClockGuard(config.max_wall_minutes)
    epochs = min(int(config.training.get("epochs", 2)), 2)
    max_steps = int(config.training.get("max_steps", 0)); global_step = 0; stopped = False
    model.train(); optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        if sampler is not None: sampler.set_epoch(epoch)
        for micro_step, (pixels, labels, _) in enumerate(loader):
            with fp16_autocast(context):
                logits = model(pixels.to(context.device, non_blocking=True))
                loss = F.binary_cross_entropy_with_logits(logits, labels.to(context.device)) / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if optimizer_step_due(micro_step, config.gradient_accumulation_steps):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); global_step += 1
                local_stop = bool((max_steps and global_step >= max_steps) or guard.should_stop())
                if context.distributed:
                    flag = torch.tensor([int(local_stop)], device=context.device)
                    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
                    local_stop = bool(flag.item())
                if local_stop: stopped = True; break
        if stopped: break
    if context.distributed: torch.distributed.barrier()
    condition_logits, labels = {}, None
    for condition in ROBUSTNESS_CONDITIONS:
        condition_logits[condition], current_labels = infer_condition(model, validation, processor, config, condition, context)
        if labels is None: labels = current_labels
        elif not torch.equal(labels, current_labels): raise ValueError("Validation label order changed across conditions")
    stacked = torch.stack([condition_logits[name] for name in ROBUSTNESS_CONDITIONS])
    threshold = select_threshold(labels.repeat(len(ROBUSTNESS_CONDITIONS)), torch.sigmoid(stacked.flatten()), "balanced")
    probabilities = {name: torch.sigmoid(condition_logits[name]) for name in ROBUSTNESS_CONDITIONS}
    metric_metadata = {"model_backbone": config.backbone, **counts, "input_resolution": config.input_resolution,
                       "training_data_counts": manifest_counts(train_records), "inference_multiplier": 1,
                       "clean_constraint_pass": False, "status": "succeeded", "candidate_id": f"{config.backbone}:{mode}",
                       "training_mode": mode, "optimizer_steps": global_step, "wall_clock_stopped": guard.should_stop()}
    metrics, condition_rows = summarize_validation(labels, probabilities, threshold, metric_metadata)
    metrics["clean_constraint_pass"] = metrics["clean_validation_balanced_accuracy"] >= config.baseline_clean_balanced_accuracy - 0.01
    metrics["checkpoint"] = str(output / "best_model.pt")
    model_to_save = model.module if hasattr(model, "module") else model
    candidate = {"experiment": "R1", "candidate_id": metrics["candidate_id"], "selection_split": "validation",
                 "final_test_evaluated": False, "threshold": threshold}
    if context.is_primary:
        write_artifact_contract(output, config.to_dict(), metrics, condition_rows, stacked.numpy(), labels.numpy(),
                                model_to_save.state_dict(), candidate)
    if context.distributed: torch.distributed.barrier()
    return metrics if context.is_primary else {"status": "worker_complete"}


def write_job_summary(rows: list[dict], output: Path, baseline_clean: float) -> None:
    ranked = rank_candidates(rows, baseline_clean, effective_tie=0.002)
    eligible = [row for row in ranked if row["validation_rank"] is not None]
    winner = min(eligible, key=lambda row: row["validation_rank"]) if eligible else None
    output.mkdir(parents=True, exist_ok=True)
    document = {"experiment": "R1", "selection_split": "validation", "final_test_evaluated": False,
                "eligible_winner": winner, "no_eligible_candidate": winner is None, "results": ranked}
    atomic_json(output / "r1_summary.json", document)
    fields = sorted({key for row in ranked for key, value in row.items() if not isinstance(value, (dict, list))})
    with (output / "r1_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in ranked)
    recommended = None if winner is None else {
        **winner,
        "checkpoint_relative_path": f"candidates/{winner['training_mode']}/best_model.pt",
    }
    atomic_json(output / "recommended_candidate.json", {
        "experiment": "R1", "selection_split": "validation", "final_test_evaluated": False,
        "candidate": recommended, "reason": None if winner else "No R1 candidate passed the validation clean constraint.",
    })


def run(config_path: Path, manifest_path: Path, output: Path, asset_override: Path | None = None) -> None:
    config = load_config(config_path); require_validation_selection(config.selection_split)
    spec = BACKBONES.get(config.backbone)
    if spec is None: raise ValueError(f"Unsupported R1 backbone: {config.backbone}")
    asset = validate_offline_asset_path(asset_override or config.model.get("asset_path", ""), spec["optional"])
    if asset is None:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "SKIPPED.json", {"status": "skipped", "reason": "optional offline asset missing",
                                               "backbone": config.backbone})
        return
    safe, reason = memory_preflight(float(config.model.get("minimum_gpu_memory_gb", 0)), spec["optional"])
    if not safe:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "SKIPPED.json", {"status": "skipped", "reason": reason, "backbone": config.backbone})
        return
    context = resolve_distributed(); initialize_process_group(context, config.distributed.backend); seed_everything(config.seed, context.rank)
    records = load_manifest(manifest_path); rows = []
    modes = config.training.get("modes", ["linear_head", "last2"])
    for index, mode in enumerate(modes):
        try:
            result = train_candidate(config, records, asset, mode, output / "candidates" / mode, context)
        except Exception as error:
            result = {"candidate_id": f"{config.backbone}:{mode}", "training_mode": mode, "status": "failed",
                      "failure_reason": f"{type(error).__name__}: {error}"}
        if context.is_primary: rows.append(result)
        continue_screen = result.get("status") == "succeeded" if context.is_primary else True
        if context.distributed:
            flag = torch.tensor([int(continue_screen)], device=context.device)
            torch.distributed.broadcast(flag, src=0); continue_screen = bool(flag.item())
        if index == 0 and not continue_screen: break
    if context.is_primary: write_job_summary(rows, output, config.baseline_clean_balanced_accuracy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-3 R1 modern vision-backbone screening")
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--asset-path", type=Path)
    args = parser.parse_args(); run(args.config, args.manifest, args.output, args.asset_path)


if __name__ == "__main__":
    main()
