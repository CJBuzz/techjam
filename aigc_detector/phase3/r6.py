from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aigc_detector.data import ROBUSTNESS_CONDITIONS

from .artifacts import atomic_json
from .config import load_config, require_validation_selection
from .data import load_manifest
from .r1 import BACKBONES, R1Dataset, _collate, pooled_features, validate_offline_asset_path
from .r2 import validate_no_split_leakage
from .r3 import CONSISTENCY_CONFIGS, train_paired
from .r4 import load_r3_candidate, select_training_records
from .ranking import rank_candidates
from .runtime import fp16_autocast, initialize_process_group, resolve_distributed, seed_everything


HEAD_MODES = ("global_only", "mean_patch", "topk_patch", "attention_pool", "global_plus_local")
LOCAL_MODES = ("mean_patch", "topk_patch", "attention_pool")
DIAGNOSTIC_CONDITIONS = ("clean", "resize_x0.25", "crop_0.8", "blur_s2.0")


def _patch_size(backbone: nn.Module) -> int:
    config = getattr(backbone, "config", None)
    value = getattr(config, "patch_size", None)
    if value is None and getattr(config, "vision_config", None) is not None:
        value = getattr(config.vision_config, "patch_size", None)
    if isinstance(value, (tuple, list)): value = value[0]
    if not value: raise ValueError("Backbone does not expose a patch size")
    return int(value)


def extract_global_and_patches(backbone: nn.Module, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = {"pixel_values": pixels, "output_hidden_states": False, "return_dict": True}
    import inspect
    parameters = inspect.signature(backbone.forward).parameters
    if "interpolate_pos_encoding" in parameters: kwargs["interpolate_pos_encoding"] = True
    output = backbone(**{key: value for key, value in kwargs.items() if key in parameters or
                         any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())})
    global_representation = pooled_features(output)
    tokens = getattr(output, "last_hidden_state", None)
    if tokens is None or tokens.ndim != 3: raise ValueError("Backbone does not expose patch token representations")
    patch = _patch_size(backbone); expected = (pixels.shape[-2] // patch) * (pixels.shape[-1] // patch)
    if tokens.shape[1] == expected + 1: patches = tokens[:, 1:]
    elif tokens.shape[1] == expected: patches = tokens
    else: raise ValueError(f"Token count {tokens.shape[1]} is incompatible with expected patch grid {expected}")
    return global_representation, patches


def topk_count(patch_count: int, fraction: float) -> int:
    if patch_count < 1: raise ValueError("At least one patch token is required")
    if not 0 < fraction <= 1: raise ValueError("top-k fraction must be in (0, 1]")
    return max(1, min(patch_count, int(round(patch_count * fraction))))


class LocalHead(nn.Module):
    def __init__(self, hidden: int, mode: str, topk_fraction: float = 0.1) -> None:
        super().__init__(); self.mode, self.topk_fraction = mode, topk_fraction
        self.patch_scorer = nn.Linear(hidden, 1)
        self.attention = nn.Linear(hidden, 1) if mode == "attention_pool" else None

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        patch_logits = self.patch_scorer(patches).squeeze(-1)
        if self.mode == "mean_patch":
            weights = torch.full_like(patch_logits, 1 / patch_logits.shape[1])
        elif self.mode == "topk_patch":
            count = topk_count(patch_logits.shape[1], self.topk_fraction)
            indices = patch_logits.topk(count, dim=1).indices
            weights = torch.zeros_like(patch_logits).scatter(1, indices, 1 / count)
        elif self.mode == "attention_pool":
            weights = torch.softmax(self.attention(patches).squeeze(-1), dim=1)
        else: raise ValueError(f"Unknown local aggregation: {self.mode}")
        representation = torch.einsum("bn,bnd->bd", weights, patches)
        logit = (weights * patch_logits).sum(dim=1)
        return logit, representation, {"patch_logits": patch_logits, "patch_weights": weights}


class GlobalLocalHead(nn.Module):
    def __init__(self, hidden: int, local_mode: str, topk_fraction: float) -> None:
        super().__init__(); self.global_scorer = nn.Linear(hidden, 1)
        self.local = LocalHead(hidden, local_mode, topk_fraction)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, global_representation, patches):
        global_logit = self.global_scorer(global_representation).squeeze(1)
        local_logit, local_representation, details = self.local(patches)
        mix = torch.sigmoid(self.mix_logit)
        details.update({"global_logit": global_logit, "local_logit": local_logit, "global_weight": mix})
        return mix * global_logit + (1 - mix) * local_logit, torch.cat((global_representation, local_representation), 1), details


class PatchDetector(nn.Module):
    def __init__(self, backbone: nn.Module, hidden: int, mode: str,
                 local_mode: str = "topk_patch", topk_fraction: float = 0.1) -> None:
        super().__init__(); self.backbone, self.mode = backbone, mode
        if mode == "global_only": self.classifier = nn.Linear(hidden, 1)
        elif mode in LOCAL_MODES: self.classifier = LocalHead(hidden, mode, topk_fraction)
        elif mode == "global_plus_local": self.classifier = GlobalLocalHead(hidden, local_mode, topk_fraction)
        else: raise ValueError(f"Unknown R6 head mode: {mode}")

    def forward(self, pixel_values, return_features=False, return_details=False):
        global_representation, patches = extract_global_and_patches(self.backbone, pixel_values)
        if self.mode == "global_only":
            logits = self.classifier(global_representation).squeeze(1)
            details = {"patch_logits": torch.empty((len(logits), 0), device=logits.device)}
            representation = global_representation
        else:
            logits, representation, details = self.classifier(global_representation, patches) if self.mode == "global_plus_local" else self.classifier(patches)
        if return_details: return logits, representation, details
        return (logits, representation) if return_features else logits


def build_patch_detector(backbone, hidden: int, state: dict, mode: str,
                         local_mode: str, topk_fraction: float) -> PatchDetector:
    model = PatchDetector(backbone, hidden, mode, local_mode, topk_fraction)
    values = state.get("state_dict", state)
    backbone_state = {key.removeprefix("backbone."): value for key, value in values.items() if key.startswith("backbone.")}
    result = model.backbone.load_state_dict(backbone_state, strict=True)
    if result.missing_keys or result.unexpected_keys: raise ValueError(f"Incompatible R4 backbone checkpoint: {result}")
    if mode == "global_only":
        model.classifier.weight.data.copy_(values["classifier.weight"])
        model.classifier.bias.data.copy_(values["classifier.bias"])
    elif mode == "global_plus_local":
        model.classifier.global_scorer.weight.data.copy_(values["classifier.weight"])
        model.classifier.global_scorer.bias.data.copy_(values["classifier.bias"])
    return model


def patch_diagnostics(model, validation, processor, config, context, output: Path) -> None:
    rows = []
    model.eval()
    for condition in DIAGNOSTIC_CONDITIONS:
        dataset = R1Dataset(validation, processor, config.input_resolution, config.seed, False, condition)
        loader = DataLoader(dataset, batch_size=int(config.training.get("validation_batch_size", 16)),
                            shuffle=False, num_workers=config.dataloader_workers, collate_fn=_collate)
        by_label = {0: [], 1: []}
        with torch.no_grad():
            for pixels, labels, _ in loader:
                with fp16_autocast(context): _, _, details = model(pixels.to(context.device), return_details=True)
                patch_logits = details["patch_logits"].float().cpu()
                if not patch_logits.shape[1]: continue
                probabilities = torch.sigmoid(patch_logits)
                count = topk_count(patch_logits.shape[1], float(config.training.get("topk_fraction", 0.1)))
                concentration = probabilities.topk(count, dim=1).values.sum(1) / probabilities.sum(1).clamp_min(1e-8)
                for index, label in enumerate(labels.int().tolist()):
                    by_label[label].append((float(patch_logits[index].mean()), float(patch_logits[index].std(unbiased=False)),
                                            float(patch_logits[index].max()), float(concentration[index])))
        for label, values in by_label.items():
            array = np.asarray(values, dtype=float)
            if len(array): rows.append({"condition": condition, "label": "fake" if label else "real", "sample_count": len(array),
                                        "mean_patch_logit": float(array[:, 0].mean()), "mean_patch_std": float(array[:, 1].mean()),
                                        "mean_max_patch_logit": float(array[:, 2].mean()), "mean_topk_concentration": float(array[:, 3].mean())})
    with (output.parent / "patch_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("condition", "label", "sample_count", "mean_patch_logit", "mean_patch_std", "mean_max_patch_logit", "mean_topk_concentration")
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def load_r4_candidate(recommendation: Path, output: Path) -> tuple[dict, Path]:
    candidate, checkpoint = load_r3_candidate(recommendation, output)
    if "bias_policy" not in candidate: raise ValueError("R4 candidate lacks its selected bias policy")
    return candidate, checkpoint


def select_candidate(summaries: list[dict], baseline_clean: float, local_only=False) -> dict:
    rows = []
    for summary in summaries:
        if summary.get("selection_split") != "validation" or summary.get("final_test_evaluated") is not False:
            raise ValueError("R6 selection is validation-only")
        rows.extend(summary["results"])
    if local_only: rows = [row for row in rows if row.get("head_mode") in LOCAL_MODES]
    ranked = rank_candidates(rows, baseline_clean, effective_tie=0.002)
    winner = next((row for row in ranked if row["validation_rank"] == 1), None)
    if winner is None: raise ValueError("No clean-eligible R6 candidate")
    return {"selected_head_mode": winner["head_mode"], "selected_local_mode": winner.get("local_mode", winner["head_mode"]),
            "selection_split": "validation", "final_test_evaluated": False, "candidate_id": winner["candidate_id"]}


def run(config_path: Path, manifest_path: Path, r4_recommendation: Path, r4_output: Path,
        output: Path, selection_config: Path | None = None) -> None:
    config = load_config(config_path); require_validation_selection(config.selection_split)
    candidate, checkpoint = load_r4_candidate(r4_recommendation, r4_output)
    config = replace(config, backbone=candidate["model_backbone"])
    mode = str(config.training["head_mode"]); local_mode = str(config.training.get("local_mode", "topk_patch"))
    if selection_config:
        selected = json.loads(selection_config.read_text(encoding="utf-8"))
        if selected.get("selection_split") != "validation" or selected.get("final_test_evaluated") is not False:
            raise ValueError("R6 selection config is not validation-only")
        if mode == "global_plus_local": local_mode = selected["selected_local_mode"]
        elif config.training.get("promotion"): mode = selected["selected_head_mode"]; local_mode = selected["selected_local_mode"]
    records = load_manifest(manifest_path); validate_no_split_leakage(records)
    selected_records, distribution = select_training_records(records, int(config.training["max_train_examples"]),
                                                             config.seed, candidate["bias_policy"])
    asset = validate_offline_asset_path(config.model.get("asset_paths", {}).get(config.backbone, ""), BACKBONES[config.backbone]["optional"])
    if asset is None: raise FileNotFoundError("R4 champion backbone assets are not attached")
    setting = candidate.get("consistency_setting", "baseline")
    if setting not in CONSISTENCY_CONFIGS: raise ValueError(f"Unknown inherited consistency setting: {setting}")
    context = resolve_distributed(); initialize_process_group(context, config.distributed.backend); seed_everything(config.seed, context.rank)
    factory = lambda backbone, hidden, state: build_patch_detector(backbone, hidden, state, mode, local_mode,
                                                                   float(config.training.get("topk_fraction", 0.1)))
    metrics = train_paired(config, records, asset, checkpoint, candidate, setting, output / "candidate", context,
                           selected_records=selected_records, training_distribution=distribution, experiment="R6",
                           extra_metadata={"candidate_id": f"r6:{mode}:{local_mode}", "head_mode": mode,
                                           "local_mode": local_mode if mode == "global_plus_local" else None,
                                           "patch_tokens_cached": False, "target_or_source_model_inputs": False},
                           model_factory=factory, model_diagnostics_callback=patch_diagnostics, calibrate_logits=True)
    if context.is_primary:
        ranked = rank_candidates([metrics], config.baseline_clean_balanced_accuracy, effective_tie=0.002)
        winner = ranked[0] if ranked[0]["validation_rank"] else None
        atomic_json(output / "r6_summary.json", {"experiment": "R6", "selection_split": "validation",
                    "final_test_evaluated": False, "results": ranked})
        fields = sorted({key for row in ranked for key, value in row.items() if not isinstance(value, (dict, list))})
        with (output / "r6_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(
                {key: row.get(key) for key in fields} for row in ranked)
        recommended = None if winner is None else {**winner, "checkpoint_relative_path": "candidate/best_model.pt"}
        atomic_json(output / "recommended_candidate.json", {"experiment": "R6", "candidate": recommended,
                    "selection_split": "validation", "final_test_evaluated": False})


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-3 R6 patch-token local forensic heads")
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--r4-recommendation", type=Path, required=True); parser.add_argument("--r4-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--selection-config", type=Path)
    args = parser.parse_args(); run(args.config, args.manifest, args.r4_recommendation, args.r4_output,
                                    args.output, args.selection_config)


if __name__ == "__main__": main()
