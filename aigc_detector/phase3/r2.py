from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image

from aigc_detector.data import ROBUSTNESS_CONDITIONS

from .artifacts import atomic_json
from .config import load_config, require_validation_selection
from .data import ManifestRecord, exact_track5_transform, load_manifest, manifest_counts
from .r1 import BACKBONES, train_candidate, validate_offline_asset_path
from .ranking import rank_candidates
from .runtime import initialize_process_group, resolve_distributed, seed_everything


REGIMES = ("single_transform", "compound_curriculum")
CURRICULUM_PROBABILITIES = {
    "early": {0: 0.55, 1: 0.45},
    "middle": {0: 0.15, 1: 0.55, 2: 0.30},
    "late": {0: 0.05, 1: 0.35, 2: 0.40, 3: 0.20},
}
MILD = ("jpeg_q90", "jpeg_q70", "blur_s0.5", "resize_x0.5", "noise_s0.02", "color_0.8", "color_1.2")
MODERATE = ("jpeg_q50", "blur_s1.0", "resize_x0.5", "noise_s0.05", "crop_0.8")
HEAVY = ("jpeg_q30", "blur_s2.0", "resize_x0.25", "noise_s0.10")
COMPOSITIONS = {
    1: tuple((name,) for name in (*MILD, *MODERATE, *HEAVY)),
    2: (
        ("resize_x0.5", "jpeg_q70"), ("blur_s0.5", "jpeg_q70"),
        ("resize_x0.5", "noise_s0.02"), ("color_0.8", "resize_x0.5"),
        ("crop_0.8", "resize_x0.5"), ("blur_s1.0", "jpeg_q50"),
    ),
    3: (
        ("crop_0.8", "resize_x0.5", "jpeg_q70"),
        ("color_1.2", "resize_x0.5", "jpeg_q70"),
        ("crop_0.8", "resize_x0.5", "noise_s0.05"),
    ),
}


def validate_no_split_leakage(records: list[ManifestRecord]) -> None:
    owners: dict[str, str] = {}
    for record in records:
        identity = record.unique_id or record.base_id or record.path
        previous = owners.setdefault(identity, record.split)
        if previous != record.split:
            raise ValueError(f"Manifest identity crosses train/validation: {identity}")


def _round_robin(groups: dict[str, list[ManifestRecord]], target: int, rng: random.Random,
                 per_group_cap: int | None = None) -> list[ManifestRecord]:
    for values in groups.values(): rng.shuffle(values)
    keys = sorted(groups); selected, counts = [], Counter()
    while len(selected) < target:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < target and (per_group_cap is None or counts[key] < per_group_cap):
                selected.append(groups[key].pop()); counts[key] += 1; progressed = True
        if not progressed: break
    return selected


def balanced_training_records(records: list[ManifestRecord], maximum: int, seed: int,
                              max_fake_per_generator: int | None = None) -> tuple[list[ManifestRecord], dict]:
    train = [row for row in records if row.split == "train"]
    real, fake = [row for row in train if row.label == 0], [row for row in train if row.label == 1]
    target_each = min(maximum // 2, len(real), len(fake))
    rng = random.Random(seed)
    real_groups: dict[str, list[ManifestRecord]] = defaultdict(list)
    fake_groups: dict[str, list[ManifestRecord]] = defaultdict(list)
    for row in real: real_groups[row.source or "unknown_real_source"].append(row)
    for row in fake: fake_groups[f"{row.source or 'unknown_source'}::{row.generator or 'unknown_generator'}"].append(row)
    selected_real = _round_robin(real_groups, target_each, rng)
    selected_fake = _round_robin(fake_groups, target_each, rng, max_fake_per_generator)
    balanced_each = min(len(selected_real), len(selected_fake))
    selected = selected_real[:balanced_each] + selected_fake[:balanced_each]
    rng.shuffle(selected)
    distribution = manifest_counts(selected)
    distribution.update({"requested_maximum": maximum, "effective_total": len(selected),
                         "effective_real": balanced_each, "effective_fake": balanced_each,
                         "max_fake_per_generator": max_fake_per_generator})
    return selected, distribution


def curriculum_level(progress: float) -> str:
    if not 0 <= progress <= 1: raise ValueError("Curriculum progress must be in [0, 1]")
    return "early" if progress < 1 / 3 else ("middle" if progress < 2 / 3 else "late")


def _weighted_choice(probabilities: dict[int, float], rng: random.Random) -> int:
    value, cumulative = rng.random(), 0.0
    for count, probability in sorted(probabilities.items()):
        cumulative += probability
        if value <= cumulative: return count
    return max(probabilities)


def curriculum_chain(regime: str, progress: float, seed: int, identity: str) -> tuple[str, ...]:
    if regime not in REGIMES: raise ValueError(f"Unknown R2 regime: {regime}")
    digest = hashlib.sha256(f"{seed}\0{identity}\0{progress:.8f}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    if regime == "single_transform":
        return (rng.choice(ROBUSTNESS_CONDITIONS),)
    level = curriculum_level(progress)
    count = _weighted_choice(CURRICULUM_PROBABILITIES[level], rng)
    if count == 0: return ("clean",)
    choices = COMPOSITIONS[count]
    if level == "early":
        choices = tuple(chain for chain in choices if all(item in MILD for item in chain)) or choices
    elif level == "middle":
        choices = tuple(chain for chain in choices if not any(item in HEAVY for item in chain)) or choices
    return rng.choice(choices)


class R2CurriculumDataset(torch.utils.data.Dataset):
    def __init__(self, records, processor, resolution: int, seed: int, regime: str, epochs: int) -> None:
        self.records, self.processor, self.resolution, self.seed = records, processor, resolution, seed
        self.regime, self.epochs, self.epoch = regime, max(epochs, 1), 0

    def set_epoch(self, epoch: int) -> None: self.epoch = epoch
    def __len__(self): return len(self.records)

    def chain_for(self, index: int) -> tuple[str, ...]:
        progress = self.epoch / max(self.epochs - 1, 1)
        record = self.records[index]
        return curriculum_chain(self.regime, progress, self.seed, f"{record.unique_id or record.path}:{self.epoch}")

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as source: image = source.convert("RGB")
        chain = self.chain_for(index)
        group = "+".join(item for item in chain if item != "clean") or "clean"
        image = exact_track5_transform(group.split("+")[0], self.seed, record.unique_id or record.path, self.epoch)(image) \
            if "+" not in group else self._compound(image, group, record)
        values = self.processor(images=image, size={"height": self.resolution, "width": self.resolution}, return_tensors="pt")
        return values["pixel_values"][0], torch.tensor(record.label, dtype=torch.float32), record.path

    def _compound(self, image: Image.Image, group: str, record: ManifestRecord) -> Image.Image:
        from aigc_detector.data import DeterministicTransform
        return DeterministicTransform(group, self.seed, record.unique_id or record.path, self.epoch)(image)


def planned_corruption_distribution(records: list[ManifestRecord], regime: str, epochs: int, seed: int) -> dict:
    counts = Counter()
    for epoch in range(max(epochs, 1)):
        progress = epoch / max(epochs - 1, 1)
        for record in records:
            chain = curriculum_chain(regime, progress, seed, f"{record.unique_id or record.path}:{epoch}")
            counts["+".join(chain)] += 1
            counts[f"length_{0 if chain == ('clean',) else len(chain)}"] += 1
    return {"regime": regime, "epochs": epochs, "planned_examples": len(records) * max(epochs, 1),
            "chain_counts": dict(sorted(counts.items()))}


def load_r1_candidate(recommendation_path: Path, r1_output: Path) -> tuple[dict, Path]:
    document = json.loads(recommendation_path.read_text(encoding="utf-8"))
    if document.get("selection_split") != "validation" or document.get("final_test_evaluated") is not False:
        raise ValueError("R1 recommendation is not validation-only")
    candidate = document.get("candidate")
    if not candidate or not candidate.get("clean_constraint_pass"):
        raise ValueError("R1 has no clean-eligible recommended candidate")
    relative = candidate.get("checkpoint_relative_path")
    checkpoint = r1_output / relative if relative else Path(candidate.get("checkpoint", ""))
    if not checkpoint.is_file(): raise FileNotFoundError(f"R1 checkpoint is missing: {checkpoint}")
    return candidate, checkpoint


def discover_r1_output(input_root: Path = Path("/kaggle/input")) -> Path:
    matches = []
    for path in input_root.glob("*/recommended_candidate.json"):
        try: document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): continue
        if document.get("experiment") == "R1" and document.get("selection_split") == "validation":
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one attached R1 winning kernel output, found {len(matches)}")
    return matches[0]


def select_promotion_regime(single: dict, compound: dict, baseline_clean: float) -> dict:
    for document in (single, compound):
        if document.get("selection_split") != "validation" or document.get("final_test_evaluated") is not False:
            raise ValueError("R2 promotion selection is validation-only")
    rows = [single["eligible_winner"], compound["eligible_winner"]]
    rows = [row for row in rows if row is not None]
    ranked = rank_candidates(rows, baseline_clean, effective_tie=0.002)
    winner = min((row for row in ranked if row["validation_rank"] is not None),
                 key=lambda row: row["validation_rank"], default=None)
    if winner is None: raise ValueError("No eligible R2-25k regime can be promoted")
    return {"selected_regime": winner["regime"], "candidate_id": winner["candidate_id"],
            "selection_split": "validation", "final_test_evaluated": False,
            "reason": "Best eligible 25k validation result under worst/mean/cost policy."}


def write_r2_summary(rows: list[dict], output: Path, baseline_clean: float, distribution: dict,
                     corruption: dict) -> None:
    ranked = rank_candidates(rows, baseline_clean, effective_tie=0.002)
    eligible = [row for row in ranked if row["validation_rank"] is not None]
    winner = min(eligible, key=lambda row: row["validation_rank"]) if eligible else None
    output.mkdir(parents=True, exist_ok=True)
    document = {"experiment": "R2", "selection_split": "validation", "final_test_evaluated": False,
                "eligible_winner": winner, "results": ranked}
    atomic_json(output / "r2_summary.json", document)
    with (output / "r2_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in ranked for key, value in row.items() if not isinstance(value, (dict, list))})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in ranked)
    atomic_json(output / "training_distribution.json", distribution)
    atomic_json(output / "corruption_distribution.json", corruption)
    recommended = None if winner is None else {**winner, "checkpoint_relative_path": "candidate/best_model.pt"}
    atomic_json(output / "recommended_candidate.json", {"experiment": "R2", "candidate": recommended,
                                                          "selection_split": "validation", "final_test_evaluated": False})


def run(config_path: Path, manifest_path: Path, r1_recommendation: Path, r1_output: Path,
        output: Path, promotion_config: Path | None = None) -> None:
    config = load_config(config_path); require_validation_selection(config.selection_split)
    candidate, checkpoint = load_r1_candidate(r1_recommendation, r1_output)
    backbone_name = candidate["model_backbone"]
    if backbone_name not in BACKBONES: raise ValueError(f"R1 recommended unsupported backbone: {backbone_name}")
    config = replace(config, backbone=backbone_name)
    regime = str(config.training["regime"])
    if promotion_config:
        promotion = json.loads(promotion_config.read_text(encoding="utf-8"))
        if promotion.get("selection_split") != "validation" or promotion.get("final_test_evaluated") is not False:
            raise ValueError("Promotion config is not validation-only")
        regime = promotion["selected_regime"]
    records = load_manifest(manifest_path); validate_no_split_leakage(records)
    maximum = int(config.training.get("max_train_examples", 25000))
    selected, distribution = balanced_training_records(
        records, maximum, config.seed, config.training.get("max_fake_per_generator")
    )
    validation = [row for row in records if row.split == "validation"]
    selected_records = selected + validation
    corruption = planned_corruption_distribution(selected, regime, int(config.training.get("epochs", 2)), config.seed)
    asset_paths = config.model.get("asset_paths", {})
    asset = validate_offline_asset_path(asset_paths.get(backbone_name, ""), BACKBONES[backbone_name]["optional"])
    if asset is None: raise FileNotFoundError("Winning R1 backbone asset is not attached to R2")
    context = resolve_distributed(); initialize_process_group(context, config.distributed.backend); seed_everything(config.seed, context.rank)
    holder = {}
    def factory(train_records, processor, current_config):
        dataset = R2CurriculumDataset(train_records, processor, current_config.input_resolution,
                                      current_config.seed, regime, int(current_config.training.get("epochs", 2)))
        holder["dataset"] = dataset; return dataset
    metrics = train_candidate(config, selected_records, asset, candidate["training_mode"],
                              output / "candidate", context, checkpoint, factory)
    if context.is_primary:
        metrics["regime"] = regime; metrics["candidate_id"] = f"{backbone_name}:{candidate['training_mode']}:{regime}:{maximum}"
        write_r2_summary([metrics], output, config.baseline_clean_balanced_accuracy, distribution, corruption)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-3 R2 balanced compound-corruption curriculum")
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--r1-recommendation", type=Path, required=True); parser.add_argument("--r1-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--promotion-config", type=Path)
    args = parser.parse_args(); run(args.config, args.manifest, args.r1_recommendation, args.r1_output,
                                    args.output, args.promotion_config)


if __name__ == "__main__": main()
