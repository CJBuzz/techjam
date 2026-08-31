from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .artifacts import atomic_json
from .config import load_config, require_validation_selection
from .data import ManifestRecord, load_manifest, manifest_counts
from .r1 import BACKBONES, validate_offline_asset_path
from .r2 import _round_robin, validate_no_split_leakage
from .r3 import CONSISTENCY_CONFIGS, discover_r2_output, train_paired
from .ranking import rank_candidates
from .runtime import initialize_process_group, resolve_distributed, seed_everything


POLICIES = ("class_balanced", "source_balanced", "source_quality_matched")
QUALITY_FIELDS = ("area_bin", "aspect_ratio_bin", "file_format", "sharpness_bin", "compression_proxy_bin")


def area_bin(record: ManifestRecord) -> str:
    if not record.width or not record.height:
        return str(record.metadata.get("area_bin", "unknown"))
    area = record.width * record.height
    return "small" if area < 256**2 else ("medium" if area < 768**2 else "large")


def aspect_ratio_bin(record: ManifestRecord) -> str:
    if not record.width or not record.height:
        return str(record.metadata.get("aspect_ratio_bin", "unknown"))
    ratio = record.width / max(record.height, 1)
    return "portrait" if ratio < 0.8 else ("landscape" if ratio > 1.25 else "squareish")


def quality_metadata(record: ManifestRecord) -> dict[str, str]:
    """Sampling-only metadata. This dictionary is never returned by a model dataset."""
    extension = Path(record.path).suffix.lower().lstrip(".") or "unknown"
    return {
        "area_bin": area_bin(record),
        "aspect_ratio_bin": aspect_ratio_bin(record),
        "file_format": str(record.metadata.get("format", extension)).lower(),
        "sharpness_bin": str(record.metadata.get("sharpness_bin", "unknown")),
        "compression_proxy_bin": str(record.metadata.get("compression_proxy_bin", "unknown")),
    }


def quality_key(record: ManifestRecord) -> tuple[str, ...]:
    values = quality_metadata(record)
    return tuple(values[name] for name in QUALITY_FIELDS)


def _class_balanced(records: list[ManifestRecord], maximum: int, rng: random.Random) -> list[ManifestRecord]:
    train = [row for row in records if row.split == "train"]
    real = [row for row in train if row.label == 0]; fake = [row for row in train if row.label == 1]
    each = min(maximum // 2, len(real), len(fake)); rng.shuffle(real); rng.shuffle(fake)
    selected = real[:each] + fake[:each]; rng.shuffle(selected)
    return selected


def _source_balanced(records: list[ManifestRecord], maximum: int, rng: random.Random) -> list[ManifestRecord]:
    train = [row for row in records if row.split == "train"]
    target = min(maximum // 2, sum(row.label == 0 for row in train), sum(row.label == 1 for row in train))
    groups = ({}, {})
    for label in (0, 1):
        grouped: dict[str, list[ManifestRecord]] = defaultdict(list)
        for row in train:
            if row.label == label:
                identity = row.source or "unknown_source"
                if label: identity += f"::{row.generator or 'unknown_generator'}"
                grouped[identity].append(row)
        groups[label].update(grouped)
    selected = _round_robin(groups[0], target, rng) + _round_robin(groups[1], target, rng)
    rng.shuffle(selected); return selected


def _quality_matched(records: list[ManifestRecord], maximum: int, rng: random.Random) -> list[ManifestRecord]:
    train = [row for row in records if row.split == "train"]
    by_quality: dict[tuple[str, ...], dict[int, list[ManifestRecord]]] = defaultdict(lambda: {0: [], 1: []})
    for row in train: by_quality[quality_key(row)][row.label].append(row)
    strata = [key for key, values in by_quality.items() if values[0] and values[1]]
    for key in strata:
        for label in (0, 1):
            # Round-robin source/generator before cross-class stratum matching.
            grouped: dict[str, list[ManifestRecord]] = defaultdict(list)
            for row in by_quality[key][label]:
                group = row.source or "unknown_source"
                if label: group += f"::{row.generator or 'unknown_generator'}"
                grouped[group].append(row)
            by_quality[key][label] = _round_robin(grouped, sum(map(len, grouped.values())), rng)
    target = min(maximum // 2, sum(row.label == 0 for row in train), sum(row.label == 1 for row in train))
    if target and not strata:
        raise ValueError("No shared real/fake quality stratum exists for source_quality_matched policy")
    selected = {0: [], 1: []}; offsets = Counter()
    while len(selected[0]) < target:
        progressed = False
        for key in sorted(strata):
            index = offsets[key]
            if len(selected[0]) < target:
                # Cycle the smaller side within a stratum. This matches the quality
                # distribution without duplicating image tensors or changing budget.
                selected[0].append(by_quality[key][0][index % len(by_quality[key][0])])
                selected[1].append(by_quality[key][1][index % len(by_quality[key][1])])
                offsets[key] += 1; progressed = True
        if not progressed: break
    result = selected[0] + selected[1]; rng.shuffle(result); return result


def select_training_records(records: list[ManifestRecord], maximum: int, seed: int, policy: str) -> tuple[list[ManifestRecord], dict]:
    if policy not in POLICIES: raise ValueError(f"Unknown R4 policy: {policy}")
    rng = random.Random(seed)
    if policy == "class_balanced": selected = _class_balanced(records, maximum, rng)
    elif policy == "source_balanced": selected = _source_balanced(records, maximum, rng)
    else: selected = _quality_matched(records, maximum, rng)
    counts = manifest_counts(selected)
    counts.update({"policy": policy, "requested_maximum": maximum, "effective_total": len(selected),
                   "canonical_preprocessing": True, "quality_features_are_model_inputs": False})
    return selected, counts


def balance_report(records: list[ManifestRecord]) -> list[dict]:
    dimensions: dict[tuple[str, str], int] = Counter()
    for row in records:
        label = "fake" if row.label else "real"
        dimensions[("class", label)] += 1
        dimensions[("source", row.source or "unknown")] += 1
        if row.label: dimensions[("fake_generator", row.generator or "unknown")] += 1
        for name, value in quality_metadata(row).items(): dimensions[(name, value)] += 1
    return [{"dimension": key[0], "value": key[1], "count": count} for key, count in sorted(dimensions.items())]


def source_leakage_diagnostic(records: list[ManifestRecord], embeddings: torch.Tensor) -> dict:
    """Deterministic nearest-centroid source probe; diagnostic only, never detector input."""
    sources = np.asarray([row.source or "unknown" for row in records])
    values = embeddings.detach().float().cpu().numpy()
    train_indices, eval_indices = [], []
    for source in sorted(set(sources)):
        indices = np.flatnonzero(sources == source)
        train_indices.extend(indices[::2]); eval_indices.extend(indices[1::2])
    if not eval_indices or len(set(sources[train_indices])) < 2:
        return {"status": "insufficient_sources", "source_probe_accuracy": None, "sample_count": len(records),
                "used_as_detector_input": False}
    centroids = {source: values[[i for i in train_indices if sources[i] == source]].mean(0)
                 for source in sorted(set(sources[train_indices]))}
    predictions = [min(centroids, key=lambda source: float(np.square(values[i] - centroids[source]).sum()))
                   for i in eval_indices]
    truth = sources[eval_indices]
    majority = max(Counter(truth).values()) / len(truth)
    return {"status": "succeeded", "method": "deterministic_nearest_centroid_on_clean_validation_embeddings",
            "source_probe_accuracy": float(np.mean(np.asarray(predictions) == truth)),
            "majority_baseline_accuracy": float(majority), "sample_count": len(eval_indices),
            "source_count": len(centroids), "used_as_detector_input": False}


def load_r3_candidate(recommendation: Path, output: Path) -> tuple[dict, Path]:
    document = json.loads(recommendation.read_text(encoding="utf-8"))
    if document.get("selection_split") != "validation" or document.get("final_test_evaluated") is not False:
        raise ValueError("R3 recommendation is not validation-only")
    candidate = document.get("candidate")
    if not candidate or not candidate.get("clean_constraint_pass"): raise ValueError("R3 has no eligible champion")
    checkpoint = output / candidate.get("checkpoint_relative_path", "candidate/best_model.pt")
    if not checkpoint.is_file(): raise FileNotFoundError(f"R3 checkpoint missing: {checkpoint}")
    return candidate, checkpoint


def discover_r3_output(input_root: Path = Path("/kaggle/input")) -> Path:
    matches = []
    for path in input_root.glob("*/recommended_candidate.json"):
        try: document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): continue
        if document.get("experiment") == "R3" and document.get("selection_split") == "validation": matches.append(path.parent)
    if len(matches) != 1: raise ValueError(f"Expected exactly one attached R3 champion output, found {len(matches)}")
    return matches[0]


def write_summary(rows: list[dict], output: Path, baseline_clean: float, policy: str) -> None:
    ranked = rank_candidates(rows, baseline_clean, effective_tie=0.002)
    winner = next((row for row in ranked if row["validation_rank"] == 1), None)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "bias_policy_summary.json", {"experiment": "R4", "selection_split": "validation",
                "final_test_evaluated": False, "results": ranked, "eligible_winner": winner})
    fields = sorted({key for row in ranked for key, value in row.items() if not isinstance(value, (dict, list))})
    with (output / "bias_policy_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(
            {key: row.get(key) for key in fields} for row in ranked)
    recommended = None if winner is None else {**winner, "checkpoint_relative_path": "candidate/best_model.pt"}
    atomic_json(output / "recommended_candidate.json", {"experiment": "R4", "candidate": recommended,
                "selection_split": "validation", "final_test_evaluated": False})


def select_promotion_policy(summaries: list[dict], baseline_clean: float) -> dict:
    rows = []
    for summary in summaries:
        if summary.get("selection_split") != "validation" or summary.get("final_test_evaluated") is not False:
            raise ValueError("R4 promotion selection is validation-only")
        rows.extend(summary["results"])
    ranked = rank_candidates(rows, baseline_clean, effective_tie=0.002)
    winner = next((row for row in ranked if row["validation_rank"] == 1), None)
    if winner is None: raise ValueError("No clean-eligible R4 policy")
    return {"selected_policy": winner["bias_policy"], "selection_split": "validation",
            "final_test_evaluated": False, "candidate_id": winner["candidate_id"]}


def run(config_path: Path, manifest_path: Path, r3_recommendation: Path, r3_output: Path,
        output: Path, promotion_config: Path | None = None) -> None:
    config = load_config(config_path); require_validation_selection(config.selection_split)
    candidate, checkpoint = load_r3_candidate(r3_recommendation, r3_output)
    config = replace(config, backbone=candidate["model_backbone"])
    policy = str(config.training["bias_policy"])
    if promotion_config:
        promotion = json.loads(promotion_config.read_text(encoding="utf-8"))
        if promotion.get("selection_split") != "validation" or promotion.get("final_test_evaluated") is not False:
            raise ValueError("R4 promotion config is not validation-only")
        policy = promotion["selected_policy"]
    records = load_manifest(manifest_path); validate_no_split_leakage(records)
    selected, distribution = select_training_records(records, int(config.training["max_train_examples"]), config.seed, policy)
    report = balance_report(selected); output.mkdir(parents=True, exist_ok=True)
    with (output / "training_balance_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dimension", "value", "count")); writer.writeheader(); writer.writerows(report)
    asset = validate_offline_asset_path(config.model.get("asset_paths", {}).get(config.backbone, ""),
                                        BACKBONES[config.backbone]["optional"])
    if asset is None: raise FileNotFoundError("R3 champion backbone assets are not attached")
    context = resolve_distributed(); initialize_process_group(context, config.distributed.backend); seed_everything(config.seed, context.rank)
    def diagnostic(validation, features, candidate_output):
        result = source_leakage_diagnostic(validation, features["clean"])
        atomic_json(output / "source_leakage_diagnostic.json", result)
    setting = candidate.get("consistency_setting", "baseline")
    if setting not in CONSISTENCY_CONFIGS: raise ValueError(f"Unknown inherited R3 consistency setting: {setting}")
    metrics = train_paired(config, records, asset, checkpoint, candidate, setting, output / "candidate", context,
                           selected_records=selected, training_distribution=distribution, experiment="R4",
                           extra_metadata={"bias_policy": policy, "candidate_id": f"r4:{policy}",
                                           "quality_features_are_model_inputs": False,
                                           "canonical_preprocessing": True},
                           post_validation_callback=diagnostic)
    if context.is_primary: write_summary([metrics], output, config.baseline_clean_balanced_accuracy, policy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-3 R4 bias-controlled source/quality matching")
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--r3-recommendation", type=Path, required=True); parser.add_argument("--r3-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--promotion-config", type=Path)
    args = parser.parse_args(); run(args.config, args.manifest, args.r3_recommendation, args.r3_output,
                                    args.output, args.promotion_config)


if __name__ == "__main__": main()
