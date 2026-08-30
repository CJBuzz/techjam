from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from functools import cmp_to_key
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aigc_detector.data import ROBUSTNESS_CONDITIONS
from aigc_detector.metrics import classification_metrics, select_threshold

from .artifacts import atomic_json
from .metrics import summarize_validation


ALLOWED_EXPERIMENTS = {"R1", "R2", "R3", "R4", "R5-low", "R5-high", "R6"}
IMPORTANT_CONDITIONS = {"resize_x0.25", "noise_s0.10", "blur_s2.0", "jpeg_q30"}
PARAMETER_LIMIT = 2_000_000_000
MINIMUM_ENSEMBLE_GAIN = 0.003


@dataclass
class Candidate:
    candidate_id: str
    experiment: str
    logits: np.ndarray
    labels: np.ndarray
    metadata: dict
    reconstruction: list[dict]
    temperature: float = 1.0

    @property
    def parameters(self) -> int: return int(self.metadata["total_deployment_parameter_count"])
    @property
    def inference_multiplier(self) -> float: return float(self.metadata.get("inference_multiplier", 1))


def normalize_weights(weights: tuple[float, ...], tolerance: float = 1e-8) -> tuple[float, ...]:
    if not 1 <= len(weights) <= 3: raise ValueError("R7 permits one to three model components")
    if any(value < 0 for value in weights): raise ValueError("Ensemble weights must be non-negative")
    total = sum(weights)
    if total <= 0: raise ValueError("Ensemble weights must have positive total")
    normalized = tuple(value / total for value in weights)
    if abs(sum(normalized) - 1) > tolerance: raise ValueError("Ensemble weights do not sum to one")
    return normalized


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    values = torch.tensor(logits.reshape(-1), dtype=torch.float32)
    targets = torch.tensor(np.tile(labels, logits.shape[0]), dtype=torch.float32)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=.1, max_iter=40)
    def closure():
        optimizer.zero_grad(); loss = F.binary_cross_entropy_with_logits(values / log_temperature.exp(), targets)
        loss.backward(); return loss
    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(.05, 20))


def _candidate_files(root: Path) -> tuple[Path | None, Path | None]:
    logits = next((path for path in (root / "val_logits.npz", root / "candidate/val_logits.npz") if path.is_file()), None)
    checkpoint = next((path for path in (root / "candidate/best_model.pt", root / "best_model.pt") if path.is_file()), None)
    return logits, checkpoint


def _reconstruction(candidate: dict, checkpoint: Path) -> dict:
    mode = candidate.get("head_mode", "global_only")
    return {"checkpoint": str(checkpoint), "backbone": candidate["model_backbone"],
            "resolution": int(candidate.get("input_resolution", 256)),
            "detector_type": "patch" if candidate.get("experiment") == "R6" or "head_mode" in candidate else "global",
            "head_mode": mode, "local_mode": candidate.get("local_mode"),
            "topk_fraction": float(candidate.get("topk_fraction", .1)),
            "preprocessing": "offline_backbone_processor_then_exact_track5_condition",
            "base_temperature": float(candidate.get("temperature", 1.0))}


def discover_candidates(input_root: Path, optional_descriptors: list[Path] | None = None) -> list[Candidate]:
    candidates, seen = [], set()
    for recommendation in input_root.rglob("recommended_candidate.json"):
        try: document = json.loads(recommendation.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): continue
        experiment, candidate = document.get("experiment"), document.get("candidate")
        if experiment not in ALLOWED_EXPERIMENTS or not candidate: continue
        if document.get("selection_split") != "validation" or document.get("final_test_evaluated") is not False: continue
        if candidate.get("status", "succeeded") != "succeeded" or not candidate.get("clean_constraint_pass", False): continue
        logits_path, checkpoint = _candidate_files(recommendation.parent)
        if logits_path is None or checkpoint is None: continue
        arrays = np.load(logits_path); logits, labels = arrays["logits"], arrays["labels"]
        identity = f"{experiment}:{candidate.get('candidate_id')}:{checkpoint}"
        if identity in seen: continue
        seen.add(identity); metadata = {**candidate, "experiment": experiment}
        candidates.append(Candidate(identity, experiment, logits, labels, metadata,
                                    [_reconstruction(metadata, checkpoint)]))
    for descriptor in optional_descriptors or []:
        document = json.loads(descriptor.read_text(encoding="utf-8"))
        if document.get("selection_split") != "validation" or document.get("final_test_evaluated") is not False:
            raise ValueError("Optional candidate descriptor must be validation-only")
        arrays = np.load(descriptor.parent / document["val_logits"])
        metadata = document["candidate"]
        if metadata.get("status") != "succeeded" or not metadata.get("clean_constraint_pass"): continue
        candidates.append(Candidate(metadata["candidate_id"], document.get("experiment", "optional"),
                                    arrays["logits"], arrays["labels"], metadata, document["reconstruction"]))
    if not candidates: raise ValueError("No successful reconstructable validation candidates discovered")
    reference_shape, reference_labels = candidates[0].logits.shape, candidates[0].labels
    if reference_shape[0] != len(ROBUSTNESS_CONDITIONS): raise ValueError("Validation logits lack exact Track-5 conditions")
    for candidate in candidates:
        if candidate.logits.shape != reference_shape or not np.array_equal(candidate.labels, reference_labels):
            raise ValueError("Candidate validation predictions are not aligned")
        candidate.temperature = fit_temperature(candidate.logits, candidate.labels)
        candidate.logits = candidate.logits / candidate.temperature
    return candidates


def diversity_rows(candidates: list[Candidate]) -> list[dict]:
    rows = []
    for left, right in itertools.combinations(candidates, 2):
        repeated = torch.tensor(np.tile(left.labels, left.logits.shape[0]), dtype=torch.float32)
        left_threshold = select_threshold(repeated, torch.sigmoid(torch.tensor(left.logits.reshape(-1))), "balanced")
        right_threshold = select_threshold(repeated, torch.sigmoid(torch.tensor(right.logits.reshape(-1))), "balanced")
        for index, condition in enumerate(ROBUSTNESS_CONDITIONS):
            left_logits, right_logits = left.logits[index], right.logits[index]
            left_prob, right_prob = 1 / (1 + np.exp(-left_logits)), 1 / (1 + np.exp(-right_logits))
            left_pred, right_pred = left_prob >= left_threshold, right_prob >= right_threshold
            truth = left.labels.astype(bool); left_error, right_error = left_pred != truth, right_pred != truth
            logit_corr = float(np.corrcoef(left_logits, right_logits)[0, 1]) if np.std(left_logits) and np.std(right_logits) else None
            prob_corr = float(np.corrcoef(left_prob, right_prob)[0, 1]) if np.std(left_prob) and np.std(right_prob) else None
            rows.append({"left": left.candidate_id, "right": right.candidate_id, "condition": condition,
                         "important_condition": condition in IMPORTANT_CONDITIONS,
                         "logit_correlation": logit_corr, "probability_correlation": prob_corr,
                         "prediction_disagreement": float(np.mean(left_pred != right_pred)),
                         "error_disagreement": float(np.mean(left_error != right_error)),
                         "left_only_correct": float(np.mean(~left_error & right_error)),
                         "right_only_correct": float(np.mean(left_error & ~right_error)),
                         "both_wrong": float(np.mean(left_error & right_error))})
    return rows


def _score(components: tuple[Candidate, ...], weights: tuple[float, ...]) -> tuple[dict, np.ndarray]:
    weights = normalize_weights(weights)
    parameters = sum(candidate.parameters for candidate in components)
    if parameters >= PARAMETER_LIMIT: raise ValueError("Ensemble violates Track-5 <2B parameter limit")
    logits = sum(weight * candidate.logits for weight, candidate in zip(weights, components, strict=True))
    labels = torch.tensor(components[0].labels, dtype=torch.float32); values = torch.tensor(logits, dtype=torch.float32)
    threshold = select_threshold(labels.repeat(len(ROBUSTNESS_CONDITIONS)), torch.sigmoid(values.flatten()), "balanced")
    probabilities = {condition: torch.sigmoid(values[index]) for index, condition in enumerate(ROBUSTNESS_CONDITIONS)}
    candidate_id = "+".join(f"{weight:.2f}*{candidate.candidate_id}" for weight, candidate in zip(weights, components, strict=True))
    metadata = {"candidate_id": candidate_id, "component_count": len(components), "weights": list(weights),
                "components": [candidate.candidate_id for candidate in components], "threshold": float(threshold),
                "model_backbone": "heterogeneous_ensemble", "trainable_parameter_count": 0,
                "total_deployment_parameter_count": parameters, "input_resolution": "heterogeneous",
                "training_data_counts": {}, "inference_multiplier": sum(c.inference_multiplier for c in components),
                "status": "succeeded", "parameter_limit_pass": True}
    summary, _ = summarize_validation(labels, probabilities, threshold, metadata)
    return summary, logits


def _ordered(rows: list[dict], baseline_clean: float) -> list[dict]:
    floor = baseline_clean - .01
    eligible = [row for row in rows if row["clean_validation_balanced_accuracy"] >= floor]
    def compare(left, right):
        worst = left["worst_transformed_validation_balanced_accuracy"] - right["worst_transformed_validation_balanced_accuracy"]
        if abs(worst) > .001: return -1 if worst > 0 else 1
        mean = left["mean_transformed_validation_balanced_accuracy"] - right["mean_transformed_validation_balanced_accuracy"]
        if abs(mean) > .001: return -1 if mean > 0 else 1
        left_cost = (left["component_count"], left["inference_multiplier"])
        right_cost = (right["component_count"], right["inference_multiplier"])
        return -1 if left_cost < right_cost else (1 if left_cost > right_cost else 0)
    eligible.sort(key=cmp_to_key(compare))
    ranks = {row["candidate_id"]: index for index, row in enumerate(eligible, 1)}
    return [{**row, "clean_constraint_pass": row["clean_validation_balanced_accuracy"] >= floor,
             "validation_rank": ranks.get(row["candidate_id"]), "selection_split": "validation",
             "final_test_evaluated": False} for row in rows]


def search(candidates: list[Candidate], baseline_clean: float, maximum_candidates: int = 8) -> tuple[list[dict], dict, np.ndarray]:
    singles = []
    for candidate in candidates:
        row, logits = _score((candidate,), (1.0,)); singles.append((row, logits, candidate))
    ranked_singles = _ordered([item[0] for item in singles], baseline_clean)
    rank_map = {row["candidate_id"]: row["validation_rank"] or math.inf for row in ranked_singles}
    pool = sorted(candidates, key=lambda item: rank_map[f"1.00*{item.candidate_id}"])[:maximum_candidates]
    rows, logits_by_id = [item[0] for item in singles], {item[0]["candidate_id"]: item[1] for item in singles}
    for pair in itertools.combinations(pool, 2):
        for left_weight in np.arange(.1, 1, .1):
            row, logits = _score(pair, (float(left_weight), float(1-left_weight)))
            rows.append(row); logits_by_id[row["candidate_id"]] = logits
    for triple in itertools.combinations(pool, 3):
        for first in (.25, .5):
            for second in (.25, .5):
                third = 1 - first - second
                if third <= 0: continue
                row, logits = _score(triple, (first, second, third)); rows.append(row); logits_by_id[row["candidate_id"]] = logits
    ranked = _ordered(rows, baseline_clean); best_single = min((row for row in ranked if row["component_count"] == 1 and row["validation_rank"]), key=lambda row: row["validation_rank"])
    best = min((row for row in ranked if row["validation_rank"]), key=lambda row: row["validation_rank"])
    gain = best["worst_transformed_validation_balanced_accuracy"] - best_single["worst_transformed_validation_balanced_accuracy"]
    winner = best if best["component_count"] > 1 and gain >= MINIMUM_ENSEMBLE_GAIN else best_single
    winner = {**winner, "ensemble_gain_over_best_single": float(gain),
              "simplicity_threshold": MINIMUM_ENSEMBLE_GAIN,
              "simplicity_rule_selected_single": winner["component_count"] == 1}
    return ranked, winner, logits_by_id[winner["candidate_id"]]


def locked_document(winner: dict, candidates: list[Candidate]) -> dict:
    lookup = {candidate.candidate_id: candidate for candidate in candidates}; components = []
    for identity, weight in zip(winner["components"], winner["weights"], strict=True):
        candidate = lookup[identity]
        if len(candidate.reconstruction) != 1: raise ValueError("Nested optional ensembles must be flattened before locking")
        instruction = {**candidate.reconstruction[0], "ensemble_weight": weight,
                       "r7_temperature": candidate.temperature,
                       "total_temperature": instruction_temperature(candidate)}
        components.append(instruction)
    return {"contract_version": 1, "experiment": "R7", "selection_split": "validation",
            "final_test_evaluated": False, "search_permitted": False,
            "candidate_id": winner["candidate_id"], "components": components,
            "ensemble_weights": winner["weights"], "decision_threshold": winner["threshold"],
            "total_deployment_parameter_count": winner["total_deployment_parameter_count"],
            "inference_multiplier": winner["inference_multiplier"],
            "preprocessing": "component-specific offline processor; official condition applied before component resize",
            "calibration": "divide each raw component logit by total_temperature before convex combination"}


def instruction_temperature(candidate: Candidate) -> float:
    return float(candidate.reconstruction[0].get("base_temperature", 1.0)) * candidate.temperature


def run(input_root: Path, output: Path, baseline_clean: float, optional: list[Path]) -> None:
    candidates = discover_candidates(input_root, optional); diversity = diversity_rows(candidates)
    ranked, winner, logits = search(candidates, baseline_clean); lock = locked_document(winner, candidates)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "model_diversity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diversity[0]) if diversity else ["left", "right", "condition"])
        writer.writeheader(); writer.writerows(diversity)
    fields = sorted({key for row in ranked for key, value in row.items() if not isinstance(value, (dict, list))})
    with (output / "ensemble_search.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(
            {key: row.get(key) for key in fields} for row in ranked)
    atomic_json(output / "phase3_summary.json", {"experiment": "R7", "selection_split": "validation",
                "final_test_evaluated": False, "candidate_count": len(candidates), "results": ranked,
                "recommended": winner})
    atomic_json(output / "recommended_candidate.json", {"experiment": "R7", "candidate": winner,
                "selection_split": "validation", "final_test_evaluated": False})
    atomic_json(output / "locked_candidate.json", lock)
    np.savez_compressed(output / "val_logits.npz", logits=logits, labels=candidates[0].labels)


def main():
    parser = argparse.ArgumentParser(description="Phase-3 R7 validation-only sparse heterogeneous ensemble search")
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input")); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-clean", type=float, default=.9681); parser.add_argument("--optional-candidate", type=Path, action="append", default=[])
    args = parser.parse_args(); run(args.input_root, args.output, args.baseline_clean, args.optional_candidate)


if __name__ == "__main__": main()
