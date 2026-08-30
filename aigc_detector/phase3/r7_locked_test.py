from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aigc_detector.data import ROBUSTNESS_CONDITIONS
from aigc_detector.metrics import classification_metrics

from .artifacts import atomic_json
from .r1 import BACKBONES, R1Dataset, VisionDetector, _collate, load_offline_model, validate_offline_asset_path
from .r6 import PatchDetector

ASSET_PATHS = {"dinov3_vitl16": "/kaggle/input/track5-dinov3-vitl16-lvd1689m",
               "siglip2_large_256": "/kaggle/input/track5-siglip2-large-256",
               "siglip2_so400m_256": "/kaggle/input/track5-siglip2-so400m-256"}


@dataclass(frozen=True)
class TestRecord:
    image_path: str
    label: int
    base_id: str | None = None


def validate_lock(document: dict) -> None:
    required = {"contract_version", "selection_split", "final_test_evaluated", "search_permitted",
                "components", "ensemble_weights", "decision_threshold", "total_deployment_parameter_count"}
    if required - set(document): raise ValueError(f"Invalid lock; missing {sorted(required - set(document))}")
    if document["selection_split"] != "validation" or document["final_test_evaluated"] is not False:
        raise ValueError("Lock was not produced by validation-only selection")
    if document["search_permitted"] is not False: raise ValueError("Locked final evaluation forbids search")
    components, weights = document["components"], document["ensemble_weights"]
    if not 1 <= len(components) <= 3 or len(weights) != len(components): raise ValueError("Lock must contain one to three aligned components")
    if any(value < 0 for value in weights) or abs(sum(weights) - 1) > 1e-8: raise ValueError("Invalid locked convex weights")
    if int(document["total_deployment_parameter_count"]) >= 2_000_000_000: raise ValueError("Locked candidate violates <2B")
    for component in components:
        checkpoint = Path(component["checkpoint"])
        if not checkpoint.is_absolute() or not checkpoint.is_relative_to(Path("/kaggle/input")):
            raise ValueError("Locked checkpoint must be an exact attached /kaggle/input path")
        if component["ensemble_weight"] < 0 or component["total_temperature"] <= 0:
            raise ValueError("Invalid component calibration or weight")


def load_test_manifest(path: Path) -> list[TestRecord]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = json.loads(line); split = row.get("split", row.get("original_split"))
        if split != "test": raise ValueError("Locked final evaluator accepts only explicit test records")
        records.append(TestRecord(row.get("image_path", row.get("path")), int(row["label"]), row.get("base_id") or row.get("unique_id")))
    if not records: raise ValueError("Final-test manifest is empty")
    return records


def load_component(instruction: dict, device: torch.device):
    backbone_name = instruction["backbone"]
    asset_path = Path(ASSET_PATHS[backbone_name])
    explicit = instruction.get("asset_path")
    if explicit: asset_path = Path(explicit)
    asset = validate_offline_asset_path(asset_path, BACKBONES[backbone_name]["optional"])
    if asset is None: raise FileNotFoundError(f"Missing locked backbone asset: {asset_path}")
    backbone, processor, hidden = load_offline_model(asset)
    if instruction["detector_type"] == "patch":
        model = PatchDetector(backbone, hidden, instruction["head_mode"], instruction.get("local_mode") or "topk_patch",
                              float(instruction.get("topk_fraction", .1)))
    elif instruction["detector_type"] == "global": model = VisionDetector(backbone, hidden)
    else: raise ValueError("Unknown locked detector type")
    state = torch.load(instruction["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("state_dict", state), strict=True)
    return model.to(device).eval(), processor


def infer_component(model, processor, records, instruction, condition, device, workers):
    dataset = R1Dataset(records, processor, int(instruction["resolution"]), 42, False, condition)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=workers, collate_fn=_collate)
    logits, labels = [], []
    with torch.no_grad():
        for pixels, target, _ in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits.append(model(pixels.to(device)).float().cpu())
            labels.append(target)
    return torch.cat(logits) / float(instruction["total_temperature"]), torch.cat(labels)


def run(lock_path: Path, manifest_path: Path, output: Path, workers: int) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8")); validate_lock(lock)
    records = load_test_manifest(manifest_path); device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaded = [load_component(instruction, device) for instruction in lock["components"]]
    condition_logits, labels = {}, None
    for condition in ROBUSTNESS_CONDITIONS:
        blended = None
        for instruction, (model, processor) in zip(lock["components"], loaded, strict=True):
            current, current_labels = infer_component(model, processor, records, instruction, condition, device, workers)
            blended = instruction["ensemble_weight"] * current if blended is None else blended + instruction["ensemble_weight"] * current
            if labels is None: labels = current_labels
            elif not torch.equal(labels, current_labels): raise ValueError("Final-test labels changed between components/conditions")
        condition_logits[condition] = blended
    threshold = float(lock["decision_threshold"]); rows = []
    for condition in ROBUSTNESS_CONDITIONS:
        rows.append({"condition": condition, **classification_metrics(labels, torch.sigmoid(condition_logits[condition]), threshold)})
    transformed = rows[1:]; summary = {"experiment": "R7-locked-final-test", "locked_candidate_id": lock["candidate_id"],
        "selection_or_search_performed": False, "threshold": threshold,
        "clean_balanced_accuracy": rows[0]["balanced_accuracy"],
        "mean_transformed_balanced_accuracy": sum(row["balanced_accuracy"] for row in transformed)/len(transformed),
        "worst_transformed_balanced_accuracy": min(row["balanced_accuracy"] for row in transformed),
        "worst_condition": min(transformed, key=lambda row: row["balanced_accuracy"])["condition"]}
    output.mkdir(parents=True, exist_ok=True); atomic_json(output / "final_test_scorecard.json", {"summary": summary, "conditions": rows})
    with (output / "final_test_conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Locked R7 final-test evaluation; no search code is available")
    parser.add_argument("--lock", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(); run(args.lock, args.manifest, args.output, args.workers)


if __name__ == "__main__": main()
