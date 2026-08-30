from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from aigc_detector.data import load_labeled_paths
from aigc_detector.model import ModelConfig
from scripts.kaggle_dataset import locate_dataset_root, validate_dataset_root
from scripts.extract_scale_features import save_cache
from aigc_detector.shortcut_audit import _leakage_report
from aigc_detector.train import build_cache_manifest


class KaggleDatasetTest(unittest.TestCase):
    def test_scale_cache_uses_trainer_manifest_and_robust_validation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("real", "ai"):
                (root / folder).mkdir()
                Image.new("RGB", (4, 4)).save(root / folder / "0.png")
            manifest_path = root / "split_manifest.csv"
            manifest_path.write_text("path,label,split\n", encoding="utf-8")
            args = argparse.Namespace(
                data_dir=root,
                split_manifest=manifest_path,
                validation_fraction=0.15,
                test_fraction=0.15,
                seed=42,
                augmentation_policy="balanced",
                augmentation_repeats=3,
                augmentation_depth=1,
                robust_validation=True,
            )
            config = ModelConfig(forensic_mode="laplacian", forensic_dim=1280)
            experiment_manifest = build_cache_manifest(args, config, load_labeled_paths(root))
            output = root / "cache.pt"
            features = torch.zeros(2, config.clip_dim + config.forensic_dim)
            labels = torch.tensor([0.0, 1.0])
            save_cache(
                output, features, labels, features, labels, features, labels,
                features, labels, ["jpeg_q30", "jpeg_q30"], ["clean", "clean"],
                torch.arange(2), experiment_manifest, manifest_path, "laplacian",
            )
            payload = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(payload["manifest"], experiment_manifest)
            self.assertIn("robust_val_features", payload)

    def test_validate_and_locate_audited_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "nested" / "mixed_100k"
            root.mkdir(parents=True)
            rows = []
            for index, split in enumerate(("train", "model_selection", "calibration", "test")):
                image = root / f"{index}.png"
                Image.new("RGB", (4, 4)).save(image)
                rows.append({"path": image.name, "label": index % 2, "split": split})
            with (root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("path", "label", "split"))
                writer.writeheader()
                writer.writerows(rows)
            (root / "audit.json").write_text(json.dumps({"unique_records": 4}), encoding="utf-8")

            summary = validate_dataset_root(root)
            self.assertEqual(summary["split_counts"]["test"], 1)
            self.assertEqual(locate_dataset_root(parent), root)

    def test_duplicate_group_leakage_audit(self) -> None:
        base = {
            "path": "a.png", "content_sha256": "hash-a", "duplicate_group": "group-a",
            "source": "one", "label": "0", "width": "32", "height": "32",
        }
        records = {
            "train": [base],
            "model_selection": [{**base, "path": "b.png", "content_sha256": "hash-b", "duplicate_group": "group-b"}],
            "calibration": [{**base, "path": "c.png", "content_sha256": "hash-c", "duplicate_group": "group-c"}],
            "test": [{**base, "path": "d.png", "content_sha256": "hash-d", "duplicate_group": "group-d"}],
        }
        self.assertTrue(_leakage_report(records)["passed"])
        records["test"][0]["duplicate_group"] = "group-a"
        self.assertFalse(_leakage_report(records)["passed"])


if __name__ == "__main__":
    unittest.main()
