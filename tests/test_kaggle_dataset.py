from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.kaggle_dataset import locate_dataset_root, validate_dataset_root
from aigc_detector.shortcut_audit import _leakage_report


class KaggleDatasetTest(unittest.TestCase):
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
