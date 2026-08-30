import csv
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aigc_detector.data import ExactSeverityTransform, SEVERITY_SPECS, DeterministicTransform, RobustTransform, load_labeled_paths, load_split_manifest, severity_key, stratified_split, stratified_train_val_test_split
from aigc_detector.metrics import expected_calibration_error, operational_thresholds
from aigc_detector.model import ExpertMixtureHead, FrozenEncoders, FusionHead, ModelConfig, image_quality_statistics


class CoreTests(unittest.TestCase):
    def test_every_robust_transform_preserves_rgb_shape(self) -> None:
        image = Image.fromarray(np.full((40, 60, 3), 127, dtype=np.uint8), "RGB")
        random.seed(1)
        np.random.seed(1)
        for mode in RobustTransform.names:
            transformed = RobustTransform(mode)(image)
            self.assertEqual(transformed.mode, "RGB")
            self.assertEqual(transformed.size, image.size)
        transformed = RobustTransform("random", max_ops=3)(image)
        self.assertEqual(transformed.mode, "RGB")
        self.assertEqual(transformed.size, image.size)

    def test_deterministic_transform_is_reproducible_and_restores_rng(self) -> None:
        image = Image.fromarray(np.arange(40 * 60 * 3, dtype=np.uint8).reshape(40, 60, 3), "RGB")
        transform = DeterministicTransform("noise+jpeg", 42, "/example/image.png", 1)
        random.seed(99)
        np.random.seed(99)
        expected_python, expected_numpy = random.random(), np.random.random()
        random.seed(99)
        np.random.seed(99)
        first = np.asarray(transform(image))
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.random(), expected_numpy)
        self.assertTrue(np.array_equal(first, np.asarray(transform(image))))

    def test_exact_severity_matrix_is_deterministic_and_shape_preserving(self) -> None:
        image = Image.fromarray(np.arange(40 * 60 * 3, dtype=np.uint8).reshape(40, 60, 3), "RGB")
        self.assertEqual(len(SEVERITY_SPECS), 16)
        self.assertEqual(len({severity_key(*spec) for spec in SEVERITY_SPECS}), 16)
        for operation, value in SEVERITY_SPECS:
            transform = ExactSeverityTransform(operation, value, seed=42, key="example")
            first, second = transform(image), transform(image)
            self.assertEqual(first.mode, "RGB")
            self.assertEqual(first.size, image.size)
            self.assertTrue(np.array_equal(np.asarray(first), np.asarray(second)))

    def test_calibration_metrics_and_thresholds(self) -> None:
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        probabilities = torch.tensor([0.1, 0.2, 0.8, 0.9])
        self.assertLess(expected_calibration_error(labels, probabilities, bins=5), 0.2)
        thresholds = operational_thresholds(labels, probabilities, recall_target=1.0, precision_target=1.0)
        self.assertGreaterEqual(thresholds["high_recall"]["recall"], 1.0)
        self.assertGreaterEqual(thresholds["high_precision"]["precision"], 1.0)

    def test_folder_loading_and_split_are_balanced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("real", "ai"):
                (root / folder).mkdir()
                for index in range(10):
                    Image.new("RGB", (8, 8)).save(root / folder / f"{index}.png")
            rows = load_labeled_paths(root)
            train, val = stratified_split(rows, 0.2, 42)
            self.assertEqual((len(train), len(val)), (16, 4))
            self.assertEqual(sum(label for _, label in val), 2)

    def test_fusion_head_shape(self) -> None:
        config = ModelConfig()
        output = FusionHead(config)(torch.zeros(3, config.clip_dim + config.forensic_dim))
        self.assertEqual(tuple(output.shape), (3,))
        combined = ModelConfig(forensic_mode="laplacian_fft", forensic_dim=2560)
        output = FusionHead(combined)(torch.zeros(3, combined.clip_dim + combined.forensic_dim))
        self.assertEqual(tuple(output.shape), (3,))

    def test_fft_forensic_view_is_finite_and_normalized_shape(self) -> None:
        image_batch = torch.rand(2, 3, 32, 48)
        view = FrozenEncoders._fft_tensor(image_batch, torch.device("cpu"))
        self.assertEqual(tuple(view.shape), (2, 3, 32, 48))
        self.assertTrue(torch.isfinite(view).all())

    def test_expert_mixture_shapes_and_gate_prior(self) -> None:
        config = ModelConfig(forensic_mode="laplacian_fft", forensic_dim=2560, head_type="mixture")
        head = ExpertMixtureHead(config).eval()
        features = torch.zeros(4, 3072)
        self.assertEqual(tuple(head(features).shape), (4,))
        self.assertTrue(torch.allclose(head.gate_weights(features), torch.full((4,), 0.2), atol=1e-5))
        fixed = ExpertMixtureHead(ModelConfig(
            forensic_mode="laplacian_fft", forensic_dim=2560, head_type="mixture", gate_mode="fixed"
        ))
        self.assertTrue(torch.equal(fixed.gate_weights(features), torch.full((4,), 0.5)))

    def test_quality_statistics_are_finite(self) -> None:
        stats = image_quality_statistics([Image.new("RGB", (32, 48), (127, 127, 127))])
        self.assertEqual(tuple(stats.shape), (1, 6))
        self.assertTrue(torch.isfinite(stats).all())

    def test_source_stratified_three_way_split_is_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label_folder in ("real", "ai"):
                for source in ("one", "two"):
                    folder = root / label_folder / source
                    folder.mkdir(parents=True)
                    for index in range(10):
                        Image.new("RGB", (8, 8)).save(folder / f"{index}.png")
            rows = load_labeled_paths(root)
            train, val, test = stratified_train_val_test_split(rows, root, 0.2, 0.2, 42)
            self.assertEqual((len(train), len(val), len(test)), (24, 8, 8))
            path_sets = [{path for path, _ in split} for split in (train, val, test)]
            self.assertFalse(path_sets[0] & path_sets[1] or path_sets[0] & path_sets[2] or path_sets[1] & path_sets[2])

    def test_load_split_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for index, split in enumerate(("train", "model_selection", "calibration", "test")):
                path = root / f"{index}.png"
                Image.new("RGB", (8, 8)).save(path)
                rows.append({"path": path.name, "label": index % 2, "split": split})
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("path", "label", "split"))
                writer.writeheader()
                writer.writerows(rows)
            splits = load_split_manifest(root, manifest)
            self.assertEqual(set(splits), {"train", "model_selection", "calibration", "test"})


if __name__ == "__main__":
    unittest.main()
