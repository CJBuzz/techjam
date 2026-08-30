import random
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aigc_detector.data import (
    ROBUSTNESS_CONDITIONS,
    DeterministicTransform,
    RobustTransform,
    image_source,
    load_labeled_paths,
    stratified_split,
    stratified_train_val_test_split,
)
from aigc_detector.evaluate import paired_generator_metrics, robustness_scorecard, write_condition_csv
from aigc_detector.metrics import classification_metrics
from scripts.prepare_bfree_new_generators import parse_checksums, prepare_dataset
from aigc_detector.model import (
    AdaptiveTriExpertHead,
    ExpertMixtureHead,
    FrozenEncoders,
    FusionHead,
    ModelConfig,
    image_quality_statistics,
)


class CoreTests(unittest.TestCase):
    def test_e0_scorecard_metrics_and_csv(self) -> None:
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
        clean = classification_metrics(labels, torch.tensor([0.1, 0.2, 0.8, 0.9]), 0.5)
        transformed = classification_metrics(labels, torch.tensor([0.1, 0.8, 0.2, 0.9]), 0.5)
        results = {"clean": {"overall": clean}, "jpeg_q30": {"overall": transformed}}
        summary = robustness_scorecard(results)
        self.assertEqual(clean["sample_count"], 4)
        self.assertEqual(summary["clean_balanced_accuracy"], 1.0)
        self.assertEqual(summary["worst_transformed_condition"], "jpeg_q30")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "conditions.csv"
            write_condition_csv(output, results)
            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertIn("false_positive_rate", lines[0])

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

    def test_every_exact_challenge_condition_preserves_shape(self) -> None:
        image = Image.fromarray(np.full((40, 60, 3), 127, dtype=np.uint8), "RGB")
        for condition in ROBUSTNESS_CONDITIONS:
            transformed = DeterministicTransform(condition, 42, "/example/image.png", 0)(image)
            self.assertEqual(transformed.mode, "RGB")
            self.assertEqual(transformed.size, image.size)

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

    def test_three_expert_mixture_shapes_and_gate_prior(self) -> None:
        config = ModelConfig(
            forensic_mode="laplacian_fft", forensic_dim=2560, head_type="tri_mixture"
        )
        head = AdaptiveTriExpertHead(config).eval()
        features = torch.zeros(4, 3072)
        self.assertEqual(tuple(head(features).shape), (4,))
        weights = head.gate_weights(features)
        self.assertEqual(tuple(weights.shape), (4, 3))
        self.assertTrue(torch.allclose(weights.sum(1), torch.ones(4), atol=1e-6))
        self.assertTrue(torch.allclose(weights[0], torch.tensor((0.45, 0.35, 0.20)), atol=1e-5))

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

    def test_nested_image_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "ai" / "flux" / "nested" / "image.png"
            path.parent.mkdir(parents=True)
            self.assertEqual(image_source(path, root), "flux")

    def test_paired_generator_metrics_macro_average(self) -> None:
        labels = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.float32)
        probabilities = torch.tensor([0.1, 0.8, 0.9, 0.8, 0.2, 0.7])
        sources = ["raise", "raise", "flux", "flux", "sd35", "sd35"]
        report = paired_generator_metrics(labels, probabilities, sources, 0.5)
        self.assertAlmostEqual(report["real_false_positive_rate"], 0.5)
        self.assertAlmostEqual(report["generators"]["flux"]["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(report["generators"]["sd35"]["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(report["macro_balanced_accuracy"], 0.625)
        self.assertEqual(report["worst_generator"], "sd35")

    def test_prepare_bfree_archives_without_reencoding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_archive, fake_archive = root / "real_RAISE_1k.zip", root / "sd3_flux.zip"

            def image_bytes(color: tuple[int, int, int]) -> bytes:
                buffer = BytesIO()
                Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
                return buffer.getvalue()

            with zipfile.ZipFile(real_archive, "w") as archive:
                archive.writestr("RAISE/one.png", image_bytes((1, 2, 3)))
                archive.writestr("RAISE/two.png", image_bytes((4, 5, 6)))
            with zipfile.ZipFile(fake_archive, "w") as archive:
                archive.writestr("FLUX/one.png", image_bytes((7, 8, 9)))
                archive.writestr("FLUX/two.png", image_bytes((10, 11, 12)))
                archive.writestr("stable-diffusion-3.5/one.png", image_bytes((13, 14, 15)))
                archive.writestr("stable-diffusion-3.5/two.png", image_bytes((16, 17, 18)))
            output = root / "prepared"
            manifest = prepare_dataset(real_archive, fake_archive, output, expected_per_source=2)
            self.assertEqual(manifest["counts"], {"real/raise": 2, "ai/flux": 2, "ai/sd35": 2})
            self.assertEqual(len(load_labeled_paths(output)), 6)
            self.assertFalse(manifest["images_reencoded"])

    def test_checksum_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checksum = Path(tmp) / "checksum.txt"
            checksum.write_text("d41d8cd98f00b204e9800998ecf8427e  file.zip\n", encoding="utf-8")
            self.assertEqual(parse_checksums(checksum)["file.zip"], ("md5", "d41d8cd98f00b204e9800998ecf8427e"))


if __name__ == "__main__":
    unittest.main()
