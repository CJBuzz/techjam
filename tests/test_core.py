import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aigc_detector.data import RobustTransform, load_labeled_paths, stratified_split, stratified_train_val_test_split
from aigc_detector.model import FrozenEncoders, FusionHead, ModelConfig


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


if __name__ == "__main__":
    unittest.main()
