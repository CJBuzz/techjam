import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aigc_detector.e7 import (
    MODEL_MODES,
    STABILITY_SCALES,
    SUPPORTED_BINS,
    DescriptorDataset,
    descriptor_cache_valid,
    extract_descriptor_tasks,
    radial_bin_indices,
    radial_fft_descriptor,
    require_validation_selection,
    select_features,
)
from aigc_detector.model import FrozenEncoders


class E7Tests(unittest.TestCase):
    def test_dimensions_finite_and_normalized_on_constant_image(self) -> None:
        image = Image.new("RGB", (32, 48), "gray")
        for bins in SUPPORTED_BINS:
            descriptor = radial_fft_descriptor(image, bins)
            self.assertEqual(descriptor.shape, (bins,))
            self.assertTrue(bool(torch.isfinite(descriptor).all()))
            self.assertGreaterEqual(float(descriptor.sum()), 0.0)
            self.assertLessEqual(float(descriptor.sum()), 1.000001)

    def test_radial_bins_cover_every_spectrum_pixel(self) -> None:
        indices = radial_bin_indices(31, 47, 32)
        counts = np.bincount(indices.ravel(), minlength=32)
        self.assertEqual(int(counts.sum()), 31 * 47)
        self.assertTrue(((indices >= 0) & (indices < 32)).all())

    def test_resize_descriptor_is_deterministic(self) -> None:
        image = Image.fromarray(np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3), "RGB")
        resized = image.resize((24, 16), Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR)
        self.assertTrue(torch.equal(radial_fft_descriptor(resized), radial_fft_descriptor(resized)))

    def test_worker_count_does_not_change_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index in range(2):
                path = Path(tmp) / f"{index}.png"
                Image.new("RGB", (24, 24), (index * 50, 20, 30)).save(path)
                paths.append(path)
            tasks = [(path, "clean", 0) for path in paths]
            serial = extract_descriptor_tasks(tasks, (16,), 42, 2, 0)[16]
            parallel = extract_descriptor_tasks(tasks, (16,), 42, 2, 2)[16]
            self.assertTrue(torch.equal(serial, parallel))

    def test_cache_resume_validation_and_fft_is_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.pt"
            manifest = {"bins": 16, "train_originals": 2, "train_repeats": 3,
                        "validation_originals": 2, "validation_conditions": ["clean", "jpeg_q90"],
                        "stability_originals": 2, "stability_scales": list(STABILITY_SCALES)}
            torch.save({"train": torch.zeros(6, 16), "validation": torch.zeros(4, 16),
                        "stability": torch.zeros(8, 16), "manifest": manifest}, path)
            self.assertTrue(descriptor_cache_valid(path, manifest))
            incompatible = {**manifest, "bins": 32}
            self.assertFalse(descriptor_cache_valid(path, incompatible))
        with self.assertRaisesRegex(ValueError, "validation-only"):
            require_validation_selection("test")
        self.assertNotIn("radial", inspect.getsource(FrozenEncoders._fft_tensor).lower())

    def test_all_model_modes_have_expected_dimensions(self) -> None:
        fused = torch.zeros(2, 3072)
        radial = torch.zeros(2, 32)
        expected = {"radial_only": 32, "clip_radial": 544, "fused_radial": 3104}
        self.assertEqual(set(MODEL_MODES), set(expected))
        for mode, width in expected.items():
            self.assertEqual(select_features(fused, radial, mode).shape, (2, width))


if __name__ == "__main__":
    unittest.main()
