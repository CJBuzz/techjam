from __future__ import annotations

import io
import hashlib
import random
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
REAL_NAMES = {"real", "authentic", "non-aigc", "non_aigc", "0"}
AI_NAMES = {"ai", "fake", "aigc", "synthetic", "generated", "1"}


def find_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def load_labeled_paths(root: str | Path) -> list[tuple[Path, int]]:
    """Load root/{real,ai}/... while accepting a few common folder aliases."""
    root = Path(root)
    rows: list[tuple[Path, int]] = []
    for child in root.iterdir() if root.exists() else []:
        if not child.is_dir():
            continue
        name = child.name.lower()
        label = 0 if name in REAL_NAMES else 1 if name in AI_NAMES else None
        if label is not None:
            rows.extend((p, label) for p in find_images(child))
    if not rows:
        raise ValueError(f"Expected labeled folders such as {root}/real and {root}/ai")
    if {label for _, label in rows} != {0, 1}:
        raise ValueError("Both real (0) and AI-generated (1) images are required")
    return sorted(rows, key=lambda row: str(row[0]))


def stratified_split(
    rows: list[tuple[Path, int]], validation_fraction: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)
    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    for label in (0, 1):
        group = [row for row in rows if row[1] == label]
        rng.shuffle(group)
        n_val = max(1, round(len(group) * validation_fraction))
        n_val = min(n_val, len(group) - 1)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def stratified_train_val_test_split(
    rows: list[tuple[Path, int]],
    root: str | Path,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Split independently by class and source folder, keeping originals disjoint.

    For ``root/real/sid/x.jpg``, the source is ``sid``. Flat class folders use
    ``default``. Augmented copies are generated only after this split.
    """
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than 1")
    root = Path(root).resolve()
    groups: dict[tuple[int, str], list[tuple[Path, int]]] = {}
    for row in rows:
        path, label = row
        parts = path.resolve().relative_to(root).parts
        source = parts[1].lower() if len(parts) >= 3 else "default"
        groups.setdefault((label, source), []).append(row)

    rng = random.Random(seed)
    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    test: list[tuple[Path, int]] = []
    for key, group in sorted(groups.items()):
        if len(group) < 3:
            raise ValueError(f"Need at least three images for class/source group {key}; got {len(group)}")
        rng.shuffle(group)
        n_val = max(1, round(len(group) * validation_fraction))
        n_test = max(1, round(len(group) * test_fraction))
        if n_val + n_test >= len(group):
            n_val, n_test = 1, 1
        val.extend(group[:n_val])
        test.extend(group[n_val : n_val + n_test])
        train.extend(group[n_val + n_test :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


class RobustTransform:
    """Apply one or more challenge-style transforms, or keep the image clean."""

    names = ("clean", "jpeg", "blur", "resize", "noise", "color", "crop")

    def __init__(self, mode: str = "random", max_ops: int = 1) -> None:
        if mode != "random" and mode not in self.names:
            raise ValueError(f"Unknown transform {mode!r}")
        if max_ops < 1:
            raise ValueError("max_ops must be at least 1")
        self.mode = mode
        self.max_ops = max_ops

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if self.mode != "random":
            return self._apply_one(image, self.mode)
        count = random.randint(1, self.max_ops)
        for mode in random.sample(self.names[1:], k=min(count, len(self.names) - 1)):
            image = self._apply_one(image, mode)
        return image

    @staticmethod
    def _apply_one(image: Image.Image, mode: str) -> Image.Image:
        if mode == "clean":
            return image
        if mode == "jpeg":
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=random.choice((30, 50, 70, 90)))
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                return decoded.convert("RGB")
        if mode == "blur":
            return image.filter(ImageFilter.GaussianBlur(random.choice((0.5, 1.0, 2.0))))
        if mode == "resize":
            scale = random.choice((0.25, 0.5))
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            return image.resize(size, Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR)
        if mode == "noise":
            array = np.asarray(image, dtype=np.float32) / 255.0
            sigma = random.choice((0.02, 0.05, 0.10))
            array = np.clip(array + np.random.normal(0.0, sigma, array.shape), 0.0, 1.0)
            return Image.fromarray((array * 255).astype(np.uint8), "RGB")
        if mode == "color":
            for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                image = enhancer(image).enhance(random.uniform(0.8, 1.2))
            return image
        crop_w, crop_h = round(image.width * 0.8), round(image.height * 0.8)
        left, top = (image.width - crop_w) // 2, (image.height - crop_h) // 2
        return image.crop((left, top, left + crop_w, top + crop_h)).resize(image.size, Image.Resampling.BICUBIC)


BALANCED_TRANSFORM_GROUPS = (
    "jpeg", "blur", "resize", "noise", "color", "crop",
    "resize+jpeg", "blur+jpeg", "crop+resize", "color+jpeg", "noise+jpeg",
)


class DeterministicTransform:
    """Apply a reproducible transform chain without perturbing global RNG state."""

    def __init__(self, group: str, seed: int, path: str, repeat: int) -> None:
        operations = tuple(group.split("+"))
        if not operations or any(operation not in RobustTransform.names for operation in operations):
            raise ValueError(f"Unknown transform group {group!r}")
        digest = hashlib.sha256(f"{seed}\0{path}\0{repeat}".encode()).digest()
        self.seed = int.from_bytes(digest[:8], "big") % (2**32)
        self.operations = operations

    def __call__(self, image: Image.Image) -> Image.Image:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            random.seed(self.seed)
            np.random.seed(self.seed)
            image = image.convert("RGB")
            for operation in self.operations:
                image = RobustTransform._apply_one(image, operation)
            return image
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


class ImagePathDataset(Dataset):
    def __init__(
        self,
        rows: Iterable[tuple[Path, int]],
        transform: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        self.rows = list(rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Image.Image, int, str]:
        path, label = self.rows[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, str(path)


def pil_collate(batch: list[tuple[Image.Image, int, str]]) -> tuple[list[Image.Image], torch.Tensor, list[str]]:
    images, labels, paths = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.float32), list(paths)
