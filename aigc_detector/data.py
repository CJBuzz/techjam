from __future__ import annotations

import io
import hashlib
import random
import csv
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


def image_source(path: str | Path, root: str | Path) -> str:
    """Return the source folder in ``root/{real,ai}/{source}/...`` layouts."""
    parts = Path(path).resolve().relative_to(Path(root).resolve()).parts
    return parts[1].lower() if len(parts) >= 3 else "default"


def load_split_manifest(root: str | Path, manifest: str | Path) -> dict[str, list[tuple[Path, int]]]:
    """Load persisted original-level splits created by the scale-data preparation script."""
    root, manifest = Path(root), Path(manifest)
    splits: dict[str, list[tuple[Path, int]]] = {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = root / row["path"]
            if not path.is_file():
                raise FileNotFoundError(f"Manifest image does not exist: {path}")
            splits.setdefault(row["split"], []).append((path, int(row["label"])))
    required = {"train", "model_selection", "calibration", "test"}
    if set(splits) != required:
        raise ValueError(f"Manifest splits must be {sorted(required)}; got {sorted(splits)}")
    return splits


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
        source = image_source(path, root)
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
        if mode != "random" and mode not in self.names and mode not in TRANSFORM_CONDITIONS:
            raise ValueError(f"Unknown transform {mode!r}")
        if max_ops < 1:
            raise ValueError("max_ops must be at least 1")
        self.mode = mode
        self.max_ops = max_ops

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if self.mode != "random":
            operation, parameter = resolve_transform_condition(self.mode)
            return self._apply_one(image, operation, parameter)
        count = random.randint(1, self.max_ops)
        for mode in random.sample(self.names[1:], k=min(count, len(self.names) - 1)):
            image = self._apply_one(image, mode)
        return image

    @staticmethod
    def _apply_one(image: Image.Image, mode: str, parameter: float | int | None = None) -> Image.Image:
        if mode == "clean":
            return image
        if mode == "jpeg":
            buffer = io.BytesIO()
            quality = int(parameter) if parameter is not None else random.choice((30, 50, 70, 90))
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                return decoded.convert("RGB")
        if mode == "blur":
            sigma = float(parameter) if parameter is not None else random.choice((0.5, 1.0, 2.0))
            return image.filter(ImageFilter.GaussianBlur(sigma))
        if mode == "resize":
            scale = float(parameter) if parameter is not None else random.choice((0.25, 0.5))
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            return image.resize(size, Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR)
        if mode == "noise":
            array = np.asarray(image, dtype=np.float32) / 255.0
            sigma = float(parameter) if parameter is not None else random.choice((0.02, 0.05, 0.10))
            array = np.clip(array + np.random.normal(0.0, sigma, array.shape), 0.0, 1.0)
            return Image.fromarray((array * 255).astype(np.uint8), "RGB")
        if mode == "color":
            for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                factor = float(parameter) if parameter is not None else random.uniform(0.8, 1.2)
                image = enhancer(image).enhance(factor)
            return image
        crop_fraction = float(parameter) if parameter is not None else 0.8
        crop_w, crop_h = round(image.width * crop_fraction), round(image.height * crop_fraction)
        left, top = (image.width - crop_w) // 2, (image.height - crop_h) // 2
        return image.crop((left, top, left + crop_w, top + crop_h)).resize(image.size, Image.Resampling.BICUBIC)


TRANSFORM_CONDITIONS: dict[str, tuple[str, float | int | None]] = {
    "clean": ("clean", None),
    "jpeg_q90": ("jpeg", 90),
    "jpeg_q70": ("jpeg", 70),
    "jpeg_q50": ("jpeg", 50),
    "jpeg_q30": ("jpeg", 30),
    "blur_s0.5": ("blur", 0.5),
    "blur_s1.0": ("blur", 1.0),
    "blur_s2.0": ("blur", 2.0),
    "resize_x0.5": ("resize", 0.5),
    "resize_x0.25": ("resize", 0.25),
    "noise_s0.02": ("noise", 0.02),
    "noise_s0.05": ("noise", 0.05),
    "noise_s0.10": ("noise", 0.10),
    "color_0.8": ("color", 0.8),
    "color_1.2": ("color", 1.2),
    "crop_0.8": ("crop", 0.8),
}

ROBUSTNESS_CONDITIONS = tuple(TRANSFORM_CONDITIONS)
ROBUST_SELECTION_CONDITIONS = (
    "jpeg_q30", "blur_s2.0", "resize_x0.25", "noise_s0.10", "color_0.8", "crop_0.8",
)

# Compatibility API used by the scale calibration and severity-reporting tools.
# These specs deliberately resolve to the same canonical conditions used by the
# Track 5 robustness evaluator, so training and reporting cannot silently drift.
SEVERITY_SPECS: tuple[tuple[str, float], ...] = (
    ("clean", 0.0),
    *(("jpeg", float(value)) for value in (90, 70, 50, 30)),
    *(("blur", value) for value in (0.5, 1.0, 2.0)),
    *(("resize", value) for value in (0.5, 0.25)),
    *(("noise", value) for value in (0.02, 0.05, 0.10)),
    *(("color", value) for value in (0.8, 1.2)),
    ("crop", 0.80),
)


def severity_key(operation: str, value: float) -> str:
    """Stable, filesystem/JSON-friendly name for an exact challenge severity."""
    if operation == "clean":
        return "clean"
    if operation == "jpeg":
        return f"jpeg_q{int(value)}"
    if operation == "blur":
        return f"blur_s{value:.1f}"
    if operation == "resize":
        return f"resize_x{value:g}"
    if operation == "noise":
        return f"noise_s{value:.2f}"
    if operation == "color":
        return f"color_{value:g}"
    if operation == "crop":
        return f"crop_{value:g}"
    raise ValueError(f"Unknown severity operation: {operation!r}")


class ExactSeverityTransform:
    """Apply one canonical challenge severity deterministically for a path."""

    def __init__(self, operation: str, value: float, seed: int = 42, key: str = "") -> None:
        if (operation, float(value)) not in SEVERITY_SPECS:
            raise ValueError(f"Unknown severity: {(operation, value)!r}")
        self.operation = operation
        self.value = float(value)
        self.condition = severity_key(operation, self.value)
        self.seed = seed
        self.key = key

    def __call__(self, image: Image.Image) -> Image.Image:
        return DeterministicTransform(self.condition, self.seed, self.key, 0)(image)

TTA_MODES = {
    "none": ("clean",),
    "mild3": ("clean", "jpeg_q90", "resize_x0.5"),
}

# Exact challenge severities prevent a random cache from under-sampling the hard cases.
# A few realistic chains retain the redistribution scenarios from the original policy.
BALANCED_TRANSFORM_GROUPS = tuple(TRANSFORM_CONDITIONS)[1:] + (
    "resize_x0.5+jpeg_q70",
    "blur_s1.0+jpeg_q70",
    "crop_0.8+resize_x0.5",
    "color_0.8+jpeg_q70",
    "noise_s0.05+jpeg_q70",
)


def resolve_transform_condition(name: str) -> tuple[str, float | int | None]:
    """Resolve an exact challenge condition or a base operation with random severity."""
    if name in TRANSFORM_CONDITIONS:
        return TRANSFORM_CONDITIONS[name]
    if name in RobustTransform.names:
        return name, None
    raise ValueError(f"Unknown transform condition {name!r}")


class DeterministicTransform:
    """Apply a reproducible transform chain without perturbing global RNG state."""

    def __init__(self, group: str, seed: int, path: str, repeat: int) -> None:
        operations = tuple(group.split("+"))
        if not operations:
            raise ValueError(f"Unknown transform group {group!r}")
        resolved = tuple(resolve_transform_condition(operation) for operation in operations)
        digest = hashlib.sha256(f"{seed}\0{path}\0{repeat}".encode()).digest()
        self.seed = int.from_bytes(digest[:8], "big") % (2**32)
        self.operations = resolved

    def __call__(self, image: Image.Image) -> Image.Image:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            random.seed(self.seed)
            np.random.seed(self.seed)
            image = image.convert("RGB")
            for operation, parameter in self.operations:
                image = RobustTransform._apply_one(image, operation, parameter)
            return image
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def test_time_views(image: Image.Image, mode: str, seed: int, identity: str) -> list[Image.Image]:
    """Create deterministic views after an official condition has already been applied."""
    if mode not in TTA_MODES:
        raise ValueError(f"Unknown TTA mode {mode!r}")
    views = []
    for view_index, condition in enumerate(TTA_MODES[mode]):
        transform = RobustTransform("clean") if condition == "clean" else DeterministicTransform(
            condition, seed, identity, view_index
        )
        views.append(transform(image.copy()))
    return views


def average_view_logits(logits: torch.Tensor) -> torch.Tensor:
    """Average pre-sigmoid logits across a non-empty final view dimension."""
    if logits.ndim < 2 or logits.shape[-1] < 1:
        raise ValueError("Expected logits with a non-empty final TTA-view dimension")
    return logits.mean(dim=-1)


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
