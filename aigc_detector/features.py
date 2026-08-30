from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import BALANCED_TRANSFORM_GROUPS, DeterministicTransform, ImagePathDataset, RobustTransform, pil_collate
from .model import FrozenEncoders
from .model import image_quality_statistics


def extract_features(
    rows: list[tuple[Path, int]],
    encoders: FrozenEncoders,
    batch_size: int,
    augmentation_repeats: int = 1,
    robust: bool = False,
    transform_mode: str | None = None,
    augmentation_depth: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_paths: list[str] = []
    for repeat in range(augmentation_repeats):
        if transform_mode is not None:
            transform = RobustTransform(transform_mode)
        else:
            transform = (
                RobustTransform("random", max_ops=augmentation_depth)
                if robust and repeat > 0
                else RobustTransform("clean")
            )
        loader = DataLoader(
            ImagePathDataset(rows, transform), batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=pil_collate
        )
        for images, labels, paths in tqdm(loader, desc=f"features {repeat + 1}/{augmentation_repeats}"):
            all_features.append(encoders(images))
            all_labels.append(labels)
            all_paths.extend(paths)
    return torch.cat(all_features), torch.cat(all_labels), all_paths


def extract_features_with_factory(
    rows: list[tuple[Path, int]],
    encoders: FrozenEncoders,
    batch_size: int,
    transform_factory: Callable[[Path, int], Callable[[Image.Image], Image.Image]],
    description: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Extract one path-specific deterministic view for every original."""
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_paths: list[str] = []
    for start in tqdm(range(0, len(rows), batch_size), desc=description):
        images, labels = [], []
        for index, (path, label) in enumerate(rows[start : start + batch_size], start=start):
            with Image.open(path) as source:
                images.append(transform_factory(path, index)(source.convert("RGB")))
            labels.append(label)
            all_paths.append(str(path))
        all_features.append(encoders(images))
        all_labels.append(torch.tensor(labels, dtype=torch.float32))
    return torch.cat(all_features), torch.cat(all_labels), all_paths


def extract_balanced_features(
    rows: list[tuple[Path, int]], encoders: FrozenEncoders, batch_size: int, augmentation_repeats: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], torch.Tensor]:
    """Extract clean plus balanced views in repeat-major order for paired losses."""
    if augmentation_repeats < 2:
        raise ValueError("Balanced augmentation requires at least two feature passes")
    all_features, all_labels, all_paths, all_groups, original_indices = [], [], [], [], []
    for repeat in range(augmentation_repeats):
        examples = []
        for index, row in enumerate(rows):
            group = "clean" if repeat == 0 else BALANCED_TRANSFORM_GROUPS[
                (index * (augmentation_repeats - 1) + repeat - 1) % len(BALANCED_TRANSFORM_GROUPS)
            ]
            transform = RobustTransform("clean") if group == "clean" else DeterministicTransform(
                group, seed, str(row[0]), repeat
            )
            examples.append((row, transform, group, index))
        for start in tqdm(range(0, len(examples), batch_size), desc=f"balanced features {repeat + 1}/{augmentation_repeats}"):
            images, labels = [], []
            for (path, label), transform, group, index in examples[start : start + batch_size]:
                with Image.open(path) as source:
                    images.append(transform(source.convert("RGB")))
                labels.append(label)
                all_paths.append(str(path))
                all_groups.append(group)
                original_indices.append(index)
            all_features.append(encoders(images))
            all_labels.append(torch.tensor(labels, dtype=torch.float32))
    return torch.cat(all_features), torch.cat(all_labels), all_paths, all_groups, torch.tensor(original_indices)


def extract_balanced_quality_statistics(
    rows: list[tuple[Path, int]], augmentation_repeats: int, seed: int
) -> torch.Tensor:
    statistics = []
    for repeat in range(augmentation_repeats):
        for index, (path, _) in enumerate(tqdm(rows, desc=f"quality stats {repeat + 1}/{augmentation_repeats}")):
            group = "clean" if repeat == 0 else BALANCED_TRANSFORM_GROUPS[
                (index * (augmentation_repeats - 1) + repeat - 1) % len(BALANCED_TRANSFORM_GROUPS)
            ]
            transform = RobustTransform("clean") if group == "clean" else DeterministicTransform(group, seed, str(path), repeat)
            with Image.open(path) as source:
                statistics.append(image_quality_statistics([transform(source.convert("RGB"))])[0])
    return torch.stack(statistics)
