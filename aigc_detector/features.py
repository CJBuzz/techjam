from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ImagePathDataset, RobustTransform, pil_collate
from .model import FrozenEncoders


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
