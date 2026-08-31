from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def require_validation_selection(split: str) -> None:
    if split != "validation":
        raise ValueError("Phase-3 training and model selection are validation-only; test is forbidden")


@dataclass(frozen=True)
class DistributedConfig:
    enabled: bool = True
    backend: str = "nccl"
    expected_gpus: int = 2


@dataclass(frozen=True)
class Phase3Config:
    experiment: str
    backbone: str
    input_resolution: int = 224
    seed: int = 42
    precision: str = "fp16"
    max_wall_minutes: int = 600
    dataloader_workers: int = 2
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    selection_split: str = "validation"
    final_test_evaluated: bool = False
    runtime_internet_required: bool = False
    baseline_clean_balanced_accuracy: float = 0.9681
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_validation_selection(self.selection_split)
        if self.final_test_evaluated:
            raise ValueError("Phase-3 search configuration cannot evaluate final test")
        if self.runtime_internet_required:
            raise ValueError("Kaggle Phase-3 jobs must be fully offline")
        if self.precision != "fp16":
            raise ValueError("T4 production precision must be fp16")
        if not 1 <= self.max_wall_minutes <= 600:
            raise ValueError("max_wall_minutes must be in [1, 600]")
        if not 0 <= self.dataloader_workers <= 4:
            raise ValueError("dataloader_workers must be conservative (0-4 total per process)")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> Phase3Config:
    path = Path(path)
    if path.suffix == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".toml":
        values = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Phase-3 configs must be JSON or TOML")
    values["distributed"] = DistributedConfig(**values.get("distributed", {}))
    config = Phase3Config(**values)
    config.validate()
    return config
