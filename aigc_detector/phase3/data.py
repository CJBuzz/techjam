from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from aigc_detector.data import DeterministicTransform, ROBUSTNESS_CONDITIONS


@dataclass(frozen=True)
class ManifestRecord:
    path: str
    label: int
    split: str
    source: str | None = None
    generator: str | None = None
    width: int | None = None
    height: int | None = None
    original_split: str | None = None
    unique_id: str | None = None
    base_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def image_path(self) -> str:
        return self.path

    def validate(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("Manifest label must be 0 or 1")
        if self.split not in {"train", "validation"}:
            raise ValueError("Phase-3 manifests cannot contain final-test records")


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            values = json.loads(line)
            if "image_path" in values and "path" not in values:
                values["path"] = values.pop("image_path")
            record = ManifestRecord(**values); record.validate(); records.append(record)
    return records


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> None:
    records = list(records)
    for record in records:
        record.validate()
    Path(path).write_text("".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records), encoding="utf-8")


def manifest_counts(records: Iterable[ManifestRecord]) -> dict[str, dict[str, int]]:
    counts = {"class": {"real": 0, "ai": 0}, "source": {}, "generator": {}}
    for record in records:
        record.validate()
        label = "ai" if record.label else "real"
        counts["class"][label] += 1
        for field_name in ("source", "generator"):
            value = getattr(record, field_name)
            if value is not None:
                counts[field_name][value] = counts[field_name].get(value, 0) + 1
    return counts


def exact_track5_transform(condition: str, seed: int, identity: str, repeat: int = 0) -> DeterministicTransform:
    if condition not in ROBUSTNESS_CONDITIONS:
        raise ValueError(f"Unknown official Track-5 condition: {condition}")
    return DeterministicTransform(condition, seed, identity, repeat)
