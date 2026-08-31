from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


SOURCE_EXCLUDES = {".git", "data", "artifacts", "checkpoints", "cache", "caches", "logs", "__pycache__",
                   ".venv", ".uv-cache", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SOURCE_SUFFIX_EXCLUDES = {".pt", ".pth", ".ckpt", ".safetensors", ".log", ".pyc"}


def kernel_metadata(kernel_id: str, title: str, code_file: str, dataset_sources: list[str],
                    kernel_sources: list[str] | None = None, model_sources: list[str] | None = None) -> dict:
    if "/" not in kernel_id or kernel_id.startswith("/"):
        raise ValueError("Kaggle kernel id must be supplied as username/slug")
    return {"id": kernel_id, "title": title, "code_file": code_file, "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": False, "dataset_sources": dataset_sources,
            "kernel_sources": kernel_sources or [], "model_sources": model_sources or []}


def write_kernel_metadata(output: Path, experiment: str, dataset_sources: list[str],
                          kernel_sources: list[str], model_sources: list[str]) -> Path:
    username = os.getenv("KAGGLE_USERNAME")
    if not username:
        raise ValueError("Set KAGGLE_USERNAME; it is never hard-coded")
    metadata = kernel_metadata(f"{username}/track5-phase3-{experiment}", f"Track5 Phase3 {experiment.upper()}",
                               "entrypoint.py", dataset_sources, kernel_sources, model_sources)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "kernel-metadata.json"; path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def source_included(relative: Path) -> bool:
    return not any(part in SOURCE_EXCLUDES for part in relative.parts) and relative.suffix not in SOURCE_SUFFIX_EXCLUDES


def package_source(source: Path, output: Path) -> None:
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source):
        raise ValueError("Source package output must be outside the repository being packaged")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite source package: {output}")
    output.mkdir(parents=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if not path.is_file() or not source_included(relative):
            continue
        destination = output / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def dataset_metadata(dataset_id: str, title: str) -> dict:
    if "/" not in dataset_id:
        raise ValueError("Kaggle dataset id must be username/slug")
    return {"title": title, "id": dataset_id, "licenses": [{"name": "other"}]}
