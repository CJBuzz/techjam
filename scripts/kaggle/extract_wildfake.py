"""Build a resumable, raw-image-free WildFake feature cache on Kaggle.

The script reads images directly from an attached Kaggle dataset.  Its output is
only frozen encoder features and scalar provenance metadata, so the WildFake
pixels never need to be copied to the local workstation or into the cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from aigc_detector.data import TRANSFORM_CONDITIONS, DeterministicTransform
from aigc_detector.model import FrozenEncoders, ModelConfig
from aigc_detector.tooling.streaming_cache import completed_cache_state, save_feature_chunk
from aigc_detector.train import choose_device


def normalize_label(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "fake", "ai", "synthetic", "generated"}:
        return 1
    if normalized in {"0", "false", "no", "real", "authentic", "human"}:
        return 0
    raise ValueError(f"Unsupported label {value!r}; map it to 0/1 in the metadata CSV first")


def cell(row: dict[str, str], column: str) -> str:
    return row.get(column, "").strip() if column else ""


def select_diverse(rows: list[dict], target: int, key: str, cap: int, seed: int) -> list[dict]:
    """Round-robin across provenance groups, then fill deterministically.

    ``cap=0`` disables a per-group cap.  Keeping this operation metadata-only
    makes selecting 100K originals inexpensive even when images are large.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row[key] or "unknown"].append(row)
    for name, bucket in buckets.items():
        random.Random(f"{seed}:{key}:{name}").shuffle(bucket)
    chosen: list[dict] = []
    positions = {name: 0 for name in buckets}
    names = sorted(buckets)
    while len(chosen) < target:
        progressed = False
        for name in names:
            position = positions[name]
            if position >= len(buckets[name]) or (cap and position >= cap):
                continue
            chosen.append(buckets[name][position])
            positions[name] += 1
            progressed = True
            if len(chosen) == target:
                break
        if not progressed:
            available = sum(min(len(bucket), cap or len(bucket)) for bucket in buckets.values())
            raise ValueError(
                f"Could select {target} rows across {key}; only {available} are available under cap={cap}. "
                "Lower --total-originals or raise the corresponding per-source cap."
            )
    return chosen


def load_rows(args: argparse.Namespace) -> list[dict]:
    if not args.metadata.is_file():
        raise FileNotFoundError(f"Metadata CSV does not exist: {args.metadata}")
    image_root = args.image_root.resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    rows: list[dict] = []
    with args.metadata.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {args.path_column, args.label_column}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Metadata must contain {sorted(required)}; found {reader.fieldnames}")
        for csv_index, raw in enumerate(reader):
            relative_path = cell(raw, args.path_column)
            if not relative_path:
                continue
            candidate = Path(relative_path)
            path = (candidate if candidate.is_absolute() else image_root / candidate).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Metadata row {csv_index + 2} points to no image: {path}")
            try:
                path.relative_to(image_root)
            except ValueError as error:
                raise ValueError(
                    f"Metadata row {csv_index + 2} resolves outside --image-root: {path}"
                ) from error
            label = normalize_label(cell(raw, args.label_column))
            generator = cell(raw, args.generator_column) or "unknown"
            real_source = cell(raw, args.real_source_column) or "unknown"
            identity = f"{label}:{generator if label else real_source}:{relative_path}"
            rows.append({
                "path": path, "label": label, "identity": identity,
                "generator": generator, "real_source": real_source,
                "architecture": cell(raw, args.architecture_column) or None,
            })
    if not rows:
        raise ValueError("No metadata rows were usable")
    return rows


def view_group(identity: str, repeat: int, seed: int) -> str:
    """Return clean, one-operation, or two-operation deterministic view names."""
    if repeat == 0:
        return "clean"
    names = tuple(name for name in TRANSFORM_CONDITIONS if name != "clean")
    digest = hashlib.sha256(f"{seed}\0{identity}\0{repeat}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return "+".join(rng.sample(names, k=repeat))


def build_selection(args: argparse.Namespace) -> list[dict]:
    rows = load_rows(args)
    real_target = args.real_count if args.real_count is not None else args.total_originals // 2
    fake_target = args.fake_count if args.fake_count is not None else args.total_originals - real_target
    if real_target < 1 or fake_target < 1 or real_target + fake_target != args.total_originals:
        raise ValueError("--real-count and --fake-count must be positive and sum to --total-originals")
    real = select_diverse([row for row in rows if row["label"] == 0], real_target,
                          "real_source", args.max_per_real_source, args.seed)
    fake = select_diverse([row for row in rows if row["label"] == 1], fake_target,
                          "generator", args.max_per_fake_generator, args.seed + 1)
    selected = real + fake
    random.Random(args.seed).shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 0–2-transform WildFake features from a Kaggle-mounted dataset")
    parser.add_argument("--image-root", type=Path, required=True, help="Attached Kaggle image directory, normally under /kaggle/input")
    parser.add_argument("--metadata", type=Path, required=True, help="CSV with image path, fake/real label, and provenance columns")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-originals", type=int, default=100_000, help="Balanced target; adjust this for a larger or smaller run")
    parser.add_argument("--real-count", type=int, default=None)
    parser.add_argument("--fake-count", type=int, default=None)
    parser.add_argument("--max-per-fake-generator", type=int, default=5_000)
    parser.add_argument("--max-per-real-source", type=int, default=10_000)
    parser.add_argument("--path-column", default="Image_path")
    parser.add_argument("--label-column", default="IsFake")
    parser.add_argument("--generator-column", default="Generator")
    parser.add_argument("--real-source-column", default="Category")
    parser.add_argument("--architecture-column", default="Architecture")
    parser.add_argument("--views-per-image", type=int, default=3, choices=(1, 2, 3),
                        help="1=clean only; 2=clean+one transform; 3=clean+one+two transforms")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-originals", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if args.chunk_originals < 1 or args.batch_size < 1:
        raise ValueError("--chunk-originals and --batch-size must be positive")
    args.image_root = args.image_root.resolve()
    args.metadata = args.metadata.resolve()

    selected = build_selection(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_digest = hashlib.sha256("\n".join(row["identity"] for row in selected).encode()).hexdigest()
    config = ModelConfig(forensic_mode="laplacian_fft", forensic_dim=2560)
    manifest_static = {
        "schema_version": 1, "kind": "kaggle_wildfake_feature_cache", "image_root": str(args.image_root),
        "metadata": str(args.metadata), "total_originals": args.total_originals,
        "real_count": sum(row["label"] == 0 for row in selected), "fake_count": sum(row["label"] == 1 for row in selected),
        "max_per_fake_generator": args.max_per_fake_generator, "max_per_real_source": args.max_per_real_source,
        "columns": {"path": args.path_column, "label": args.label_column, "generator": args.generator_column,
                    "real_source": args.real_source_column, "architecture": args.architecture_column},
        "seed": args.seed, "views_per_image": args.views_per_image, "robust_views": args.views_per_image - 1,
        "model_config": asdict(config), "selection_sha256": selection_digest,
        "selection_by_generator": dict(sorted(Counter(row["generator"] for row in selected if row["label"]).items())),
        "selection_by_real_source": dict(sorted(Counter(row["real_source"] for row in selected if not row["label"]).items())),
        "raw_images_in_output": False,
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in manifest_static.items()):
            raise ValueError("Existing cache manifest does not match this request; choose a new --output-dir")
        manifest = existing
    else:
        manifest = dict(manifest_static)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    completed, chunk_index = completed_cache_state(args.output_dir, args.views_per_image - 1)
    if completed == len(selected):
        print(f"Feature cache already complete: {args.output_dir}")
        return
    if completed > len(selected):
        raise ValueError("Existing cache has more originals than this selection")
    device = choose_device(args.device)
    print(f"Loading frozen encoders on {device}; selected={len(selected)} completed={completed}", flush=True)
    encoders = FrozenEncoders(config, device)

    for start in range(completed, len(selected), args.chunk_originals):
        batch_rows = selected[start : start + args.chunk_originals]
        images, metadata = [], []
        for row in tqdm(batch_rows, desc=f"prepare chunk {chunk_index}"):
            with Image.open(row["path"]) as source:
                original = source.convert("RGB")
            for repeat in range(args.views_per_image):
                group = view_group(row["identity"], repeat, args.seed)
                transform = DeterministicTransform(group, args.seed, row["identity"], repeat)
                images.append(transform(original.copy()))
                metadata.append({
                    "image_name": str(row["path"].relative_to(args.image_root)), "label": row["label"],
                    "model_name": row["generator"] if row["label"] else None,
                    "real_source": row["real_source"] if not row["label"] else None,
                    "subset": "kaggle_wildfake", "architecture": row["architecture"],
                    "transform_group": group, "original_id": row["identity"], "repeat": repeat, "seed": args.seed,
                })
        features = []
        for offset in tqdm(range(0, len(images), args.batch_size), desc=f"extract chunk {chunk_index}"):
            features.append(encoders(images[offset : offset + args.batch_size]))
        save_feature_chunk(args.output_dir / f"chunk-{chunk_index:05d}.pt", torch.cat(features), metadata)
        print(f"Saved chunk {chunk_index}: originals {start + len(batch_rows)}/{len(selected)}", flush=True)
        chunk_index += 1
    print(json.dumps({"output_dir": str(args.output_dir), "originals": len(selected),
                      "feature_rows": len(selected) * args.views_per_image,
                      "raw_images_in_output": False}, indent=2))


if __name__ == "__main__":
    main()
