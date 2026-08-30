from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch
from PIL import Image

from .data import BALANCED_TRANSFORM_GROUPS, DeterministicTransform, RobustTransform


def _label(row: dict) -> int:
    value = row.get("label")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"real", "authentic", "0"}:
            return 0
        if normalized in {"fake", "ai", "synthetic", "generated", "1"}:
            return 1
    if value in {0, False}:
        return 0
    if value in {1, True}:
        return 1
    raise ValueError(f"Unsupported streamed label: {value!r}")


def bounded_stream_sample(
    rows: Iterable[dict],
    real_count: int,
    fake_count: int,
    max_fake_per_model: int,
    max_real_per_source: int,
    progress_every: int = 100,
    progress_seconds: float = 30.0,
    progress_callback: Callable[[dict], None] | None = None,
    relax_after_no_progress: int = 2000,
    max_inspected_rows: int = 0,
    sampler_state: dict | None = None,
) -> Iterator[dict]:
    """Select lazily without retaining prior rows or their encoded image bytes."""
    if min(real_count, fake_count, max_fake_per_model, max_real_per_source, relax_after_no_progress) < 1:
        raise ValueError("Counts and per-source quotas must be positive")
    inspection_limit = max_inspected_rows or max(10_000, 50 * (real_count + fake_count))
    if inspection_limit < real_count + fake_count:
        raise ValueError("max_inspected_rows cannot be smaller than the requested sample count")
    selected = Counter()
    fake_models = Counter()
    real_sources = Counter()
    rejected_quota = Counter()
    relaxed = {0: False, 1: False}
    relaxed_at = {0: None, 1: None}
    last_accepted = {0: 0, 1: 0}
    started = last_report = time.monotonic()

    def update_state(inspected: int, complete: bool = False) -> dict:
        state = {
            "inspected": inspected,
            "accepted_real": selected[0],
            "target_real": real_count,
            "accepted_fake": selected[1],
            "target_fake": fake_count,
            "fake_models": len(fake_models),
            "real_sources": len(real_sources),
            "rejected_real_quota": rejected_quota[0],
            "rejected_fake_quota": rejected_quota[1],
            "real_quota_relaxed": relaxed[0],
            "fake_quota_relaxed": relaxed[1],
            "real_quota_relaxed_at": relaxed_at[0],
            "fake_quota_relaxed_at": relaxed_at[1],
            "real_source_counts": dict(real_sources),
            "fake_model_counts": dict(fake_models),
            "max_inspected_rows": inspection_limit,
            "complete": complete,
            "elapsed_seconds": time.monotonic() - started,
        }
        if sampler_state is not None:
            sampler_state.clear()
            sampler_state.update(state)
        return state

    def report(inspected: int) -> None:
        nonlocal last_report
        state = update_state(inspected)
        if progress_callback:
            progress_callback(state)
        else:
            print(
                "stream "
                f"inspected={state['inspected']} "
                f"real={state['accepted_real']}/{state['target_real']} "
                f"fake={state['accepted_fake']}/{state['target_fake']} "
                f"fake_models={state['fake_models']} real_sources={state['real_sources']} "
                f"rejected_real_quota={state['rejected_real_quota']} "
                f"rejected_fake_quota={state['rejected_fake_quota']} "
                f"elapsed={state['elapsed_seconds']:.1f}s",
                flush=True,
            )
        last_report = time.monotonic()

    for stream_index, row in enumerate(rows):
        inspected = stream_index + 1
        if inspected > inspection_limit:
            update_state(inspection_limit)
            raise RuntimeError(
                f"Reached max-inspected-rows={inspection_limit} with "
                f"real={selected[0]}/{real_count}, fake={selected[1]}/{fake_count}; "
                "targets remain unsatisfied after quota fallback"
            )
        label = _label(row)
        if selected[label] >= (real_count if label == 0 else fake_count):
            if inspected % progress_every == 0 or time.monotonic() - last_report >= progress_seconds:
                report(inspected)
            continue
        model_name = str(row.get("model_name") or "unknown") if label else None
        real_source = str(row.get("real_source") or "unknown") if not label else None
        over_quota = (
            label == 1 and fake_models[model_name] >= max_fake_per_model
        ) or (
            label == 0 and real_sources[real_source] >= max_real_per_source
        )
        if over_quota and not relaxed[label]:
            rejected_quota[label] += 1
            if inspected - last_accepted[label] >= relax_after_no_progress:
                relaxed[label] = True
                relaxed_at[label] = inspected
                quota_name = "fake model" if label else "real_source"
                accepted = selected[label]
                target = fake_count if label else real_count
                distinct = len(fake_models) if label else len(real_sources)
                print(
                    f"relaxing {quota_name} cap after {inspected - last_accepted[label]} rows "
                    f"without {('fake' if label else 'real')} progress: "
                    f"accepted {accepted}/{target} from {distinct} sources at inspected={inspected}",
                    flush=True,
                )
            else:
                if inspected % progress_every == 0 or time.monotonic() - last_report >= progress_seconds:
                    report(inspected)
                continue
        if over_quota and relaxed[label]:
            pass
        elif over_quota:
            if inspected % progress_every == 0 or time.monotonic() - last_report >= progress_seconds:
                report(inspected)
            continue
        image_name = str(row.get("image_name") or f"stream-{stream_index}")
        source_name = model_name if label else real_source
        original_id = f"{label}:{source_name}:{image_name}"
        selected[label] += 1
        last_accepted[label] = inspected
        if label:
            fake_models[model_name] += 1
        else:
            real_sources[real_source] += 1
        if inspected % progress_every == 0 or time.monotonic() - last_report >= progress_seconds:
            report(inspected)
        selected_row = {
            "image_data": row["image_data"],
            "image_name": image_name,
            "label": label,
            "model_name": model_name,
            "real_source": real_source,
            "subset": row.get("subset"),
            "architecture": row.get("architecture"),
            "original_id": original_id,
        }
        yield selected_row
        if selected[0] >= real_count and selected[1] >= fake_count:
            report(inspected)
            update_state(inspected, complete=True)
            break
    else:
        inspected = locals().get("inspected", 0)
        update_state(inspected)
        raise RuntimeError(
            f"Stream exhausted after {inspected} inspected rows with "
            f"real={selected[0]}/{real_count}, fake={selected[1]}/{fake_count}"
        )


def paired_views(selected: dict, robust_views: int, seed: int) -> tuple[list[Image.Image], list[dict]]:
    """Create clean and deterministic robust views with cache-safe metadata."""
    if robust_views < 1:
        raise ValueError("robust_views must be at least one")
    with Image.open(io.BytesIO(selected["image_data"])) as decoded:
        image = decoded.convert("RGB")
    images: list[Image.Image] = []
    metadata: list[dict] = []
    for repeat in range(robust_views + 1):
        digest = hashlib.sha256(f"{selected['original_id']}\0{repeat}".encode()).digest()
        group = "clean" if repeat == 0 else BALANCED_TRANSFORM_GROUPS[
            int.from_bytes(digest[:4], "big") % len(BALANCED_TRANSFORM_GROUPS)
        ]
        transform = RobustTransform("clean") if repeat == 0 else DeterministicTransform(
            group, seed, selected["original_id"], repeat
        )
        images.append(transform(image.copy()))
        metadata.append({
            "image_name": selected["image_name"],
            "label": selected["label"],
            "model_name": selected["model_name"],
            "real_source": selected["real_source"],
            "subset": selected["subset"],
            "architecture": selected["architecture"],
            "transform_group": group,
            "original_id": selected["original_id"],
            "repeat": repeat,
            "seed": seed,
        })
    return images, metadata


def save_feature_chunk(path: Path, features: torch.Tensor, metadata: list[dict]) -> None:
    """Persist only tensors and scalar metadata; streamed image objects are excluded."""
    if len(features) != len(metadata):
        raise ValueError("Feature and metadata row counts differ")
    required = {
        "image_name", "label", "model_name", "real_source", "subset", "architecture",
        "transform_group", "original_id", "repeat", "seed",
    }
    if any(set(row) != required for row in metadata):
        raise ValueError("Stream cache metadata has an unexpected structure")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features.cpu(), "metadata": metadata}, path)


def completed_cache_state(directory: Path, robust_views: int) -> tuple[int, int]:
    """Validate complete original/view groups in saved chunks before resuming."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    repair = manifest.get("metadata_repair", {})
    excluded = set(repair.get("excluded_original_ids", []))
    repeats_by_original: dict[str, set[int]] = {}
    chunk_paths = sorted(directory.glob("chunk-*.pt"))
    for chunk_path in chunk_paths:
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=True)
        if len(chunk.get("features", ())) != len(chunk.get("metadata", ())):
            raise ValueError(f"Corrupted feature/metadata counts in {chunk_path}")
        for metadata in chunk["metadata"]:
            original_id = metadata["original_id"]
            if original_id in excluded:
                continue
            repeat = int(metadata["repeat"])
            repeats = repeats_by_original.setdefault(original_id, set())
            if repeat in repeats:
                raise ValueError(f"Duplicate cached view for {original_id!r}, repeat {repeat}")
            repeats.add(repeat)
    expected = set(range(robust_views + 1))
    incomplete = [original_id for original_id, repeats in repeats_by_original.items() if repeats != expected]
    if incomplete:
        raise ValueError(f"Incomplete cached views for {incomplete[0]!r}; refusing unsafe resume")
    return len(repeats_by_original) + int(repair.get("excluded_selected_originals", 0)), len(chunk_paths)


def audit_stream_cache_metadata(directory: Path) -> dict:
    """Audit pairing using memory-mapped tensor storage without materializing feature payloads."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected = set(range(int(manifest["robust_views"]) + 1))
    records: dict[str, list[tuple[str, int, dict]]] = {}
    total_records = 0
    for chunk_path in sorted(directory.glob("chunk-*.pt")):
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=True, mmap=True)
        if len(chunk.get("features", ())) != len(chunk.get("metadata", ())):
            raise ValueError(f"Corrupted feature/metadata counts in {chunk_path}")
        total_records += len(chunk["metadata"])
        for row_index, metadata in enumerate(chunk["metadata"]):
            records.setdefault(metadata["original_id"], []).append(
                (chunk_path.name, row_index, metadata)
            )
    invalid: dict[str, dict] = {}
    categories = Counter()
    cross_chunk_invalid = 0
    for original_id, items in records.items():
        repeat_counts = Counter(int(item[2]["repeat"]) for item in items)
        reasons = []
        if set(repeat_counts) != expected:
            reasons.append("missing_repeat")
        if any(count != 1 for count in repeat_counts.values()):
            reasons.append("duplicate_repeat")
        if len(items) != len(expected):
            reasons.append("wrong_record_count")
        if reasons:
            chunks = sorted({item[0] for item in items})
            cross_chunk_invalid += len(chunks) > 1
            categories.update(reasons)
            invalid[original_id] = {
                "reasons": reasons,
                "records": len(items),
                "repeat_counts": dict(sorted(repeat_counts.items())),
                "chunks": chunks,
                "selected_original_occurrences": max(repeat_counts.values(), default=1),
            }
    invalid_records = sum(item["records"] for item in invalid.values())
    invalid_occurrences = sum(item["selected_original_occurrences"] for item in invalid.values())
    return {
        "total_unique_original_ids": len(records),
        "total_records": total_records,
        "valid_original_ids": len(records) - len(invalid),
        "valid_records": total_records - invalid_records,
        "invalid_unique_original_ids": len(invalid),
        "invalid_selected_originals": invalid_occurrences,
        "invalid_records": invalid_records,
        "failure_categories": dict(categories),
        "invalid_cross_chunk": cross_chunk_invalid,
        "invalid": invalid,
    }


def repair_stream_cache_metadata(directory: Path) -> dict:
    """Atomically record exclusions for ambiguous IDs; feature tensors remain untouched."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "metadata_repair" in manifest:
        return manifest["metadata_repair"]
    audit = audit_stream_cache_metadata(directory)
    excluded_ids = sorted(audit["invalid"])
    if not excluded_ids:
        return {}
    repair = {
        "version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "action": "exclude_ambiguous_original_ids",
        "reason": "original_id collisions contain multiple complete, non-identical paired view sets",
        "excluded_original_ids": excluded_ids,
        "excluded_unique_original_ids": audit["invalid_unique_original_ids"],
        "excluded_selected_originals": audit["invalid_selected_originals"],
        "excluded_records": audit["invalid_records"],
        "retained_originals": audit["valid_original_ids"],
        "retained_records": audit["valid_records"],
        "failure_categories": audit["failure_categories"],
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = directory / f"manifest.backup-pre-metadata-repair-{stamp}.json"
    if backup.exists():
        raise FileExistsError(f"Metadata backup already exists: {backup}")
    shutil.copy2(manifest_path, backup)
    manifest["metadata_repair"] = repair
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return repair


def load_stream_feature_cache(
    directory: Path, expected_model_config: dict
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor, int, dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["model_config"] != expected_model_config:
        raise ValueError("Stream cache encoder configuration does not match training")
    excluded = set(manifest.get("metadata_repair", {}).get("excluded_original_ids", []))
    records: dict[str, list[tuple[dict, torch.Tensor]]] = {}
    for chunk_path in sorted(directory.glob("chunk-*.pt")):
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=True, mmap=True)
        for feature, metadata in zip(chunk["features"], chunk["metadata"], strict=True):
            if metadata["original_id"] in excluded:
                continue
            records.setdefault(metadata["original_id"], []).append((metadata, feature))
    if not records:
        raise ValueError(f"No completed feature chunks in {directory}")
    repeats = manifest["robust_views"] + 1
    expected_feature_dim = sum(
        int(manifest["model_config"].get(key, 0))
        for key in ("clip_dim", "forensic_dim", "quality_dim")
    )
    ordered_ids = sorted(records)
    features, labels, groups, original_indices = [], [], [], []
    required_metadata = {"label", "original_id", "repeat", "transform_group"}
    for original_id, items in records.items():
        if any(not required_metadata.issubset(metadata) for metadata, _ in items):
            raise ValueError(f"Incomplete structural metadata for {original_id!r}")
        if len({int(metadata["label"]) for metadata, _ in items}) != 1:
            raise ValueError(f"Inconsistent labels for {original_id!r}")
        if expected_feature_dim and any(feature.numel() != expected_feature_dim for _, feature in items):
            raise ValueError(f"Feature dimension mismatch for {original_id!r}")
    for repeat in range(repeats):
        for original_index, original_id in enumerate(ordered_ids):
            matches = [item for item in records[original_id] if item[0]["repeat"] == repeat]
            if len(matches) != 1:
                raise ValueError(f"Incomplete paired metadata for {original_id!r}, repeat {repeat}")
            metadata, feature = matches[0]
            if repeat == 0 and metadata["transform_group"] != "clean":
                raise ValueError(f"Clean-view metadata mismatch for {original_id!r}")
            if repeat > 0 and metadata["transform_group"] == "clean":
                raise ValueError(f"Robust-view metadata mismatch for {original_id!r}, repeat {repeat}")
            features.append(feature)
            labels.append(metadata["label"])
            groups.append(metadata["transform_group"])
            original_indices.append(original_index)
    return (
        torch.stack(features),
        torch.tensor(labels, dtype=torch.float32),
        groups,
        torch.tensor(original_indices),
        len(ordered_ids),
        manifest,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a bounded raw-image-free streamed feature cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="OwensLab/CommunityForensics-Small")
    parser.add_argument("--split", default="train")
    parser.add_argument("--real-count", type=int, default=3000)
    parser.add_argument("--fake-count", type=int, default=3000)
    parser.add_argument("--max-fake-per-model", type=int, default=200)
    parser.add_argument("--max-real-per-source", type=int, default=750)
    parser.add_argument("--robust-views", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument(
        "--relax-after-no-progress", type=int, default=2000,
        help="Relax only a blocking diversity quota after this many rows without class progress",
    )
    parser.add_argument(
        "--max-inspected-rows", type=int, default=0,
        help="Hard inspection guard; 0 chooses max(10000, 50 * requested samples)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-originals", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()

    from datasets import load_dataset

    from .model import FrozenEncoders, ModelConfig

    config = ModelConfig(forensic_mode="laplacian_fft", forensic_dim=2560)
    manifest_static = {
        "schema_version": 2,
        "dataset": args.dataset,
        "split": args.split,
        "streaming": True,
        "shuffle_seed": args.seed,
        "shuffle_buffer_size": args.buffer_size,
        "requested_real": args.real_count,
        "requested_fake": args.fake_count,
        "max_fake_per_model": args.max_fake_per_model,
        "max_real_per_source": args.max_real_per_source,
        "relax_after_no_progress": args.relax_after_no_progress,
        "max_inspected_rows": args.max_inspected_rows,
        "robust_views": args.robust_views,
        "model_config": asdict(config),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if existing_manifest:
        existing_static = {
            key: value for key, value in existing_manifest.items()
            if key not in {"sampling_state", "metadata_repair"}
        }
        existing_version = existing_static.pop("schema_version", 1)
        comparable_static = {key: manifest_static.get(key) for key in existing_static}
        if existing_version not in {1, 2} or existing_static != comparable_static:
            raise ValueError("Existing stream cache manifest does not match requested settings")
    manifest = {**manifest_static, "sampling_state": (existing_manifest or {}).get("sampling_state", {})}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    completed, existing_chunks = completed_cache_state(args.output_dir, args.robust_views)
    if completed == args.real_count + args.fake_count:
        print(f"Stream feature cache already complete: {args.output_dir}")
        return
    if completed > args.real_count + args.fake_count:
        raise ValueError("Existing stream cache contains more originals than requested")
    print(
        f"Opening streaming dataset={args.dataset} split={args.split} "
        f"shuffle_buffer={args.buffer_size} completed={completed} chunks={existing_chunks}",
        flush=True,
    )
    stream = load_dataset(args.dataset, split=args.split, streaming=True).shuffle(
        seed=args.seed, buffer_size=args.buffer_size
    )
    sampling_state: dict = {}
    selected = bounded_stream_sample(
        stream, args.real_count, args.fake_count, args.max_fake_per_model, args.max_real_per_source,
        relax_after_no_progress=args.relax_after_no_progress,
        max_inspected_rows=args.max_inspected_rows,
        sampler_state=sampling_state,
    )
    print("Loading frozen encoders; first streamed row follows after the bounded shuffle buffer fills.", flush=True)
    encoders = FrozenEncoders(config, torch.device(args.device))
    print("Frozen encoders ready.", flush=True)
    pending_images, pending_metadata = [], []
    chunk_index = existing_chunks
    selected_index = 0

    def flush_chunk(images: list[Image.Image], metadata: list[dict], index: int) -> None:
        original_count = len(metadata) // (args.robust_views + 1)
        print(
            f"extract chunk={index} originals={original_count} views={len(images)} "
            f"cached_chunks={index} elapsed_stage=active",
            flush=True,
        )
        feature_batches = []
        batch_total = (len(images) + args.batch_size - 1) // args.batch_size
        for batch_index, start in enumerate(range(0, len(images), args.batch_size), start=1):
            feature_batches.append(encoders(images[start : start + args.batch_size]))
            if batch_index == batch_total or batch_index % 10 == 0:
                print(f"feature chunk={index} batch={batch_index}/{batch_total}", flush=True)
        save_feature_chunk(
            args.output_dir / f"chunk-{index:05d}.pt", torch.cat(feature_batches), metadata
        )
        manifest["sampling_state"] = dict(sampling_state)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"saved chunk={index} cached_chunks={index + 1}", flush=True)

    for row in selected:
        if selected_index < completed:
            selected_index += 1
            continue
        images, metadata = paired_views(row, args.robust_views, args.seed)
        pending_images.extend(images)
        pending_metadata.extend(metadata)
        selected_index += 1
        if selected_index % 25 == 0 or selected_index == completed + 1:
            print(
                f"accepted originals={selected_index}/{args.real_count + args.fake_count} "
                f"pending_views={len(pending_images)} cached_chunks={chunk_index}",
                flush=True,
            )
        if len(pending_metadata) >= args.chunk_originals * (args.robust_views + 1):
            flush_chunk(pending_images, pending_metadata, chunk_index)
            pending_images, pending_metadata = [], []
            chunk_index += 1
    if pending_metadata:
        flush_chunk(pending_images, pending_metadata, chunk_index)
    if selected_index < args.real_count + args.fake_count:
        raise ValueError(
            f"Stream ended after {selected_index} selected originals; requested "
            f"{args.real_count + args.fake_count}. Increase source quotas or lower requested counts."
        )
    manifest["sampling_state"] = dict(sampling_state)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Stream feature cache ready: {args.output_dir}")


if __name__ == "__main__":
    main()
