from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from prepare_mixed_scale import BKTree, content_hash, dhash


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing destination has the wrong size: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def read_json_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def image_record(source_path: Path, destination: Path, root: Path, source: str, label: int, split: str, upstream: str) -> dict:
    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        pixel_hash = content_hash(image)
        difference_hash = dhash(image)
        image.close()
    link_or_copy(source_path, destination)
    return {
        "upstream_identifier": upstream,
        "source": source,
        "label": label,
        "class": "ai" if label else "real",
        "path": destination.relative_to(root).as_posix(),
        "width": width,
        "height": height,
        "content_sha256": pixel_hash,
        "dhash64": f"{difference_hash:016x}",
        "split": split,
    }


def filter_new_cross_split_duplicates(records: list[dict], legacy_count: int, radius: int, root: Path) -> tuple[dict[str, object], list[dict]]:
    """Trust the locked legacy grouping and reject new rows that cross its split boundary.

    Re-clustering the legacy corpus transitively can merge broad low-information
    dHash chains that its persisted duplicate-atomic groups intentionally kept
    separate. New rows are therefore checked directly against all earlier rows;
    they never alter a legacy split or group.
    """
    tree = BKTree()
    exact: dict[str, list[int]] = {}
    for index, record in enumerate(records[:legacy_count]):
        value = int(record["dhash64"], 16)
        tree.add(value, index)
        exact.setdefault(record["content_sha256"], []).append(index)
        record["duplicate_group"] = f"legacy_{record['duplicate_group']}"

    rejected: set[int] = set()
    conflicts = []
    for index in tqdm(range(legacy_count, len(records)), desc="audit new near-duplicates"):
        record = records[index]
        value = int(record["dhash64"], 16)
        neighbors = set(tree.search(value, radius))
        neighbors.update(exact.get(record["content_sha256"], []))
        cross = [neighbor for neighbor in neighbors if records[neighbor]["split"] != record["split"]]
        if cross:
            rejected.add(index)
            conflicts.append({
                "new_path": record["path"],
                "new_split": record["split"],
                "conflicting_paths": [records[neighbor]["path"] for neighbor in cross[:20]],
                "conflicting_splits": sorted({records[neighbor]["split"] for neighbor in cross}),
            })
            continue
        record["duplicate_group"] = f"wild_{index - legacy_count:06d}"
        tree.add(value, index)
        exact.setdefault(record["content_sha256"], []).append(index)

    kept_indices = [index for index in range(len(records)) if index not in rejected]
    counts = Counter((records[index]["split"], int(records[index]["label"])) for index in kept_indices)
    balance_drops = []
    for split in ("train", "model_selection"):
        target = min(counts[(split, 0)], counts[(split, 1)])
        for label in (0, 1):
            surplus = counts[(split, label)] - target
            candidates = [
                index for index in reversed(kept_indices)
                if index >= legacy_count and records[index]["split"] == split and int(records[index]["label"]) == label
            ]
            for index in candidates[:surplus]:
                rejected.add(index)
                balance_drops.append(records[index]["path"])

    kept = []
    for index, record in enumerate(records):
        if index not in rejected:
            kept.append(record)
            continue
        path = root / record["path"]
        if path.exists():
            excluded = root / ".excluded" / record["path"]
            excluded.parent.mkdir(parents=True, exist_ok=True)
            if excluded.exists():
                path.unlink()
            else:
                path.replace(excluded)
    return {
        "perceptual_radius": radius,
        "new_rows_rejected_for_cross_split_similarity": len(conflicts),
        "cross_split_conflicts": conflicts,
        "new_rows_dropped_for_class_balance": len(balance_drops),
        "class_balance_drops": balance_drops,
        "legacy_policy": "persisted mixed-40K duplicate groups and splits preserved exactly",
    }, kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine locked mixed-40K splits with audited WildFake training sources")
    parser.add_argument("--mixed-root", type=Path, default=Path("data/mixed_40k"))
    parser.add_argument("--wild-root", type=Path, default=Path("data/wildfake_diverse"))
    parser.add_argument("--output", type=Path, default=Path("data/mixed_wildfake_66k"))
    parser.add_argument("--imagenet-train", type=int, default=10_000)
    parser.add_argument("--imagenet-model-selection", type=int, default=3_000)
    parser.add_argument("--dhash-radius", type=int, default=4)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with (args.mixed_root / "split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        mixed_rows = list(csv.DictReader(handle))
    for row in tqdm(mixed_rows, desc="link mixed-40K"):
        source_path = args.mixed_root / row["path"]
        relative = Path(row["class"]) / f"mixed40k_{row['source']}" / Path(row["path"]).name
        destination = args.output / relative
        link_or_copy(source_path, destination)
        copied = dict(row)
        copied["path"] = relative.as_posix()
        records.append(copied)

    manifest_root = args.wild_root / "manifests"
    wild_specs = [
        ("ddim", 1, "train"),
        ("ddpm", 1, "train"),
        ("biggan", 1, "train"),
        ("stylegan", 1, "train"),
        ("adm_holdout", 1, "model_selection"),
    ]
    for source, label, split in wild_specs:
        for index, row in enumerate(tqdm(read_json_records(manifest_root / f"{source}.json"), desc=f"add {source}")):
            source_path = Path(row["path"])
            destination = args.output / "ai" / f"wildfake_{source}" / source_path.name
            records.append(image_record(
                source_path, destination, args.output, f"wildfake_{source}", label, split,
                f"WildFake:{row['official_train_path']}",
            ))

    imagenet_rows = read_json_records(manifest_root / "imagenet.json")
    required = args.imagenet_train + args.imagenet_model_selection
    if len(imagenet_rows) != required:
        raise ValueError(f"Expected {required} ImageNet records, found {len(imagenet_rows)}")
    for index, row in enumerate(tqdm(imagenet_rows, desc="add imagenet real")):
        split = "train" if index < args.imagenet_train else "model_selection"
        source_path = Path(row["path"])
        destination = args.output / "real" / "wildfake_imagenet" / source_path.name
        records.append(image_record(
            source_path, destination, args.output, "wildfake_imagenet", 0, split,
            f"WildFake:{row['official_train_path']}",
        ))

    audit, records = filter_new_cross_split_duplicates(records, len(mixed_rows), args.dhash_radius, args.output)

    fieldnames = (
        "upstream_identifier", "source", "label", "class", "path", "width", "height",
        "content_sha256", "dhash64", "duplicate_group", "split",
    )
    with (args.output / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: record[field] for field in fieldnames} for record in records)
    split_counts = Counter((record["split"], int(record["label"])) for record in records)
    source_counts = Counter((record["split"], record["source"], int(record["label"])) for record in records)
    report = {
        "total_images": len(records),
        "split_class_counts": {f"{split}:label_{label}": count for (split, label), count in sorted(split_counts.items())},
        "split_source_counts": {f"{split}:{source}:label_{label}": count for (split, source, label), count in sorted(source_counts.items())},
        "duplicate_audit": audit,
        "wildfake_policy": {
            "official_rows": "WildFake total_split train_metadata.csv only",
            "training_fake_sources": ["DDIM", "DDPM", "BigGAN", "StyleGAN"],
            "generator_held_out_model_selection_source": "ADM",
            "real_source": "ImageNet training rows",
            "explicitly_excluded": ["DALL-E Advanced", "COCO val2017", "all WildFake official test rows"],
        },
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
