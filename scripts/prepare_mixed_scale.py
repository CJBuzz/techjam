from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


SOURCES = {
    "cifake": "dragonintelligence/CIFAKE-image-dataset",
    "sid": "saberzl/SID_Set",
}


def content_hash(image: Image.Image) -> str:
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    try:
        payload = rgb.width.to_bytes(4, "big") + rgb.height.to_bytes(4, "big") + rgb.tobytes()
    finally:
        if rgb is not image:
            rgb.close()
    return hashlib.sha256(payload).hexdigest()


def dhash(image: Image.Image) -> int:
    gray = image.convert("L")
    resized = gray.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    resized.close()
    gray.close()
    return sum((pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8))


class BKTree:
    def __init__(self) -> None:
        self.root: list | None = None

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def search(self, value: int, radius: int) -> list[int]:
        found: list[int] = []
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                found.extend(node[1])
            pending.extend(child for edge, child in node[2].items() if distance - radius <= edge <= distance + radius)
        return found


def normalize_label(source: str, row: dict, feature) -> int | None:
    if source == "sid":
        raw = int(row["label"])
        return None if raw == 2 else int(raw == 1)
    raw = row["label"]
    name = raw.lower() if isinstance(raw, str) else feature.int2str(int(raw)).lower()
    if name not in {"fake", "real"}:
        raise ValueError(f"Unknown CIFAKE label {name!r}")
    return int(name == "fake")


def sample_source(output: Path, source: str, per_group: int, seed: int, shuffle_buffer: int) -> list[dict]:
    state_dir = output / ".prep"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{source}.jsonl"
    records = []
    if state_path.is_file():
        with state_path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    counts = Counter(record["label"] for record in records)
    seen_hashes = {record["content_sha256"] for record in records}
    for record in records:
        if not (output / record["path"]).is_file():
            raise FileNotFoundError(f"Preparation state references a missing image: {record['path']}")
    if counts[0] == counts[1] == per_group:
        print(f"Reusing completed {source} preparation state: {state_path}")
        return records
    if counts[0] > per_group or counts[1] > per_group:
        raise ValueError(f"Preparation state exceeds requested quota for {source}: {dict(counts)}")

    dataset = load_dataset(SOURCES[source], split="train", streaming=True)
    feature = dataset.features.get("label") if dataset.features else None
    keep_columns = {"image", "label", "img_id"}
    unused_columns = [name for name in dataset.column_names if name not in keep_columns]
    dataset = dataset.remove_columns(unused_columns)
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    iterator = iter(dataset)
    with state_path.open("a", encoding="utf-8", buffering=1) as state_handle:
        for stream_index, row in enumerate(tqdm(iterator, desc=f"sample {source}")):
            label = normalize_label(source, row, feature)
            if label is None or counts[label] >= per_group:
                continue
            source_image = row["image"]
            image = source_image.convert("RGB")
            source_image.close()
            digest = content_hash(image)
            if digest in seen_hashes:
                image.close()
                continue
            identifier = str(row.get("img_id") or f"shuffle-seed-{seed}-row-{stream_index}")
            relative = Path("ai" if label else "real") / source / f"{counts[label]:06d}-{digest[:12]}.png"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Lossless output avoids adding a dataset-wide JPEG signature that the
            # detector could learn as a source/class shortcut.
            image.save(destination, "PNG", optimize=False)
            record = {
                "upstream_identifier": identifier, "source": source, "label": label,
                "class": "ai" if label else "real", "path": relative.as_posix(),
                "width": image.width, "height": image.height, "content_sha256": digest,
                "dhash64": f"{dhash(image):016x}",
            }
            records.append(record)
            state_handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            seen_hashes.add(digest)
            counts[label] += 1
            image.close()
            del image, source_image, row
            if sum(counts.values()) % 100 == 0:
                gc.collect()
            if counts[0] == counts[1] == per_group:
                break
    close = getattr(iterator, "close", None)
    if close:
        close()
    if counts != Counter({0: per_group, 1: per_group}):
        raise RuntimeError(f"Insufficient {source} examples: {dict(counts)}")
    return records


def deduplicate_and_split(records: list[dict], seed: int, radius: int) -> tuple[list[dict], dict]:
    # Exact decoded-pixel duplicates retain the first seeded-shuffle occurrence.
    unique, exact_seen = [], {}
    exact_removed = []
    for record in records:
        digest = record["content_sha256"]
        if digest in exact_seen:
            exact_removed.append({"removed": record["path"], "kept": exact_seen[digest]})
        else:
            exact_seen[digest] = record["path"]
            unique.append(record)

    parent = list(range(len(unique)))
    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    tree = BKTree()
    for index, record in enumerate(tqdm(unique, desc="near-duplicate audit")):
        value = int(record["dhash64"], 16)
        for match in tree.search(value, radius):
            union(index, match)
        tree.add(value, index)
    components = defaultdict(list)
    for index in range(len(unique)):
        components[find(index)].append(index)

    fractions = {"train": 0.70, "model_selection": 0.10, "calibration": 0.05, "test": 0.15}
    strata = sorted({(r["source"], r["label"]) for r in unique})
    totals = Counter((r["source"], r["label"]) for r in unique)
    targets = {(split, stratum): totals[stratum] * fraction for split, fraction in fractions.items() for stratum in strata}
    assigned = Counter()
    rng = random.Random(seed)
    groups = list(components.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    for group_number, indices in enumerate(groups):
        makeup = Counter((unique[i]["source"], unique[i]["label"]) for i in indices)
        split = min(fractions, key=lambda candidate: sum(
            max(0.0, assigned[(candidate, stratum)] + count - targets[(candidate, stratum)]) ** 2
            - max(0.0, assigned[(candidate, stratum)] - targets[(candidate, stratum)]) ** 2
            for stratum, count in makeup.items()
        ))
        group_id = f"g{group_number:06d}"
        for index in indices:
            unique[index]["duplicate_group"] = group_id
            unique[index]["split"] = split
        for stratum, count in makeup.items():
            assigned[(split, stratum)] += count
    audit = {
        "input_records": len(records), "unique_records": len(unique),
        "exact_duplicates_removed": len(exact_removed), "exact_duplicate_examples": exact_removed[:100],
        "perceptual_radius": radius, "duplicate_groups": len(groups),
        "near_duplicate_groups": sum(len(group) > 1 for group in groups),
        "largest_duplicate_group": max(map(len, groups), default=0),
        "split_counts": dict(Counter(record["split"] for record in unique)),
        "split_stratum_counts": {f"{split}:{source}:{label}": assigned[(split, (source, label))]
                                  for split in fractions for source, label in strata},
    }
    return unique, audit


def write_csv(path: Path, records: list[dict]) -> None:
    fields = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a balanced, audited CIFAKE + SID corpus")
    parser.add_argument("--output", type=Path, default=Path("data/mixed_100k"))
    parser.add_argument("--per-class-source", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-buffer", type=int, default=256,
        help="Bounded streaming shuffle buffer; large high-resolution SID buffers can exhaust RAM",
    )
    parser.add_argument("--dhash-radius", type=int, default=4)
    args = parser.parse_args()
    if args.shuffle_buffer < 1:
        raise ValueError("--shuffle-buffer must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not (args.output / ".prep").is_dir():
        raise ValueError(
            f"Output contains data without resumable preparation state: {args.output}. "
            "Move it aside and start again."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for offset, source in enumerate(("cifake", "sid")):
        records.extend(sample_source(args.output, source, args.per_class_source, args.seed + offset, args.shuffle_buffer))
    write_csv(args.output / "sampled_manifest.csv", records)
    records, audit = deduplicate_and_split(records, args.seed, args.dhash_radius)
    write_csv(args.output / "split_manifest.csv", records)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for state_path in (args.output / ".prep").glob("*.jsonl"):
        state_path.unlink()
    (args.output / ".prep").rmdir()
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
