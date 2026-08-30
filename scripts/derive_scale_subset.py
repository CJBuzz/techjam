from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SPLIT_PER_STRATUM = {
    "train": 8750,
    "model_selection": 1250,
    "calibration": 625,
    "test": 1875,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive an exact grouped 50K subset from a larger persisted manifest")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with args.input_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields:
        raise ValueError("Input manifest has no header")

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["duplicate_group"]].append(row)
    strata = sorted({(row["split"], row["source"], row["label"]) for row in rows})
    targets = {(split, source, label): SPLIT_PER_STRATUM[split] for split, source, label in strata}
    available = Counter((row["split"], row["source"], row["label"]) for row in rows)
    short = {key: (available[key], target) for key, target in targets.items() if available[key] < target}
    if short:
        raise ValueError(f"Insufficient rows for targets: {short}")

    rng = random.Random(args.seed)
    grouped = list(groups.items())
    rng.shuffle(grouped)
    multi = sorted((item for item in grouped if len(item[1]) > 1), key=lambda item: len(item[1]), reverse=True)
    single = [item for item in grouped if len(item[1]) == 1]
    selected: list[dict[str, str]] = []
    counts = Counter()
    for _, members in multi:
        makeup = Counter((row["split"], row["source"], row["label"]) for row in members)
        if all(counts[key] + value <= targets[key] for key, value in makeup.items()):
            selected.extend(members)
            counts.update(makeup)

    singles_by_stratum: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for _, members in single:
        row = members[0]
        singles_by_stratum[(row["split"], row["source"], row["label"])].append(row)
    for key in sorted(targets):
        need = targets[key] - counts[key]
        candidates = singles_by_stratum[key]
        if len(candidates) < need:
            raise RuntimeError(f"Grouped selection cannot fill {key}: need {need}, have {len(candidates)} singletons")
        selected.extend(candidates[:need])
        counts[key] += need

    selected.sort(key=lambda row: (row["split"], row["source"], row["label"], row["path"]))
    output_groups = Counter(row["duplicate_group"] for row in selected)
    input_group_sizes = Counter(row["duplicate_group"] for row in rows)
    partial_groups = [group for group, count in output_groups.items() if count != input_group_sizes[group]]
    if len(selected) != 50_000 or counts != Counter(targets) or partial_groups:
        raise RuntimeError("Subset validation failed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    audit = {
        "input_manifest": str(args.input_manifest),
        "seed": args.seed,
        "selected_records": len(selected),
        "selected_duplicate_groups": len(output_groups),
        "partial_duplicate_groups": len(partial_groups),
        "split_counts": dict(Counter(row["split"] for row in selected)),
        "split_stratum_counts": {"|".join(key): counts[key] for key in sorted(counts)},
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
