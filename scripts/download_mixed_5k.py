from __future__ import annotations

import argparse
import gc
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def save_image(image: Image.Image, folder: Path, index: int, jpeg: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if jpeg:
        image.convert("RGB").save(folder / f"{index:05d}.jpg", quality=95, subsampling=0)
    else:
        image.convert("RGB").save(folder / f"{index:05d}.png")


def download_cifake(output: Path, per_class: int) -> dict[int, int]:
    dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset", split="train", streaming=True)
    counts = {0: 0, 1: 0}
    feature = dataset.features.get("label") if dataset.features else None
    iterator = iter(dataset)
    for row in tqdm(iterator, desc="CIFAKE"):
        raw = row["label"]
        name = raw.lower() if isinstance(raw, str) else feature.int2str(int(raw)).lower()
        label = 1 if name == "fake" else 0 if name == "real" else None
        if label is None:
            raise ValueError(f"Unknown CIFAKE label {name!r}")
        if counts[label] >= per_class:
            continue
        save_image(row["image"], output / ("ai" if label else "real") / "cifake", counts[label], jpeg=False)
        counts[label] += 1
        if min(counts.values()) >= per_class:
            break
    close = getattr(iterator, "close", None)
    if close:
        close()
    del iterator, dataset
    gc.collect()
    return counts


def download_sid(output: Path, per_class: int) -> dict[int, int]:
    dataset = load_dataset("saberzl/SID_Set", split="train", streaming=True)
    counts = {0: 0, 1: 0}
    iterator = iter(dataset)
    for row in tqdm(iterator, desc="SID_Set"):
        raw = int(row["label"])
        if raw == 2:  # Exclude localized tampering: this prototype targets fully generated images.
            continue
        label = 0 if raw == 0 else 1
        if counts[label] >= per_class:
            continue
        save_image(row["image"], output / ("ai" if label else "real") / "sid", counts[label], jpeg=True)
        counts[label] += 1
        if min(counts.values()) >= per_class:
            break
    close = getattr(iterator, "close", None)
    if close:
        close()
    del iterator, dataset
    gc.collect()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced 5K CIFAKE + SID_Set subset")
    parser.add_argument("--output", type=Path, default=Path("data/mixed_5k"))
    parser.add_argument("--per-class-source", type=int, default=1250)
    args = parser.parse_args()
    if any(args.output.rglob("*")):
        raise ValueError(f"Output directory must be empty to prevent mixed/stale samples: {args.output}")
    results = {
        "cifake": download_cifake(args.output, args.per_class_source),
        "sid": download_sid(args.output, args.per_class_source),
    }
    expected = {0: args.per_class_source, 1: args.per_class_source}
    if any(counts != expected for counts in results.values()):
        raise RuntimeError(f"Could not collect the requested balanced subset: {results}")
    print(f"Saved {4 * args.per_class_source} images under {args.output}: {results}")


if __name__ == "__main__":
    main()
