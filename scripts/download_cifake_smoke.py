from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a balanced CIFAKE smoke-test subset from Hugging Face")
    parser.add_argument("--output", type=Path, default=Path("data/cifake_smoke"))
    parser.add_argument("--per-class", type=int, default=50)
    args = parser.parse_args()
    # This compact Parquet mirror contains the CIFAKE training data (CIFAR-10 real, SD 1.4 fake).
    dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset", split="train", streaming=True)
    counts = {0: 0, 1: 0}
    for row in tqdm(dataset, desc="CIFAKE samples"):
        raw_label = row["label"]
        if isinstance(raw_label, str):
            class_name = raw_label.lower()
        else:
            feature = dataset.features.get("label") if dataset.features else None
            class_name = feature.int2str(int(raw_label)).lower() if hasattr(feature, "int2str") else str(raw_label)
        if class_name in {"fake", "ai", "aigc", "synthetic", "generated", "0"}:
            label = 1
        elif class_name in {"real", "authentic", "1"}:
            label = 0
        else:
            raise ValueError(f"Unknown CIFAKE class label: {class_name!r}")
        if label not in counts or counts[label] >= args.per_class:
            continue
        folder = args.output / ("ai" if label == 1 else "real")
        folder.mkdir(parents=True, exist_ok=True)
        row["image"].convert("RGB").save(folder / f"{counts[label]:04d}.png")
        counts[label] += 1
        if all(value >= args.per_class for value in counts.values()):
            break
    if counts != {0: args.per_class, 1: args.per_class}:
        raise RuntimeError(f"Could not collect balanced classes; got {counts}")
    print(f"Saved {sum(counts.values())} images under {args.output}")


if __name__ == "__main__":
    main()
