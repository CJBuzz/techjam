from __future__ import annotations

import argparse
import struct
from collections import Counter, defaultdict
from pathlib import Path


CENTRAL_HEADER = b"PK\x01\x02"


def inspect(path: Path) -> None:
    payload = path.read_bytes()
    offset = payload.find(CENTRAL_HEADER)
    if offset < 0:
        raise ValueError(f"No ZIP central directory found in {path}")

    disks: Counter[int] = Counter()
    prefixes: dict[str, Counter[int]] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)
    entries = 0
    while payload[offset : offset + 4] == CENTRAL_HEADER:
        if offset + 46 > len(payload):
            raise ValueError("Truncated central-directory header")
        header = payload[offset : offset + 46]
        name_length, extra_length, comment_length = struct.unpack_from("<HHH", header, 28)
        disk = struct.unpack_from("<H", header, 34)[0]
        start = offset + 46
        name = payload[start : start + name_length].decode("utf-8", errors="replace")
        normalized = name.replace("\\", "/")
        lower = normalized.lower()
        split = "other"
        for candidate in ("train/ai", "train/nature", "val/ai", "val/nature"):
            if f"/{candidate}/" in f"/{lower}":
                split = candidate
                break
        disks[disk] += 1
        prefixes[split][disk] += 1
        if len(examples[split]) < 3:
            examples[split].append(normalized)
        entries += 1
        offset = start + name_length + extra_length + comment_length

    print(f"entries={entries}")
    print(f"all_disks={dict(sorted(disks.items()))}")
    for split in sorted(prefixes):
        print(f"{split}: disks={dict(sorted(prefixes[split].items()))}")
        for example in examples[split]:
            print(f"  {example}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the central directory of a split ZIP final volume")
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    inspect(args.archive)


if __name__ == "__main__":
    main()
