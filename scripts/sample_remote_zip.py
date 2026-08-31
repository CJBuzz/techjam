from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"


@dataclass(frozen=True)
class ZipEntry:
    name: str
    flags: int
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int


def request_range(url: str, start: int, end: int) -> bytes:
    if start < 0 or end < start:
        raise ValueError(f"Invalid HTTP byte range {start}-{end}")
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "techjam-wildfake-sampler/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
        content_range = response.headers.get("Content-Range")
        if response.status != 206 or not content_range:
            raise RuntimeError(f"Server ignored byte range {start}-{end}; status={response.status}")
    expected = end - start + 1
    if len(payload) != expected:
        raise RuntimeError(f"Range {start}-{end} returned {len(payload)} bytes, expected {expected}")
    return payload


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "techjam-wildfake-sampler/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        length = response.headers.get("Content-Length")
        if not length or "bytes" not in response.headers.get("Accept-Ranges", "").lower():
            raise RuntimeError("Remote archive does not advertise byte ranges and a content length")
        return int(length)


def _zip64_value(extra: bytes, needs: tuple[bool, bool, bool, bool]) -> tuple[int | None, ...]:
    cursor = 0
    while cursor + 4 <= len(extra):
        kind, size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        value = extra[cursor : cursor + size]
        cursor += size
        if kind != 0x0001:
            continue
        offset = 0
        result: list[int | None] = []
        for index, needed in enumerate(needs):
            if not needed:
                result.append(None)
                continue
            width = 4 if index == 3 else 8
            if offset + width > len(value):
                raise ValueError("Truncated Zip64 central-directory extra field")
            result.append(int.from_bytes(value[offset : offset + width], "little"))
            offset += width
        return tuple(result)
    raise ValueError("Missing required Zip64 central-directory extra field")


def central_directory(url: str, cache_dir: Path) -> tuple[list[ZipEntry], dict[str, int]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(url.encode()).hexdigest()[:20]
    binary_path = cache_dir / f"{cache_key}.central.bin"
    metadata_path = cache_dir / f"{cache_key}.json"
    if binary_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        central = binary_path.read_bytes()
    else:
        size = remote_size(url)
        tail_start = max(0, size - 131_072)
        tail = request_range(url, tail_start, size - 1)
        eocd_relative = tail.rfind(EOCD_SIGNATURE)
        if eocd_relative < 0 or eocd_relative + 22 > len(tail):
            raise ValueError("ZIP end-of-central-directory record was not found")
        eocd_offset = tail_start + eocd_relative
        _, disk, central_disk, entries_disk, entries_total, central_size, central_offset, _ = struct.unpack_from(
            "<4s4H2LH", tail, eocd_relative
        )
        if disk != 0 or central_disk != 0:
            raise ValueError("Multi-disk ZIP archives are not supported")
        if entries_total == 0xFFFF or central_size == 0xFFFFFFFF or central_offset == 0xFFFFFFFF:
            locator_offset = eocd_offset - 20
            locator = request_range(url, locator_offset, eocd_offset - 1)
            signature, zip64_disk, zip64_offset, disks = struct.unpack("<4sLQL", locator)
            if signature != ZIP64_LOCATOR_SIGNATURE or zip64_disk != 0 or disks != 1:
                raise ValueError("Invalid Zip64 locator")
            record = request_range(url, zip64_offset, zip64_offset + 55)
            unpacked = struct.unpack("<4sQ2H2L4Q", record)
            if unpacked[0] != ZIP64_EOCD_SIGNATURE or unpacked[4] != 0 or unpacked[5] != 0:
                raise ValueError("Invalid Zip64 end-of-central-directory record")
            entries_disk, entries_total, central_size, central_offset = unpacked[6:10]
        if entries_disk != entries_total:
            raise ValueError("Central-directory entry count mismatch")
        central = request_range(url, central_offset, central_offset + central_size - 1)
        binary_path.write_bytes(central)
        metadata = {
            "url": url,
            "archive_size": size,
            "entries": int(entries_total),
            "central_size": int(central_size),
            "central_offset": int(central_offset),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    entries: list[ZipEntry] = []
    cursor = 0
    while cursor < len(central):
        if central[cursor : cursor + 4] != CENTRAL_SIGNATURE or cursor + 46 > len(central):
            raise ValueError(f"Invalid central-directory record at byte {cursor}")
        values = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        flags, method, crc32 = values[3], values[4], values[7]
        compressed_size, uncompressed_size = values[8], values[9]
        name_size, extra_size, comment_size, disk_start = values[10], values[11], values[12], values[13]
        local_offset = values[16]
        name_bytes = central[cursor + 46 : cursor + 46 + name_size]
        extra = central[cursor + 46 + name_size : cursor + 46 + name_size + extra_size]
        needs = (
            uncompressed_size == 0xFFFFFFFF,
            compressed_size == 0xFFFFFFFF,
            local_offset == 0xFFFFFFFF,
            disk_start == 0xFFFF,
        )
        if any(needs):
            zip64_uncompressed, zip64_compressed, zip64_offset, zip64_disk = _zip64_value(extra, needs)
            uncompressed_size = zip64_uncompressed if zip64_uncompressed is not None else uncompressed_size
            compressed_size = zip64_compressed if zip64_compressed is not None else compressed_size
            local_offset = zip64_offset if zip64_offset is not None else local_offset
            disk_start = zip64_disk if zip64_disk is not None else disk_start
        if disk_start != 0:
            raise ValueError("Multi-disk ZIP entry is not supported")
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = name_bytes.decode(encoding)
        entries.append(ZipEntry(name, flags, method, crc32, int(compressed_size), int(uncompressed_size), int(local_offset)))
        cursor += 46 + name_size + extra_size + comment_size
    if len(entries) != int(metadata["entries"]):
        raise ValueError(f"Parsed {len(entries)} entries; metadata expected {metadata['entries']}")
    return entries, metadata


def allowed_paths(metadata_csv: Path, architecture: str, label: int) -> dict[str, str]:
    variants: dict[str, str] = {}
    collisions: set[str] = set()
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["Architecture"].casefold() != architecture.casefold() or int(row["IsFake"]) != label:
                continue
            original = row["Image_path"].removeprefix("./").replace("\\", "/")
            parts = PurePosixPath(original).parts
            for drop in range(min(4, len(parts))):
                variant = "/".join(parts[drop:])
                previous = variants.get(variant)
                if previous is not None and previous != original:
                    collisions.add(variant)
                else:
                    variants[variant] = original
    for variant in collisions:
        variants.pop(variant, None)
    return variants


def match_entries(entries: list[ZipEntry], variants: dict[str, str]) -> list[tuple[ZipEntry, str]]:
    matched = []
    for entry in entries:
        normalized = entry.name.removeprefix("./").replace("\\", "/")
        if PurePosixPath(normalized).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = PurePosixPath(normalized).parts
        original = None
        for drop in range(min(4, len(parts))):
            original = variants.get("/".join(parts[drop:]))
            if original is not None:
                break
        if original is not None:
            matched.append((entry, original))
    return matched


def extract_window(url: str, selected: list[tuple[ZipEntry, str]], output: Path, source: str, label: int) -> list[dict[str, object]]:
    first_offset = min(entry.local_offset for entry, _ in selected)
    last_entry = max((entry for entry, _ in selected), key=lambda item: item.local_offset)
    header = request_range(url, last_entry.local_offset, last_entry.local_offset + 29)
    if header[:4] != LOCAL_SIGNATURE:
        raise ValueError(f"Invalid local header for {last_entry.name}")
    name_size, extra_size = struct.unpack_from("<HH", header, 26)
    last_end = last_entry.local_offset + 30 + name_size + extra_size + last_entry.compressed_size - 1
    block = request_range(url, first_offset, last_end)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (entry, official_path) in enumerate(selected):
        relative = entry.local_offset - first_offset
        if block[relative : relative + 4] != LOCAL_SIGNATURE:
            raise ValueError(f"Invalid local header for {entry.name}")
        local_flags, local_method = struct.unpack_from("<HH", block, relative + 6)
        local_name_size, local_extra_size = struct.unpack_from("<HH", block, relative + 26)
        if local_flags & 0x1:
            raise ValueError(f"Encrypted ZIP entry is unsupported: {entry.name}")
        data_start = relative + 30 + local_name_size + local_extra_size
        compressed = block[data_start : data_start + entry.compressed_size]
        if len(compressed) != entry.compressed_size:
            raise ValueError(f"Truncated compressed payload for {entry.name}")
        if local_method == 0:
            payload = compressed
        elif local_method == 8:
            payload = zlib.decompress(compressed, -15)
        else:
            raise ValueError(f"Unsupported ZIP compression method {local_method}: {entry.name}")
        if len(payload) != entry.uncompressed_size or zlib.crc32(payload) & 0xFFFFFFFF != entry.crc32:
            raise ValueError(f"CRC or size mismatch for {entry.name}")
        suffix = PurePosixPath(entry.name).suffix.lower()
        digest = hashlib.sha256(payload).hexdigest()
        destination = output / f"{index:06d}-{digest[:12]}{suffix}"
        destination.write_bytes(payload)
        with Image.open(destination) as image:
            image.verify()
        records.append({
            "path": destination.as_posix(),
            "label": label,
            "source": source,
            "archive_member": entry.name,
            "official_train_path": official_path,
            "content_sha256": digest,
            "crc32": f"{entry.crc32:08x}",
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Range-extract an audited official-train subset from a remote ZIP")
    parser.add_argument("--url", required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--label", type=int, choices=(0, 1), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/wildfake_remote_cache"))
    args = parser.parse_args()

    entries, archive_metadata = central_directory(args.url, args.cache_dir)
    variants = allowed_paths(args.metadata_csv, args.architecture, args.label)
    matched = match_entries(entries, variants)
    if len(matched) < args.count:
        raise ValueError(f"Only {len(matched)} official training entries matched; requested {args.count}")
    rng = random.Random(args.seed)
    start = rng.randrange(0, len(matched) - args.count + 1)
    selected = matched[start : start + args.count]
    records = extract_window(args.url, selected, args.output, args.source, args.label)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "archive": archive_metadata,
        "architecture": args.architecture,
        "source": args.source,
        "label": args.label,
        "seed": args.seed,
        "matched_official_train_entries": len(matched),
        "selected_start": start,
        "records": records,
    }
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Extracted and verified {len(records)} {args.source} images from {len(matched)} official training entries")


if __name__ == "__main__":
    main()
