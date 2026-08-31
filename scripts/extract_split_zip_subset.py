from __future__ import annotations

import argparse
import binascii
import struct
import zlib
from pathlib import Path, PurePosixPath


CENTRAL_HEADER = b"PK\x01\x02"
LOCAL_HEADER = b"PK\x03\x04"


def zip64_values(extra: bytes, uncomp: int, comp: int, local: int, disk: int) -> tuple[int, int, int, int]:
    cursor = 0
    while cursor + 4 <= len(extra):
        field_id, length = struct.unpack_from("<HH", extra, cursor)
        field = extra[cursor + 4 : cursor + 4 + length]
        cursor += 4 + length
        if field_id != 1:
            continue
        position = 0
        values = [uncomp, comp, local, disk]
        sizes = [8, 8, 8, 4]
        sentinels = [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFF]
        for index, (value, size, sentinel) in enumerate(zip(values, sizes, sentinels)):
            if value == sentinel:
                values[index] = int.from_bytes(field[position : position + size], "little")
                position += size
        return tuple(values)  # type: ignore[return-value]
    return uncomp, comp, local, disk


def central_entries(final_volume: Path) -> list[dict[str, int | str]]:
    payload = final_volume.read_bytes()
    offset = payload.find(CENTRAL_HEADER)
    if offset < 0:
        raise ValueError(f"No ZIP central directory found in {final_volume}")
    entries = []
    while payload[offset : offset + 4] == CENTRAL_HEADER:
        header = payload[offset : offset + 46]
        if len(header) != 46:
            raise ValueError("Truncated central-directory header")
        flags, method = struct.unpack_from("<HH", header, 8)
        crc, comp, uncomp = struct.unpack_from("<III", header, 16)
        name_len, extra_len, comment_len, disk = struct.unpack_from("<HHHH", header, 28)
        local = struct.unpack_from("<I", header, 42)[0]
        start = offset + 46
        name_bytes = payload[start : start + name_len]
        extra = payload[start + name_len : start + name_len + extra_len]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = name_bytes.decode(encoding)
        uncomp, comp, local, disk = zip64_values(extra, uncomp, comp, local, disk)
        entries.append({
            "name": name,
            "flags": flags,
            "method": method,
            "crc": crc,
            "compressed": comp,
            "uncompressed": uncomp,
            "disk": disk,
            "local": local,
        })
        offset = start + name_len + extra_len + comment_len
    return entries


def read_across(volumes: dict[int, Path], disk: int, offset: int, count: int) -> bytes:
    chunks = []
    remaining = count
    current_disk = disk
    current_offset = offset
    while remaining:
        path = volumes.get(current_disk)
        if path is None:
            raise FileNotFoundError(f"Missing split volume for disk {current_disk}")
        available = path.stat().st_size - current_offset
        if available <= 0:
            current_disk += 1
            current_offset = 0
            continue
        take = min(remaining, available)
        with path.open("rb") as handle:
            handle.seek(current_offset)
            chunk = handle.read(take)
        if len(chunk) != take:
            raise ValueError(f"Short read from {path}")
        chunks.append(chunk)
        remaining -= take
        current_disk += 1
        current_offset = 0
    return b"".join(chunks)


def extract(final_volume: Path, volumes: dict[int, Path], output: Path) -> None:
    selected = []
    for entry in central_entries(final_volume):
        normalized = str(entry["name"]).replace("\\", "/")
        lower = normalized.lower()
        label = "ai" if "/val/ai/" in f"/{lower}" else "real" if "/val/nature/" in f"/{lower}" else None
        if label and not normalized.endswith("/"):
            selected.append((entry, label, normalized))
    if not selected:
        raise ValueError("No validation images found")

    counts = {"ai": 0, "real": 0}
    output.mkdir(parents=True, exist_ok=False)
    try:
        for entry, label, normalized in selected:
            path = PurePosixPath(normalized)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe member path: {normalized!r}")
            disk, local = int(entry["disk"]), int(entry["local"])
            local_header = read_across(volumes, disk, local, 30)
            if local_header[:4] != LOCAL_HEADER:
                raise ValueError(f"Invalid local header for {normalized!r} on disk {disk} at {local}")
            name_len, extra_len = struct.unpack_from("<HH", local_header, 26)
            data_offset = local + 30 + name_len + extra_len
            compressed = read_across(volumes, disk, data_offset, int(entry["compressed"]))
            method = int(entry["method"])
            if method == 0:
                data = compressed
            elif method == 8:
                data = zlib.decompress(compressed, -15)
            else:
                raise ValueError(f"Unsupported ZIP compression method {method} for {normalized!r}")
            if len(data) != int(entry["uncompressed"]):
                raise ValueError(f"Size mismatch for {normalized!r}")
            if binascii.crc32(data) & 0xFFFFFFFF != int(entry["crc"]):
                raise ValueError(f"CRC mismatch for {normalized!r}")
            suffix = Path(path.name).suffix.lower()
            target = output / label / "glide" / f"{counts[label]:06d}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            counts[label] += 1
        print(f"extracted={counts}")
    except Exception:
        # Leave the partial directory in place for forensic inspection; never
        # silently present it as a completed evaluation dataset.
        (output / "INCOMPLETE").write_text("Extraction failed before validation completed.\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract validation images from selected split-ZIP volumes")
    parser.add_argument("--final-volume", type=Path, required=True)
    parser.add_argument("--volume", action="append", required=True, help="DISK=PATH; repeat for every required disk")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    volumes = {}
    for value in args.volume:
        disk, path = value.split("=", 1)
        volumes[int(disk)] = Path(path)
    extract(args.final_volume, volumes, args.output)


if __name__ == "__main__":
    main()
