from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
AUDITED_SPLITS = {"train", "model_selection", "calibration"}


def pixel_hash(image: Image.Image) -> str:
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    try:
        payload = rgb.width.to_bytes(4, "big") + rgb.height.to_bytes(4, "big") + rgb.tobytes()
        return hashlib.sha256(payload).hexdigest()
    finally:
        if rgb is not image:
            rgb.close()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_file(path: Path) -> tuple[str, str]:
    with Image.open(path) as image:
        return str(path), pixel_hash(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit decoded-pixel overlap with the TechJam demonstration set")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--demo-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with args.split_manifest.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [row for row in all_rows if row["split"] in AUDITED_SPLITS]
    if not rows:
        raise ValueError("No train/model-selection/calibration rows found")

    provenance_terms = ("coco", "dall", "dalle", "dall-e", "val2017", "advanced")
    provenance_hits = [
        row for row in rows
        if any(term in (row.get("upstream_identifier", "") + " " + row.get("source", "")).lower()
               for term in provenance_terms)
    ]

    demo_by_hash: dict[str, list[str]] = defaultdict(list)
    demo_failures = []
    demo_classes = Counter()
    with zipfile.ZipFile(args.demo_zip) as archive:
        members = [
            info for info in archive.infolist()
            if not info.is_dir() and Path(info.filename).suffix.lower() in IMAGE_SUFFIXES
        ]
        for info in tqdm(members, desc="hash demonstration ZIP"):
            try:
                with archive.open(info) as raw, Image.open(raw) as image:
                    digest = pixel_hash(image)
                demo_by_hash[digest].append(info.filename)
                lowered = info.filename.lower()
                demo_classes["dalle_advanced" if "dalle" in lowered else "coco_val2017" if "coco" in lowered else "other"] += 1
            except Exception as error:  # record every unreadable supplied member
                demo_failures.append({"member": info.filename, "error": repr(error)})

    paths = [args.data_root / row["path"] for row in rows]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest references {len(missing)} missing files; first: {missing[0]}")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        recomputed = dict(tqdm(
            executor.map(hash_file, paths), total=len(paths), desc="rehash audited splits"
        ))

    stored_hash_mismatches = []
    overlaps = []
    for row, path in zip(rows, paths, strict=True):
        digest = recomputed[str(path)]
        if digest != row["content_sha256"]:
            stored_hash_mismatches.append({
                "path": row["path"], "split": row["split"],
                "stored": row["content_sha256"], "recomputed": digest,
            })
        if digest in demo_by_hash:
            overlaps.append({
                "dataset_path": row["path"],
                "split": row["split"],
                "source": row["source"],
                "upstream_identifier": row["upstream_identifier"],
                "decoded_pixel_sha256": digest,
                "demo_members": demo_by_hash[digest],
            })

    report = {
        "verdict": "FAIL" if overlaps or demo_failures or stored_hash_mismatches else "PASS",
        "challenge_compliant_exact_pixel_audit": not overlaps and not demo_failures and not stored_hash_mismatches,
        "manifest": {
            "path": str(args.split_manifest.resolve()),
            "sha256": file_sha256(args.split_manifest),
            "total_rows": len(all_rows),
            "audited_rows": len(rows),
            "audited_split_counts": dict(Counter(row["split"] for row in rows)),
            "audited_source_counts": dict(Counter(row["source"] for row in rows)),
            "provenance_name_hits": len(provenance_hits),
            "provenance_hit_rows": provenance_hits,
        },
        "demonstration_archive": {
            "path": str(args.demo_zip.resolve()),
            "size_bytes": args.demo_zip.stat().st_size,
            "sha256": file_sha256(args.demo_zip),
            "image_members": len(members),
            "unique_decoded_pixel_hashes": len(demo_by_hash),
            "class_counts_from_paths": dict(demo_classes),
            "decode_failures": demo_failures,
        },
        "stored_manifest_hash_mismatches": stored_hash_mismatches,
        "exact_decoded_pixel_overlaps": overlaps,
        "overlap_count": len(overlaps),
        "policy": "Any overlap or incomplete decode/hash verification is a failure.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": report["verdict"],
        "audited_rows": len(rows),
        "demo_images": len(members),
        "demo_unique_hashes": len(demo_by_hash),
        "overlaps": len(overlaps),
        "hash_mismatches": len(stored_hash_mismatches),
        "decode_failures": len(demo_failures),
        "provenance_name_hits": len(provenance_hits),
        "report": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
