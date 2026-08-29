from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image
from tqdm import tqdm

from aigc_detector.data import IMAGE_SUFFIXES


OFFICIAL_DIRECTORY = "https://www.grip.unina.it/download/prog/B-Free/extended_synthbuster/"


def parse_checksums(path: Path) -> dict[str, tuple[str, str]]:
    """Parse common sha256sum/md5sum and ``ALGO (file) = digest`` formats."""
    checksums: dict[str, tuple[str, str]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        standard = re.fullmatch(r"([0-9a-fA-F]{32}|[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\s+\*?(.+)", line)
        functional = re.fullmatch(
            r"(MD5|SHA1|SHA256)\s*\((.+)\)\s*=\s*([0-9a-fA-F]+)", line, re.IGNORECASE
        )
        if standard:
            digest, filename = standard.groups()
            algorithm = {32: "md5", 40: "sha1", 64: "sha256"}[len(digest)]
        elif functional:
            algorithm, filename, digest = functional.groups()
            algorithm = algorithm.lower()
        else:
            raise ValueError(f"Unsupported checksum line: {raw_line!r}")
        checksums[Path(filename.strip()).name] = (algorithm, digest.lower())
    return checksums


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archives(paths: list[Path], checksum_path: Path) -> dict[str, str]:
    expected = parse_checksums(checksum_path)
    verified = {}
    for path in paths:
        if path.name not in expected:
            raise ValueError(f"No checksum entry for {path.name} in {checksum_path}")
        algorithm, wanted = expected[path.name]
        actual = file_digest(path, algorithm)
        if actual != wanted:
            raise ValueError(f"Checksum mismatch for {path.name}: expected {wanted}, got {actual}")
        verified[path.name] = f"{algorithm}:{actual}"
    return verified


def safe_image_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for member in archive.infolist():
        normalized = member.filename.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive member: {member.filename!r}")
        if not member.is_dir() and Path(path.name).suffix.lower() in IMAGE_SUFFIXES:
            members.append(member)
    return members


def fake_generator(member_name: str) -> str | None:
    name = member_name.lower().replace(" ", "_")
    if "flux" in name:
        return "flux"
    if any(token in name for token in ("sd3", "sd_3", "sd-3", "stable_diffusion_3", "stable-diffusion-3")):
        return "sd35"
    return None


def extract_images(
    archive_path: Path,
    dataset_root: Path,
    fixed_source: tuple[str, str] | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    unknown: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = safe_image_members(archive)
        for member in tqdm(members, desc=f"Preparing {archive_path.name}"):
            if fixed_source:
                label_folder, source = fixed_source
            else:
                label_folder, source = "ai", fake_generator(member.filename)
                if source is None:
                    unknown.append(member.filename)
                    continue
            blob = archive.read(member)
            try:
                with Image.open(io.BytesIO(blob)) as image:
                    image.verify()
            except Exception as error:
                raise ValueError(f"Invalid image {member.filename!r} in {archive_path}") from error
            key = f"{label_folder}/{source}"
            index = counts.get(key, 0)
            suffix = Path(member.filename).suffix.lower()
            target = dataset_root / label_folder / source / f"{index:05d}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)  # Preserve original bytes and generator traces; do not re-encode.
            counts[key] = index + 1
    if unknown:
        examples = ", ".join(repr(item) for item in unknown[:3])
        raise ValueError(f"Could not identify the generator for {len(unknown)} images, e.g. {examples}")
    return counts


def prepare_dataset(
    real_archive: Path,
    fake_archive: Path,
    output: Path,
    checksum: Path | None = None,
    expected_per_source: int = 1000,
) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"Output must not already exist: {output}")
    for archive in (real_archive, fake_archive):
        if not archive.is_file():
            raise FileNotFoundError(archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    verified = verify_archives([real_archive, fake_archive], checksum) if checksum else {}
    with tempfile.TemporaryDirectory(prefix="bfree-prepare-", dir=output.parent) as temporary:
        dataset_root = Path(temporary) / "dataset"
        counts = {}
        counts.update(extract_images(real_archive, dataset_root, ("real", "raise")))
        counts.update(extract_images(fake_archive, dataset_root))
        expected = {"real/raise": expected_per_source, "ai/flux": expected_per_source, "ai/sd35": expected_per_source}
        if counts != expected:
            raise ValueError(f"Unexpected image counts: expected {expected}, got {counts}")
        manifest = {
            "dataset": "B-Free new-generators",
            "official_directory": OFFICIAL_DIRECTORY,
            "purpose": "external evaluation only until a new untouched generator benchmark is reserved",
            "counts": counts,
            "verified_checksums": verified,
            "archives": [real_archive.name, fake_archive.name],
            "images_reencoded": False,
        }
        (dataset_root / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        dataset_root.replace(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and arrange the manually downloaded B-Free new-generators archives"
    )
    parser.add_argument("--real-archive", type=Path, required=True, help="Path to real_RAISE_1k.zip")
    parser.add_argument("--fake-archive", type=Path, required=True, help="Path to sd3_flux.zip")
    parser.add_argument("--checksum", type=Path, help="Optional path to the official checksum.txt")
    parser.add_argument("--output", type=Path, default=Path("data/bfree_new_generators"))
    parser.add_argument("--expected-per-source", type=int, default=1000)
    args = parser.parse_args()
    manifest = prepare_dataset(
        args.real_archive, args.fake_archive, args.output, args.checksum, args.expected_per_source
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
