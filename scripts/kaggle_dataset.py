from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


REQUIRED_COLUMNS = {"path", "label", "split"}
REQUIRED_SPLITS = {"train", "model_selection", "calibration", "test"}


def validate_dataset_root(root: Path, check_images: bool = True) -> dict[str, object]:
    root = root.resolve()
    manifest = root / "split_manifest.csv"
    audit = root / "audit.json"
    if not manifest.is_file() or not audit.is_file():
        raise FileNotFoundError(
            f"Expected split_manifest.csv and audit.json directly under {root}"
        )

    split_counts: Counter[str] = Counter()
    missing: list[str] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or ()):
            raise ValueError(
                f"Manifest must contain {sorted(REQUIRED_COLUMNS)}; got {reader.fieldnames}"
            )
        for row in reader:
            split_counts[row["split"]] += 1
            if check_images and len(missing) < 20 and not (root / row["path"]).is_file():
                missing.append(row["path"])
    if set(split_counts) != REQUIRED_SPLITS:
        raise ValueError(
            f"Manifest splits must be {sorted(REQUIRED_SPLITS)}; got {sorted(split_counts)}"
        )
    if missing:
        raise FileNotFoundError(f"Manifest images are missing under {root}: {missing[:5]}")

    return {
        "dataset_root": str(root),
        "manifest": str(manifest),
        "audit": str(audit),
        "split_counts": dict(split_counts),
        "images_checked": check_images,
    }


def locate_dataset_root(downloaded: Path) -> Path:
    candidates = sorted(path.parent for path in downloaded.rglob("split_manifest.csv"))
    candidates = [path for path in candidates if (path / "audit.json").is_file()]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one audited dataset root below {downloaded}; found {candidates}"
        )
    return candidates[0]


def upload(args: argparse.Namespace) -> None:
    summary = validate_dataset_root(args.data_dir, check_images=not args.skip_image_check)
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        print("Dry run only; nothing was uploaded.")
        return
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "Run with: uv run --with kagglehub python scripts/kaggle_dataset.py ..."
        ) from error
    if args.login:
        kagglehub.login()
    try:
        identity = kagglehub.whoami()
    except Exception as error:
        raise RuntimeError(
            "Kaggle authentication is unavailable in this process. Re-run with --login, or set "
            "KAGGLE_API_TOKEN / ~/.kaggle/access_token before starting the upload."
        ) from error
    username = str(identity["username"])
    handle_owner = args.handle.split("/", 1)[0]
    if username.casefold() != handle_owner.casefold():
        raise ValueError(
            f"Authenticated as {username!r}, but the Dataset handle is owned by {handle_owner!r}. "
            f"Use --handle {username}/YOUR_DATASET_SLUG."
        )
    print(f"Authenticated with Kaggle as: {username}")
    kagglehub.dataset_upload(
        args.handle,
        str(args.data_dir.resolve()),
        version_notes=args.version_notes,
    )
    print(f"Uploaded {args.handle}. Keep the Kaggle Dataset private unless its source licenses permit redistribution.")


def validate(args: argparse.Namespace) -> None:
    print(json.dumps(validate_dataset_root(args.data_dir, check_images=not args.skip_image_check), indent=2))


def download(args: argparse.Namespace) -> None:
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "Run with: uv run --with kagglehub python scripts/kaggle_dataset.py ..."
        ) from error
    kwargs: dict[str, object] = {"force_download": args.force}
    if args.output_dir:
        kwargs["output_dir"] = str(args.output_dir)
    downloaded = Path(kagglehub.dataset_download(args.handle, **kwargs))
    root = locate_dataset_root(downloaded)
    summary = validate_dataset_root(root, check_images=not args.skip_image_check)
    print(json.dumps(summary, indent=2))
    print(f"DATA_ROOT={root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, upload, or mount/download the audited mixed-100K corpus with KaggleHub"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload", help="Upload the local corpus as a Kaggle Dataset")
    upload_parser.add_argument("--handle", required=True, help="Kaggle handle: username/dataset-slug")
    upload_parser.add_argument("--data-dir", type=Path, default=Path("data/mixed_100k"))
    upload_parser.add_argument(
        "--version-notes", default="Audited duplicate-grouped mixed 100K corpus, seed 42"
    )
    upload_parser.add_argument("--skip-image-check", action="store_true")
    upload_parser.add_argument("--dry-run", action="store_true")
    upload_parser.add_argument(
        "--login", action="store_true",
        help="Prompt for a Kaggle token and keep it in this process through the upload",
    )
    upload_parser.set_defaults(func=upload)

    validate_parser = subparsers.add_parser("validate", help="Validate a local or mounted corpus")
    validate_parser.add_argument("--data-dir", type=Path, required=True)
    validate_parser.add_argument("--skip-image-check", action="store_true")
    validate_parser.set_defaults(func=validate)

    download_parser = subparsers.add_parser(
        "download", help="Mount/download a Kaggle Dataset and print its validated data root"
    )
    download_parser.add_argument("--handle", required=True, help="Kaggle handle: username/dataset-slug")
    download_parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Optional copy destination. Omit inside Kaggle to use the read-only mounted cache.",
    )
    download_parser.add_argument("--force", action="store_true")
    download_parser.add_argument("--skip-image-check", action="store_true")
    download_parser.set_defaults(func=download)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
