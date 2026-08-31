from __future__ import annotations

import argparse
import json
from pathlib import Path


IGNORE_PATTERNS = [
    "data/",
    "artifacts/",
    ".venv/",
    ".uv-cache/",
    ".hf-cache/",
    ".torch-cache/",
    ".git/",
    "**/__pycache__/",
    "*.pyc",
    "*.pt",
    "*.log",
    ".env",
    ".env.*",
    "kaggle.json",
]
EXCLUDED_TOP_LEVEL = {
    "data", "artifacts", ".venv", ".uv-cache", ".hf-cache", ".torch-cache", ".git"
}


def source_summary(root: Path) -> dict[str, object]:
    root = root.resolve()
    required = (root / "pyproject.toml", root / "aigc_detector", root / "scripts")
    if not all(path.exists() for path in required):
        raise FileNotFoundError(f"Not a detector repository root: {root}")
    files = [
        path for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).parts[0] not in EXCLUDED_TOP_LEVEL
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pt", ".log"}
        and path.name not in {".env", "kaggle.json"}
        and not path.name.startswith(".env.")
    ]
    return {
        "repo_root": str(root),
        "included_files": len(files),
        "included_bytes": sum(path.stat().st_size for path in files),
        "ignored_patterns": IGNORE_PATTERNS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload detector source as a small Kaggle Dataset")
    parser.add_argument("--handle", required=True, help="Kaggle handle: username/dataset-slug")
    parser.add_argument("--repo-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--version-notes", default="Detector source for Kaggle GPU training")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--login", action="store_true",
        help="Prompt for a Kaggle token and keep it in this process through the upload",
    )
    args = parser.parse_args()

    print(json.dumps(source_summary(args.repo_dir), indent=2))
    if args.dry_run:
        print("Dry run only; nothing was uploaded.")
        return
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "Run with: uv run --with kagglehub python scripts/kaggle/repository.py ..."
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
        str(args.repo_dir.resolve()),
        version_notes=args.version_notes,
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"Uploaded source dataset {args.handle}. Verify that it is Private in Kaggle.")


if __name__ == "__main__":
    main()
