from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from aigc_detector.phase3.kaggle import dataset_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an offline Hugging Face snapshot as a private Kaggle dataset")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot-source", type=Path, help="Existing complete local Hugging Face snapshot")
    source.add_argument("--model-id", help="Hugging Face model id; requires explicit --allow-download")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--version-message", default="Update offline pretrained model snapshot")
    args = parser.parse_args()
    username = os.getenv("KAGGLE_USERNAME")
    if not username:
        raise ValueError("Set KAGGLE_USERNAME")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    if args.snapshot_source:
        shutil.copytree(args.snapshot_source, args.output)
        model_id = None
    else:
        if not args.allow_download:
            raise ValueError("--model-id requires explicit --allow-download; Kaggle runtime remains offline")
        from huggingface_hub import snapshot_download
        snapshot_download(args.model_id, revision=args.revision, local_dir=args.output)
        model_id = args.model_id
    metadata = dataset_metadata(f"{username}/{args.slug}", f"Offline model assets: {args.slug}")
    (args.output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.output / "asset-provenance.json").write_text(json.dumps({
        "model_id": model_id, "revision": args.revision, "kaggle_runtime_internet_required": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"kaggle datasets create -p '{args.output}' --dir-mode zip -r skip")
    print(f"kaggle datasets version -p '{args.output}' -m '{args.version_message}' --dir-mode zip -r skip")


if __name__ == "__main__":
    main()
