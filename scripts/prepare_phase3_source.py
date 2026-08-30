from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from aigc_detector.phase3.kaggle import dataset_metadata, package_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an offline private Kaggle source dataset")
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slug", default="track5-phase3-source")
    parser.add_argument("--version-message", default="Track-5 Phase-3 source update")
    args = parser.parse_args()
    username = os.getenv("KAGGLE_USERNAME")
    if not username:
        raise ValueError("Set KAGGLE_USERNAME")
    package_source(args.source.resolve(), args.output.resolve())
    metadata = dataset_metadata(f"{username}/{args.slug}", "Track-5 Phase-3 Source")
    (args.output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"kaggle datasets create -p '{args.output}' --dir-mode zip -r skip")
    print(f"kaggle datasets version -p '{args.output}' -m '{args.version_message}' --dir-mode zip -r skip")


if __name__ == "__main__":
    main()
