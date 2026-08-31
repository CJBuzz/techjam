from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from aigc_detector.phase3.artifacts import atomic_json
from aigc_detector.phase3.r3 import select_promotion_setting


def main():
    parser = argparse.ArgumentParser(description="Select one R3 consistency setting using validation only")
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-clean", type=float, default=0.9681)
    args = parser.parse_args()
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in args.summary]
    selected = select_promotion_setting(documents, args.baseline_clean)
    username = os.getenv("KAGGLE_USERNAME")
    if not username: raise ValueError("Set KAGGLE_USERNAME")
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "promotion_config.json", selected)
    atomic_json(args.output / "dataset-metadata.json", {
        "title": "Track5 R3 Promotion Config", "id": f"{username}/track5-r3-promotion-config",
        "licenses": [{"name": "other"}],
    })
    print(args.output / "promotion_config.json")
    print(f"kaggle datasets create -p '{args.output}' --dir-mode zip -r skip")


if __name__ == "__main__": main()
