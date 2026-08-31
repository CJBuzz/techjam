from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from aigc_detector.phase3.artifacts import atomic_json
from aigc_detector.phase3.r2 import select_promotion_regime


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock the R2 100k regime using 25k validation summaries only")
    parser.add_argument("--single-summary", type=Path, required=True)
    parser.add_argument("--compound-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-clean", type=float, default=0.9681)
    args = parser.parse_args()
    single = json.loads(args.single_summary.read_text(encoding="utf-8"))
    compound = json.loads(args.compound_summary.read_text(encoding="utf-8"))
    selected = select_promotion_regime(single, compound, args.baseline_clean)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "promotion_config.json", selected)
    username = os.getenv("KAGGLE_USERNAME")
    if not username: raise ValueError("Set KAGGLE_USERNAME")
    metadata = {"title": "Track5 R2 Promotion Config", "id": f"{username}/track5-r2-promotion-config",
                "licenses": [{"name": "other"}]}
    atomic_json(args.output / "dataset-metadata.json", metadata)
    print(args.output / "promotion_config.json")
    print(f"kaggle datasets create -p '{args.output}' --dir-mode zip -r skip")


if __name__ == "__main__": main()
