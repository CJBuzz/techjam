from __future__ import annotations

import argparse
import json
from pathlib import Path

from aigc_detector.phase3.artifacts import atomic_json
from aigc_detector.phase3.r4 import select_promotion_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Select exactly one R4 policy using validation summaries")
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--baseline-clean", type=float, default=0.9681); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summary]
    selected = select_promotion_policy(summaries, args.baseline_clean)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "promotion_config.json", selected)
    atomic_json(args.output / "dataset-metadata.json", {"title": "Track5 R4 promotion config",
                "id": "REPLACE_USERNAME/track5-r4-promotion-config", "licenses": [{"name": "CC0-1.0"}]})
    print(args.output / "promotion_config.json")


if __name__ == "__main__": main()
