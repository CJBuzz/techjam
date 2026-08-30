from __future__ import annotations

import argparse
import json
from pathlib import Path

from aigc_detector.phase3.r1 import write_job_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine independent R1 Kaggle job summaries")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-clean", type=float, default=0.9681)
    args = parser.parse_args()
    rows = []
    for path in args.input:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("selection_split") != "validation" or document.get("final_test_evaluated") is not False:
            raise ValueError(f"R1 summary is not validation-only: {path}")
        rows.extend(document["results"])
    write_job_summary(rows, args.output, args.baseline_clean)


if __name__ == "__main__":
    main()
