from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_evaluation(arguments: list[str]) -> None:
    subprocess.run([sys.executable, "-m", "aigc_detector.evaluate", *arguments], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen-checkpoint in-domain test and B-Free unseen-generator evaluation"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/diverse_initialized_40k_calibrated.pt")
    )
    parser.add_argument("--local-data", type=Path, default=Path("data/mixed_5k"))
    parser.add_argument("--external-data", type=Path, default=Path("data/bfree_new_generators"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/generalization"))
    parser.add_argument("--profile", choices=("full", "worst"), default="full")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for path in (args.checkpoint, args.local_data, args.external_data):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_output = args.output_dir / "stage1_local_test.json"
    local_errors = args.output_dir / "stage1_local_test_errors.json"
    external_output = args.output_dir / "stage2_bfree_generators.json"
    external_errors = args.output_dir / "stage2_bfree_generators_errors.json"
    common = [
        "--checkpoint", str(args.checkpoint),
        "--profile", args.profile,
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--seed", str(args.seed),
        "--top-errors", "50",
    ]
    run_evaluation([
        "--data-dir", str(args.local_data),
        "--split", "test",
        "--protocol", "standard",
        "--output", str(local_output),
        "--error-analysis-output", str(local_errors),
        *common,
    ])
    run_evaluation([
        "--data-dir", str(args.external_data),
        "--split", "all",
        "--protocol", "paired-generators",
        "--output", str(external_output),
        "--error-analysis-output", str(external_errors),
        *common,
    ])
    local = json.loads(local_output.read_text(encoding="utf-8"))
    external = json.loads(external_output.read_text(encoding="utf-8"))
    summary = {
        "checkpoint": str(args.checkpoint),
        "selection_policy": "checkpoint, calibration, and threshold frozen before both stages",
        "stage1_in_domain_test": local["_robust_summary"],
        "stage2_unseen_generators": external["_generalization_summary"],
        "stage2_clean_generator_pairs": external["clean"]["paired_generators"],
        "reports": {
            "stage1": str(local_output),
            "stage2": str(external_output),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
