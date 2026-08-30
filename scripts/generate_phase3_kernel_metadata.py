from __future__ import annotations

import argparse
from pathlib import Path

from aigc_detector.phase3.kaggle import write_kernel_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate private offline Kaggle kernel metadata")
    parser.add_argument("--experiment", choices=[f"r{i}" for i in range(1, 8)], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-source", action="append", default=[])
    parser.add_argument("--kernel-source", action="append", default=[])
    parser.add_argument("--model-source", action="append", default=[])
    args = parser.parse_args()
    path = write_kernel_metadata(args.output, args.experiment, args.dataset_source,
                                 args.kernel_source, args.model_source)
    print(path)


if __name__ == "__main__":
    main()
