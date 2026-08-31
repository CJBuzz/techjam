from __future__ import annotations

import argparse
from pathlib import Path

from aigc_detector.phase3.kaggle import write_kernel_metadata


JOBS = ("r2_25k_single", "r2_25k_compound", "r2_100k_promotion")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate private offline Kaggle metadata for one R2 job")
    parser.add_argument("--job", choices=JOBS, required=True); parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True); parser.add_argument("--data-dataset", required=True)
    parser.add_argument("--model-dataset", action="append", default=[])
    parser.add_argument("--r1-kernel-source", required=True)
    parser.add_argument("--promotion-config-dataset")
    args = parser.parse_args()
    if args.job == "r2_100k_promotion" and not args.promotion_config_dataset:
        raise ValueError("Promotion job requires --promotion-config-dataset")
    datasets = [args.source_dataset, args.data_dataset, *args.model_dataset]
    if args.promotion_config_dataset: datasets.append(args.promotion_config_dataset)
    path = write_kernel_metadata(args.job_dir, args.job.replace("_", "-"), datasets,
                                 [args.r1_kernel_source], [])
    print(path); print(f"kaggle kernels push -p '{args.job_dir}'")


if __name__ == "__main__": main()
