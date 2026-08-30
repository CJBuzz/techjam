from __future__ import annotations

import argparse
from pathlib import Path

from aigc_detector.phase3.kaggle import write_kernel_metadata


JOBS = ("r4_class_balanced_25k", "r4_source_balanced_25k", "r4_source_quality_matched_25k", "r4_100k_promotion")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate private offline Kaggle metadata for one R4 job")
    parser.add_argument("--job", choices=JOBS, required=True); parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True); parser.add_argument("--data-dataset", required=True)
    parser.add_argument("--model-dataset", action="append", default=[]); parser.add_argument("--r3-kernel-source", required=True)
    parser.add_argument("--promotion-config-dataset")
    args = parser.parse_args()
    if args.job == "r4_100k_promotion" and not args.promotion_config_dataset:
        raise ValueError("Promotion job requires --promotion-config-dataset")
    datasets = [args.source_dataset, args.data_dataset, *args.model_dataset]
    if args.promotion_config_dataset: datasets.append(args.promotion_config_dataset)
    path = write_kernel_metadata(args.job_dir, args.job.replace("_", "-"), datasets,
                                 [args.r3_kernel_source], [])
    print(path); print(f"kaggle kernels push -p '{args.job_dir}'")


if __name__ == "__main__": main()
