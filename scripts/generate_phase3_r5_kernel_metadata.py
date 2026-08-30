from __future__ import annotations

import argparse
from pathlib import Path

from aigc_detector.phase3.kaggle import write_kernel_metadata


JOBS = ("r5_low", "r5_high", "r5_ensemble")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate private offline Kaggle metadata for one R5 job")
    parser.add_argument("--job", choices=JOBS, required=True); parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True); parser.add_argument("--data-dataset")
    parser.add_argument("--model-dataset", action="append", default=[]); parser.add_argument("--r4-kernel-source")
    parser.add_argument("--low-kernel-source"); parser.add_argument("--high-kernel-source")
    args = parser.parse_args()
    if args.job in {"r5_low", "r5_high"} and (not args.data_dataset or not args.r4_kernel_source):
        raise ValueError("Expert jobs require --data-dataset and --r4-kernel-source")
    if args.job == "r5_ensemble" and (not args.low_kernel_source or not args.high_kernel_source):
        raise ValueError("Ensemble job requires both expert kernel sources")
    datasets = [args.source_dataset, *args.model_dataset]
    if args.data_dataset: datasets.append(args.data_dataset)
    kernels = ([args.r4_kernel_source] if args.r4_kernel_source else [])
    if args.low_kernel_source: kernels.append(args.low_kernel_source)
    if args.high_kernel_source: kernels.append(args.high_kernel_source)
    path = write_kernel_metadata(args.job_dir, args.job.replace("_", "-"), datasets, kernels, [])
    print(path); print(f"kaggle kernels push -p '{args.job_dir}'")


if __name__ == "__main__": main()
