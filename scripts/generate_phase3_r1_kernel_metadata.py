from __future__ import annotations

import argparse
from pathlib import Path

from aigc_detector.phase3.kaggle import write_kernel_metadata


JOBS = ("dinov3_vitl16", "siglip2_large_256", "siglip2_so400m_256")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one private offline Kaggle kernel config per R1 backbone")
    parser.add_argument("--job", choices=JOBS, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True, help="username/track5-phase3-source")
    parser.add_argument("--data-dataset", required=True, help="username/Track-5 train+validation manifest/images")
    parser.add_argument("--model-dataset", help="username/offline-model-assets dataset")
    parser.add_argument("--model-source", help="Optional native Kaggle model reference instead of a dataset")
    args = parser.parse_args()
    if bool(args.model_dataset) == bool(args.model_source):
        raise ValueError("Provide exactly one of --model-dataset or --model-source")
    datasets = [args.source_dataset, args.data_dataset] + ([args.model_dataset] if args.model_dataset else [])
    path = write_kernel_metadata(args.job_dir, f"r1-{args.job.replace('_', '-')}", datasets, [],
                                 [args.model_source] if args.model_source else [])
    print(path)
    print(f"kaggle kernels push -p '{args.job_dir}'")


if __name__ == "__main__":
    main()
