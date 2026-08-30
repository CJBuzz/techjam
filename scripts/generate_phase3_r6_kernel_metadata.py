from __future__ import annotations
import argparse
from pathlib import Path
from aigc_detector.phase3.kaggle import write_kernel_metadata

JOBS = ("r6_global_only", "r6_mean_patch", "r6_topk_patch", "r6_attention_pool", "r6_global_plus_local", "r6_promotion")
def main():
    parser = argparse.ArgumentParser(description="Generate offline R6 Kaggle metadata")
    parser.add_argument("--job", choices=JOBS, required=True); parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True); parser.add_argument("--data-dataset", required=True)
    parser.add_argument("--model-dataset", required=True); parser.add_argument("--r4-kernel-source", required=True)
    parser.add_argument("--selection-dataset")
    args = parser.parse_args()
    if args.job in {"r6_global_plus_local", "r6_promotion"} and not args.selection_dataset:
        raise ValueError("Selected R6 job requires --selection-dataset")
    datasets = [args.source_dataset, args.data_dataset, args.model_dataset]
    if args.selection_dataset: datasets.append(args.selection_dataset)
    path = write_kernel_metadata(args.job_dir, args.job.replace("_", "-"), datasets, [args.r4_kernel_source], [])
    print(path); print(f"kaggle kernels push -p '{args.job_dir}'")
if __name__ == "__main__": main()
