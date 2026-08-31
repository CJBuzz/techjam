from __future__ import annotations
import argparse
from pathlib import Path
from aigc_detector.phase3.kaggle import write_kernel_metadata

def main():
    parser = argparse.ArgumentParser(description="Generate R7 search or locked-final Kaggle metadata")
    parser.add_argument("--job", choices=("r7_search", "r7_locked_final"), required=True)
    parser.add_argument("--job-dir", type=Path, required=True); parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--candidate-kernel-source", action="append", default=[])
    parser.add_argument("--data-dataset", action="append", default=[]); parser.add_argument("--model-dataset", action="append", default=[])
    parser.add_argument("--lock-dataset")
    args = parser.parse_args()
    if not args.candidate_kernel_source: raise ValueError("R7 requires at least one candidate kernel source")
    if args.job == "r7_locked_final" and not args.lock_dataset: raise ValueError("Locked final job requires --lock-dataset")
    datasets = [args.source_dataset, *args.data_dataset, *args.model_dataset]
    if args.lock_dataset: datasets.append(args.lock_dataset)
    path = write_kernel_metadata(args.job_dir, args.job.replace("_", "-"), datasets,
                                 args.candidate_kernel_source, [])
    print(path); print(f"kaggle kernels push -p '{args.job_dir}'")
if __name__ == "__main__": main()
