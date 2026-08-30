import os, shutil, sys
from pathlib import Path
SOURCE = Path(os.getenv("PHASE3_SOURCE_DIR", "/kaggle/input/track5-phase3-source"))
if str(SOURCE) not in sys.path: sys.path.insert(0, str(SOURCE))
from aigc_detector.phase3.r5 import discover_output, run_ensemble
if __name__ == "__main__":
    output = Path(os.getenv("PHASE3_OUTPUT", "/kaggle/working/ensemble"))
    run_ensemble(discover_output("R5-low"), discover_output("R5-high"), output,
                 float(os.getenv("BASELINE_CLEAN_BACC", "0.9681")))
    for name in ("r5_summary.json", "r5_summary.csv", "recommended_candidate.json", "val_logits.npz"):
        source = output / name
        if source.is_file() and source.parent != Path("/kaggle/working"):
            shutil.copy2(source, Path("/kaggle/working") / name)
