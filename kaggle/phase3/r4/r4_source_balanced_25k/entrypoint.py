import os, sys
from pathlib import Path
SOURCE = Path(os.getenv("PHASE3_SOURCE_DIR", "/kaggle/input/track5-phase3-source"))
if str(SOURCE) not in sys.path: sys.path.insert(0, str(SOURCE))
from aigc_detector.phase3.job import relaunch_with_torchrun_if_needed
from aigc_detector.phase3.r4 import discover_r3_output, run
if __name__ == "__main__":
    relaunch_with_torchrun_if_needed(__file__, sys.argv[1:]); upstream = discover_r3_output()
    run(Path(__file__).with_name("config.json"), Path(os.getenv("PHASE3_MANIFEST", "/kaggle/input/track5-phase3-data/train_validation_manifest.jsonl")), upstream / "recommended_candidate.json", upstream, Path(os.getenv("PHASE3_OUTPUT", "/kaggle/working/r4_source_balanced_25k")))
