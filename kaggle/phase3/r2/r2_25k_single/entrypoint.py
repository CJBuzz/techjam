import os
import sys
from pathlib import Path

SOURCE = Path(os.getenv("PHASE3_SOURCE_DIR", "/kaggle/input/track5-phase3-source"))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from aigc_detector.phase3.job import relaunch_with_torchrun_if_needed
from aigc_detector.phase3.r2 import discover_r1_output, run


if __name__ == "__main__":
    relaunch_with_torchrun_if_needed(__file__, sys.argv[1:])
    r1_output = Path(os.environ["R1_OUTPUT"]) if os.getenv("R1_OUTPUT") else discover_r1_output()
    run(
        Path(__file__).with_name("config.json"),
        Path(os.getenv("PHASE3_MANIFEST", "/kaggle/input/track5-phase3-data/train_validation_manifest.jsonl")),
        Path(os.getenv("R1_RECOMMENDATION", str(r1_output / "recommended_candidate.json"))),
        r1_output,
        Path(os.getenv("PHASE3_OUTPUT", "/kaggle/working/r2_25k_single")),
        None,
    )
