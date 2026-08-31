import os
import sys
from pathlib import Path

SOURCE = Path(os.getenv("PHASE3_SOURCE_DIR", "/kaggle/input/track5-phase3-source"))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from aigc_detector.phase3.job import relaunch_with_torchrun_if_needed
from aigc_detector.phase3.r3 import discover_r2_output, run


if __name__ == "__main__":
    relaunch_with_torchrun_if_needed(__file__, sys.argv[1:])
    r2_output = Path(os.environ["R2_OUTPUT"]) if os.getenv("R2_OUTPUT") else discover_r2_output()
    run(
        Path(__file__).with_name("config.json"),
        Path(os.getenv("PHASE3_MANIFEST", "/kaggle/input/track5-phase3-data/train_validation_manifest.jsonl")),
        Path(os.getenv("R2_RECOMMENDATION", str(r2_output / "recommended_candidate.json"))),
        r2_output,
        Path(os.getenv("PHASE3_OUTPUT", "/kaggle/working/r3_medium_25k")),
        None,
    )
