import os
import sys
from pathlib import Path

SOURCE = Path(os.getenv("PHASE3_SOURCE_DIR", "/kaggle/input/track5-phase3-source"))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from aigc_detector.phase3.job import relaunch_with_torchrun_if_needed
from aigc_detector.phase3.r1 import run


if __name__ == "__main__":
    arguments = sys.argv[1:]
    relaunch_with_torchrun_if_needed(__file__, arguments)
    run(
        Path(__file__).with_name("config.json"),
        Path(os.getenv("PHASE3_MANIFEST", "/kaggle/input/track5-phase3-data/train_validation_manifest.jsonl")),
        Path(os.getenv("PHASE3_OUTPUT", "/kaggle/working/r1_siglip2_large_256")),
        Path(os.environ["PHASE3_MODEL_ASSET"]) if os.getenv("PHASE3_MODEL_ASSET") else None,
    )
