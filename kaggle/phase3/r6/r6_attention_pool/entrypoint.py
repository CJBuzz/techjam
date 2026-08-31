import os,sys
from pathlib import Path
S=Path(os.getenv("PHASE3_SOURCE_DIR","/kaggle/input/track5-phase3-source"));sys.path.insert(0,str(S))
from aigc_detector.phase3.job import relaunch_with_torchrun_if_needed
from aigc_detector.phase3.r5 import discover_output
from aigc_detector.phase3.r6 import run
if __name__=="__main__":
 relaunch_with_torchrun_if_needed(__file__,sys.argv[1:]);u=discover_output("R4");run(Path(__file__).with_name("config.json"),Path("/kaggle/input/track5-phase3-data/train_validation_manifest.jsonl"),u/"recommended_candidate.json",u,Path("/kaggle/working/attention_pool"))
