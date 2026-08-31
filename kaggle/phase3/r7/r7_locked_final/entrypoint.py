import os,sys
from pathlib import Path
SOURCE=Path(os.getenv("PHASE3_SOURCE_DIR","/kaggle/input/track5-phase3-source"));sys.path.insert(0,str(SOURCE))
from aigc_detector.phase3.r7_locked_test import run,validate_lock
if __name__=="__main__":
 locks=list(Path("/kaggle/input").glob("*/locked_candidate.json"))
 if len(locks)!=1: raise ValueError(f"Expected exactly one locked candidate, found {len(locks)}")
 run(locks[0],Path(os.getenv("FINAL_TEST_MANIFEST","/kaggle/input/track5-final-test/test_manifest.jsonl")),Path("/kaggle/working/final_test_scorecard"),int(os.getenv("NUM_WORKERS","2")))
