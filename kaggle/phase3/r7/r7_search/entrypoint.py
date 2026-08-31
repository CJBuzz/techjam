import os,sys
from pathlib import Path
SOURCE=Path(os.getenv("PHASE3_SOURCE_DIR","/kaggle/input/track5-phase3-source"));sys.path.insert(0,str(SOURCE))
from aigc_detector.phase3.r7 import run
if __name__=="__main__":
 optional=list(Path("/kaggle/input").rglob("r7_optional_candidate.json"))
 run(Path("/kaggle/input"),Path(os.getenv("PHASE3_OUTPUT","/kaggle/working/r7")),float(os.getenv("BASELINE_CLEAN_BACC","0.9681")),optional)
