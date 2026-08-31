from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from aigc_detector.phase3.r7_locked_test import validate_lock

def main():
    parser = argparse.ArgumentParser(description="Validate and package only the locked R7 candidate")
    parser.add_argument("--lock", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args(); document = json.loads(args.lock.read_text(encoding="utf-8")); validate_lock(document)
    args.output.mkdir(parents=True, exist_ok=False); shutil.copy2(args.lock, args.output / "locked_candidate.json")
    (args.output / "dataset-metadata.json").write_text(json.dumps({"title":"Track5 R7 locked candidate",
        "id":args.dataset_id,"licenses":[{"name":"other"}]}, indent=2)+"\n", encoding="utf-8")
    print(args.output / "locked_candidate.json")
if __name__ == "__main__": main()
