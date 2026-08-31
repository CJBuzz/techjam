from __future__ import annotations
import argparse, csv, json, shutil
from pathlib import Path
from aigc_detector.phase3.artifacts import atomic_json
from aigc_detector.phase3.r6 import select_candidate
from aigc_detector.phase3.ranking import rank_candidates

def main():
    parser = argparse.ArgumentParser(description="Validation-only R6 local/champion selection")
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--purpose", choices=("local", "champion"), required=True)
    parser.add_argument("--baseline-clean", type=float, default=.9681); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); summaries = [json.loads(path.read_text()) for path in args.summary]
    selected = select_candidate(summaries, args.baseline_clean, args.purpose == "local")
    args.output.mkdir(parents=True, exist_ok=True); atomic_json(args.output / "selection_config.json", selected)
    atomic_json(args.output / "dataset-metadata.json", {"title": f"Track5 R6 {args.purpose} selection",
                "id": f"REPLACE_USERNAME/track5-r6-{args.purpose}-selection", "licenses": [{"name": "CC0-1.0"}]})
    if args.purpose == "champion":
        rows = [row for summary in summaries for row in summary["results"]]
        ranked = rank_candidates(rows, args.baseline_clean, effective_tie=.002)
        winner = next(row for row in ranked if row["validation_rank"] == 1)
        atomic_json(args.output / "r6_summary.json", {"experiment": "R6", "selection_split": "validation",
                    "final_test_evaluated": False, "results": ranked, "eligible_winner": winner})
        fields = sorted({key for row in ranked for key, value in row.items() if not isinstance(value, (dict, list))})
        with (args.output / "r6_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(
                {key: row.get(key) for key in fields} for row in ranked)
        atomic_json(args.output / "recommended_candidate.json", {"experiment": "R6", "candidate": winner,
                    "selection_split": "validation", "final_test_evaluated": False})
        for path, summary in zip(args.summary, summaries, strict=True):
            if any(row.get("candidate_id") == winner["candidate_id"] for row in summary["results"]):
                logits = path.parent / "candidate/val_logits.npz"
                if not logits.is_file(): raise FileNotFoundError(f"Winning validation logits missing: {logits}")
                shutil.copy2(logits, args.output / "val_logits.npz"); break
    print(args.output / "selection_config.json")
if __name__ == "__main__": main()
