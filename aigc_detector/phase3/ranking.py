from __future__ import annotations

from functools import cmp_to_key


def rank_candidates(rows: list[dict], baseline_clean: float, effective_tie: float = 1e-4) -> list[dict]:
    floor = baseline_clean - 0.01
    eligible = [row for row in rows if row.get("status", "succeeded") == "succeeded"
                and row["clean_validation_balanced_accuracy"] >= floor]
    def compare(left: dict, right: dict) -> int:
        worst_delta = left["worst_transformed_validation_balanced_accuracy"] - right["worst_transformed_validation_balanced_accuracy"]
        if abs(worst_delta) > effective_tie:
            return -1 if worst_delta > 0 else 1
        left_cost = (float(left.get("inference_multiplier", 1)), int(left.get("total_deployment_parameter_count", 0)))
        right_cost = (float(right.get("inference_multiplier", 1)), int(right.get("total_deployment_parameter_count", 0)))
        if left_cost != right_cost:
            return -1 if left_cost < right_cost else 1
        mean_delta = left["mean_transformed_validation_balanced_accuracy"] - right["mean_transformed_validation_balanced_accuracy"]
        return -1 if mean_delta > 0 else (1 if mean_delta < 0 else 0)
    eligible.sort(key=cmp_to_key(compare))
    ranks = {row["candidate_id"]: index for index, row in enumerate(eligible, 1)}
    return [{**row, "clean_constraint_pass": row.get("status", "succeeded") == "succeeded"
             and row.get("clean_validation_balanced_accuracy", float("-inf")) >= floor,
             "validation_rank": ranks.get(row["candidate_id"]), "selection_split": "validation",
             "final_test_evaluated": False} for row in rows]
