"""Rolls up per-case judge verdicts into a suite report, and compares two configs."""
from . import config


def aggregate_suite(cases: list, verdicts: list) -> dict:
    assert len(cases) == len(verdicts)
    n = len(cases)
    overall_scores = [v["overall_score"] for v in verdicts]
    passed = [s >= config.PASS_THRESHOLD for s in overall_scores]

    criteria_names = set()
    for v in verdicts:
        criteria_names.update(v.get("criteria", {}).keys())
    per_criterion_mean = {}
    for c in criteria_names:
        scores = [v["criteria"][c]["score"] for v in verdicts if c in v.get("criteria", {})]
        per_criterion_mean[c] = sum(scores) / len(scores) if scores else None

    return {
        "n_cases": n,
        "pass_rate": sum(passed) / n if n else 0.0,
        "mean_overall_score": sum(overall_scores) / n if n else 0.0,
        "per_criterion_mean": per_criterion_mean,
        "per_case": [
            {
                "case_id": case.get("id"),
                "overall_score": v["overall_score"],
                "passed": s >= config.PASS_THRESHOLD,
                "criteria": v.get("criteria", {}),
            }
            for case, v, s in zip(cases, verdicts, overall_scores)
        ],
    }


def compare_configs(report_a: dict, report_b: dict, label_a: str = "config_a", label_b: str = "config_b") -> dict:
    """Simple mean-score win rate comparison between two suite reports (pointwise A/B)."""
    diff = report_b["mean_overall_score"] - report_a["mean_overall_score"]
    if abs(diff) < 1e-9:
        winner = "tie"
    else:
        winner = label_b if diff > 0 else label_a

    per_case_wins = {"a": 0, "b": 0, "tie": 0}
    for ca, cb in zip(report_a["per_case"], report_b["per_case"]):
        if ca["overall_score"] > cb["overall_score"]:
            per_case_wins["a"] += 1
        elif cb["overall_score"] > ca["overall_score"]:
            per_case_wins["b"] += 1
        else:
            per_case_wins["tie"] += 1

    return {
        "label_a": label_a,
        "label_b": label_b,
        "mean_score_a": report_a["mean_overall_score"],
        "mean_score_b": report_b["mean_overall_score"],
        "score_diff_b_minus_a": diff,
        "winner": winner,
        "per_case_win_counts": per_case_wins,
    }
