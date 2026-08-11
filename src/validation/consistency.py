"""
Test-retest consistency: re-run the same case N times and check how often the
pass/fail verdict (and the raw score) flips. A judge that gives a different
pass/fail verdict on the same input across runs is not trustworthy enough to
gate a release, regardless of how good its average scores look.
"""
from .. import config
from ..judge import judge_case


def run_consistency_check(client, cases: list, n_runs: int = None, log_fn=None) -> dict:
    n_runs = n_runs or config.CONSISTENCY_RUNS
    per_case = []
    total_flips = 0

    for case in cases:
        scores = []
        passes = []
        for run_idx in range(n_runs):
            verdict = judge_case(client, case, log_fn=log_fn)
            score = verdict["overall_score"]
            scores.append(score)
            passes.append(score >= config.PASS_THRESHOLD)

        flipped = len(set(passes)) > 1  # pass/fail disagreement across runs
        if flipped:
            total_flips += 1

        per_case.append({
            "case_id": case.get("id"),
            "scores": scores,
            "passes": passes,
            "flipped": flipped,
            "score_range": max(scores) - min(scores),
        })

    return {
        "per_case": per_case,
        "flip_rate": total_flips / len(cases) if cases else 0.0,
        "n_cases": len(cases),
        "n_runs_per_case": n_runs,
    }
