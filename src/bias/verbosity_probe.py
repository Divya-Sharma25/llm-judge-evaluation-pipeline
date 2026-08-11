"""
Verbosity bias: judges can rate a longer, more elaborate-sounding answer higher
even when it is wrong, and a short correct answer lower for looking "thin".

We test this empirically by scoring a matched pair per probe case:
- a padded_wrong answer: long, confident, well-formatted, but factually wrong
- a terse_correct answer: short, correct, no padding
against the SAME rubric used in production (src/judge.py), and checking whether
the padded-wrong answer scores higher than the terse-correct one. If it does,
the rubric's "ignore length" instruction is not actually working and needs
strengthening (e.g. adding an explicit length-normalization step).
"""
from ..judge import judge_case


def run_verbosity_probe(client, probe_cases: list, log_fn=None) -> dict:
    """
    probe_cases: list of dicts like
        {id, input, system_prompt, padded_wrong_output, terse_correct_output, criteria}
    """
    results = []
    fooled_count = 0
    for probe in probe_cases:
        wrong_case = {**probe, "model_output": probe["padded_wrong_output"]}
        correct_case = {**probe, "model_output": probe["terse_correct_output"]}

        wrong_verdict = judge_case(client, wrong_case, log_fn=log_fn)
        correct_verdict = judge_case(client, correct_case, log_fn=log_fn)

        wrong_score = wrong_verdict["overall_score"]
        correct_score = correct_verdict["overall_score"]
        fooled = wrong_score >= correct_score  # judge preferred (or tied with) the wrong-but-long answer

        if fooled:
            fooled_count += 1

        results.append({
            "case_id": probe.get("id"),
            "padded_wrong_score": wrong_score,
            "terse_correct_score": correct_score,
            "fooled_by_length": fooled,
        })

    return {
        "per_case": results,
        "fooled_rate": fooled_count / len(probe_cases) if probe_cases else 0.0,
        "n_probes": len(probe_cases),
    }
