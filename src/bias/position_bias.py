"""
Position bias: an LLM judge asked to compare A vs B can favor whichever answer
appears first/second, independent of quality. We detect this by running every
comparison in BOTH orders and checking whether the winner flips.

This module implements a pairwise comparison prompt purely for this bias check
(the main scoring pipeline stays pointwise per src/judge.py; this satisfies the
"pointwise OR pairwise, implement at least one, explain when each fits" requirement
by using pairwise specifically where it is strongest: head-to-head config comparison).
"""
import json
import re
from .. import config

PAIRWISE_SYSTEM_PROMPT = """You compare two candidate answers to the same input and decide
which is better, or declare a tie. Judge substance only: correctness, completeness,
and instruction-following. Length and confidence of tone are NOT quality signals.

Return ONLY JSON: {"winner": "A" | "B" | "tie", "rationale": "<1-2 sentences>"}"""


def _build_prompt(test_case: dict, answer_a: str, answer_b: str) -> str:
    return f"""INPUT:
{test_case['input']}

CANDIDATE A:
{answer_a}

CANDIDATE B:
{answer_b}

Which is better? Return only the JSON verdict."""


def _parse(raw_text: str) -> dict:
    raw_text = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in pairwise verdict: {raw_text[:300]}")
    return json.loads(match.group(0))


def compare_pair(client, test_case: dict, answer_a: str, answer_b: str, model: str = None) -> dict:
    model = model or config.JUDGE_MODEL
    messages = [
        {"role": "system", "content": PAIRWISE_SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(test_case, answer_a, answer_b)},
    ]
    result = client.chat(model=model, messages=messages, temperature=0.0)
    return _parse(result["text"])


def _flip(winner: str) -> str:
    return {"A": "B", "B": "A", "tie": "tie"}[winner]


def run_position_bias_check(client, cases: list, answers_a: list, answers_b: list, model: str = None) -> dict:
    """
    cases, answers_a, answers_b are parallel lists (config A output vs config B output per case).
    Runs each comparison in both orders and reports the flip rate.
    """
    per_case = []
    flips = 0
    for case, a, b in zip(cases, answers_a, answers_b):
        forward = compare_pair(client, case, a, b, model=model)          # A first, B second
        reversed_ = compare_pair(client, case, b, a, model=model)        # B first, A second
        reversed_normalized = _flip(reversed_["winner"])  # translate back into A/B terms

        flipped = forward["winner"] != reversed_normalized
        if flipped:
            flips += 1

        per_case.append({
            "case_id": case.get("id"),
            "forward_winner": forward["winner"],
            "reversed_raw_winner": reversed_["winner"],
            "reversed_normalized_winner": reversed_normalized,
            "flipped": flipped,
        })

    flip_rate = flips / len(cases) if cases else 0.0
    return {"per_case": per_case, "flip_rate": flip_rate, "n_cases": len(cases)}
