"""
Pointwise LLM-as-judge. Scores a single (input, system_prompt, model_output) case
against an explicit rubric, returning a structured verdict:
{criteria: {name: {score, rationale}}, overall_score, overall_rationale}

Design choices tied to bias mitigation (see README for full discussion):
- Rubric forces per-criterion GROUNDING (a quoted-or-paraphrased evidence pointer
  from the model_output) before a score is accepted -> reduces sycophancy bias,
  since the judge can't just vibe a high score without pointing at text.
- Few-shot score anchors are embedded in the prompt -> reduces score clustering
  (judges tend to default to the middle of a scale without anchors).
- Explicit instruction to ignore length/formality and evaluate substance only
  -> primary defense against verbosity bias (paired with the empirical probe
  in bias/verbosity_probe.py which tests whether this instruction actually works).
"""
import json
import re
from . import config

RUBRIC = ["correctness", "faithfulness", "completeness", "instruction_following"]

RUBRIC_DESCRIPTIONS = {
    "correctness": "Is the output factually/technically correct given the input?",
    "faithfulness": "Does the output avoid claims unsupported by the given input/context "
                     "(no hallucinated facts)?",
    "instruction_following": "Does the output do what the system_prompt/input actually asked, "
                              "including format constraints?",
    "completeness": "Does the output cover what a full, adequate answer would cover, "
                     "without unnecessary padding?",
}

JUDGE_SYSTEM_PROMPT = """You are a strict technical evaluator. You score a single model output
against a fixed rubric. Judge substance only - a longer, more elaborate, or more
confident-sounding answer is NOT automatically better. A short correct answer must
score at least as high as a long correct answer that says the same thing.

For EVERY criterion you must:
1. Quote or closely paraphrase the specific part of the output that justifies your score
   (evidence field). If you cannot point to evidence, the score must be <= 2.
2. Give an integer score from 1 to 5 using this anchor scale:
   1 = wrong / contradicts the input / ignores the instruction
   2 = major gaps or errors, barely addresses the task
   3 = partially correct, missing something a competent answer would include
   4 = correct and complete, minor stylistic issues only
   5 = fully correct, complete, and precisely on-instruction

Return ONLY a single JSON object, no markdown fences, no commentary, in exactly this shape:
{
  "criteria": {
    "<criterion_name>": {"score": <int 1-5>, "evidence": "<short quote/paraphrase>", "rationale": "<1 sentence>"}
  },
  "overall_score": <float, mean of criteria unless you have a strong reason to deviate>,
  "overall_rationale": "<1-2 sentences>"
}
"""


def build_judging_prompt(test_case: dict, rubric: list = None) -> str:
    rubric = rubric or RUBRIC
    rubric_block = "\n".join(f"- {c}: {RUBRIC_DESCRIPTIONS[c]}" for c in rubric)
    expected = test_case.get("expected_output")
    expected_block = f"\nEXPECTED / REFERENCE ANSWER:\n{expected}\n" if expected else ""
    criteria_block = (
        "\nADDITIONAL CASE-SPECIFIC CRITERIA:\n" + "\n".join(f"- {c}" for c in test_case["criteria"])
        if test_case.get("criteria") else ""
    )
    return f"""RUBRIC:
{rubric_block}
{criteria_block}

SYSTEM PROMPT GIVEN TO THE MODEL BEING EVALUATED:
{test_case.get('system_prompt', '(none)')}

USER INPUT:
{test_case['input']}
{expected_block}
MODEL OUTPUT TO EVALUATE:
{test_case['model_output']}

Score this output now. Return only the JSON object described in your instructions."""


def _extract_json(text: str) -> dict:
    text = text.strip()
    # strip markdown fences if the model added them anyway
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fallback: grab the largest {...} block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse judge output as JSON:\n{text[:500]}")


def parse_verdict(raw_text: str) -> dict:
    """Raises ValueError on unrecoverable malformed JSON; caller handles repair retry."""
    verdict = _extract_json(raw_text)
    if "criteria" not in verdict or "overall_score" not in verdict:
        raise ValueError(f"Judge JSON missing required keys: {verdict}")
    return verdict


def judge_case(client, test_case: dict, rubric: list = None, model: str = None, log_fn=None) -> dict:
    """Runs one case through the judge, retrying with a JSON-repair prompt on parse failure."""
    model = model or config.JUDGE_MODEL
    prompt = build_judging_prompt(test_case, rubric)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    last_error = None
    for attempt in range(config.MAX_JSON_REPAIR_ATTEMPTS + 1):
        result = client.chat(model=model, messages=messages, temperature=0.0)
        raw_text = result["text"]
        if log_fn:
            log_fn({
                "case_id": test_case.get("id"),
                "attempt": attempt,
                "model": model,
                "prompt": messages,
                "raw_response": raw_text,
                "usage": result["usage"],
            })
        try:
            verdict = parse_verdict(raw_text)
            verdict["_meta"] = {"attempts": attempt + 1, "model": model}
            return verdict
        except ValueError as e:
            last_error = e
            # ask the judge to fix its own output rather than silently dropping the case
            messages = messages + [
                {"role": "assistant", "content": raw_text},
                {"role": "user", "content": "That was not valid JSON matching the required schema. "
                                             "Return ONLY the corrected JSON object, nothing else."},
            ]
    raise ValueError(f"Judge failed to produce parseable JSON after retries: {last_error}")
