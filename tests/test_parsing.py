"""
These tests mock the Groq client entirely, so they run offline and verify the
pipeline's logic (JSON parsing/repair, aggregation, bias flip detection)
independent of actual API calls. Run with: python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.judge import parse_verdict, judge_case
from src.aggregate import aggregate_suite, compare_configs
from src.bias.position_bias import _flip


VALID_VERDICT_JSON = """{
  "criteria": {
    "correctness": {"score": 5, "evidence": "O(1) average case", "rationale": "matches expected"}
  },
  "overall_score": 5,
  "overall_rationale": "fully correct"
}"""


def test_parse_verdict_clean_json():
    v = parse_verdict(VALID_VERDICT_JSON)
    assert v["overall_score"] == 5
    assert "correctness" in v["criteria"]


def test_parse_verdict_markdown_fenced():
    fenced = "```json\n" + VALID_VERDICT_JSON + "\n```"
    v = parse_verdict(fenced)
    assert v["overall_score"] == 5


def test_parse_verdict_malformed_raises():
    try:
        parse_verdict("this is not json at all")
        assert False, "expected ValueError"
    except ValueError:
        pass


class _MockClient:
    """Returns a queued sequence of responses, simulating a malformed-then-fixed judge."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, model, messages, temperature=0.0, max_tokens=1024):
        self.calls += 1
        text = self.responses.pop(0)
        return {"text": text, "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "raw": {}}


def test_judge_case_repairs_malformed_json():
    client = _MockClient(["not valid json{{{", VALID_VERDICT_JSON])
    case = {"id": "c1", "input": "x", "model_output": "y"}
    verdict = judge_case(client, case)
    assert verdict["overall_score"] == 5
    assert client.calls == 2  # first attempt failed, repair attempt succeeded


def test_aggregate_suite_pass_rate():
    cases = [{"id": "a"}, {"id": "b"}]
    verdicts = [
        {"overall_score": 4.0, "criteria": {"correctness": {"score": 4}}},
        {"overall_score": 2.0, "criteria": {"correctness": {"score": 2}}},
    ]
    report = aggregate_suite(cases, verdicts)
    assert report["n_cases"] == 2
    assert report["pass_rate"] == 0.5  # threshold default 3.5, only case 'a' passes
    assert report["mean_overall_score"] == 3.0


def test_compare_configs_winner():
    report_a = {"mean_overall_score": 3.0, "per_case": [{"overall_score": 3.0}, {"overall_score": 3.0}]}
    report_b = {"mean_overall_score": 4.0, "per_case": [{"overall_score": 4.0}, {"overall_score": 4.0}]}
    result = compare_configs(report_a, report_b, "v1", "v2")
    assert result["winner"] == "v2"
    assert result["per_case_win_counts"]["b"] == 2


def test_position_bias_flip_helper():
    assert _flip("A") == "B"
    assert _flip("B") == "A"
    assert _flip("tie") == "tie"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
