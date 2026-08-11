"""
Generates model_output for a test case when the suite doesn't already supply one.
Kept separate from judge.py so generator and judge models are fully independent
(requirement #4: "judge and generator configurable independently").
"""
from . import config


def generate_output(client, test_case: dict, model: str = None) -> str:
    model = model or config.GENERATOR_MODEL
    messages = []
    if test_case.get("system_prompt"):
        messages.append({"role": "system", "content": test_case["system_prompt"]})
    messages.append({"role": "user", "content": test_case["input"]})
    result = client.chat(model=model, messages=messages, temperature=0.2)
    return result["text"]


def ensure_outputs(client, test_suite: list, model: str = None) -> list:
    """Fills in model_output for any case that doesn't already have one. Mutates and returns the list."""
    for case in test_suite:
        if not case.get("model_output"):
            case["model_output"] = generate_output(client, case, model=model)
    return test_suite
