"""
Config is loaded entirely from environment variables. No secrets are hardcoded.
Copy .env.example to .env and fill in GROQ_API_KEY before running.
"""
import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")

# Generator and judge are configurable independently (requirement #4).
# Using two different model families on Groq approximates the
# "judge from a different model family than the generator" mitigation
# for self-enhancement bias, since we don't have a second provider key.
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "llama-3.1-8b-instant")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemma2-9b-it")

# Retry behaviour for malformed JSON from the judge
MAX_JSON_REPAIR_ATTEMPTS = int(os.environ.get("MAX_JSON_REPAIR_ATTEMPTS", "2"))

# How many times to re-run the same case for test-retest consistency
CONSISTENCY_RUNS = int(os.environ.get("CONSISTENCY_RUNS", "3"))

# Pass/fail threshold on a 1-5 overall score, used for consistency + pass-rate reporting
PASS_THRESHOLD = float(os.environ.get("PASS_THRESHOLD", "3.5"))

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
JUDGE_LOG_PATH = os.path.join(RESULTS_DIR, "judge_logs.jsonl")
