"""
CLI entrypoint. Examples:

  # Full run: generate any missing outputs, judge the suite, run bias checks + validation
  python -m src.pipeline run --suite data/test_suite.json --probes data/probes.json

  # A/B comparison between two generator configs (e.g. two system prompts / two models)
  python -m src.pipeline compare --suite data/test_suite_a.json --suite-b data/test_suite_b.json \\
      --label-a prompt_v1 --label-b prompt_v2
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

from . import config
from .groq_client import GroqClient
from .generator import ensure_outputs
from .judge import judge_case
from .aggregate import aggregate_suite, compare_configs
from .bias.position_bias import run_position_bias_check
from .bias.verbosity_probe import run_verbosity_probe
from .bias.sycophancy_probe import run_sycophancy_probe
from .validation.consistency import run_consistency_check


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _make_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, "a")

    def log_fn(entry):
        entry["_logged_at"] = datetime.now(timezone.utc).isoformat()
        f.write(json.dumps(entry) + "\n")
        f.flush()

    return log_fn, f


def cmd_run(args):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    client = GroqClient()
    log_fn, log_file = _make_logger(config.JUDGE_LOG_PATH)

    suite = _load_json(args.suite)
    print(f"Loaded {len(suite)} cases from {args.suite}")

    ensure_outputs(client, suite, model=args.generator_model)

    print("Judging suite (pointwise)...")
    verdicts = [judge_case(client, case, model=args.judge_model, log_fn=log_fn) for case in suite]
    suite_report = aggregate_suite(suite, verdicts)

    bias_report = {}
    if args.probes:
        probes = _load_json(args.probes)
        print("Running verbosity probe...")
        bias_report["verbosity"] = run_verbosity_probe(client, probes.get("verbosity", []), log_fn=log_fn)
        print("Running sycophancy probe...")
        bias_report["sycophancy"] = run_sycophancy_probe(client, probes.get("sycophancy", []), log_fn=log_fn)

    validation_report = {}
    if args.consistency_n_cases > 0:
        print(f"Running test-retest consistency on {args.consistency_n_cases} case(s)...")
        subset = suite[: args.consistency_n_cases]
        validation_report["consistency"] = run_consistency_check(client, subset, log_fn=log_fn)

    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "generator_model": args.generator_model or config.GENERATOR_MODEL,
        "judge_model": args.judge_model or config.JUDGE_MODEL,
        "suite_report": suite_report,
        "bias_report": bias_report,
        "validation_report": validation_report,
        "judge_usage": client.usage_summary(),
    }

    out_path = os.path.join(config.RESULTS_DIR, "suite_report.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote report to {out_path}")
    print(f"Pass rate: {suite_report['pass_rate']:.2%}  Mean score: {suite_report['mean_overall_score']:.2f}")
    log_file.close()


def cmd_compare(args):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    client = GroqClient()
    log_fn, log_file = _make_logger(config.JUDGE_LOG_PATH)

    suite_a = _load_json(args.suite)
    suite_b = _load_json(args.suite_b)
    if len(suite_a) != len(suite_b):
        print("WARNING: suite A and suite B have different lengths; comparison assumes matched case order.")

    ensure_outputs(client, suite_a, model=args.generator_model)
    ensure_outputs(client, suite_b, model=args.generator_model)

    verdicts_a = [judge_case(client, c, model=args.judge_model, log_fn=log_fn) for c in suite_a]
    verdicts_b = [judge_case(client, c, model=args.judge_model, log_fn=log_fn) for c in suite_b]
    report_a = aggregate_suite(suite_a, verdicts_a)
    report_b = aggregate_suite(suite_b, verdicts_b)

    comparison = compare_configs(report_a, report_b, args.label_a, args.label_b)

    position_bias = run_position_bias_check(
        client,
        suite_a,
        [c["model_output"] for c in suite_a],
        [c["model_output"] for c in suite_b],
        model=args.judge_model,
    )

    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "report_a": report_a,
        "report_b": report_b,
        "comparison": comparison,
        "position_bias_check": position_bias,
        "judge_usage": client.usage_summary(),
    }

    out_path = os.path.join(config.RESULTS_DIR, "ab_comparison.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote A/B comparison to {out_path}")
    print(f"Winner: {comparison['winner']}  "
          f"({args.label_a}={comparison['mean_score_a']:.2f} vs {args.label_b}={comparison['mean_score_b']:.2f})")
    print(f"Position-bias flip rate: {position_bias['flip_rate']:.2%}")
    log_file.close()


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Judge a single suite and produce a report")
    p_run.add_argument("--suite", required=True)
    p_run.add_argument("--probes", default=None, help="Path to probes.json for bias checks")
    p_run.add_argument("--generator-model", default=None)
    p_run.add_argument("--judge-model", default=None)
    p_run.add_argument("--consistency-n-cases", type=int, default=3,
                        help="How many cases to re-run N times for test-retest validation")
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="A/B compare two configs (e.g. two prompts/models)")
    p_cmp.add_argument("--suite", required=True, help="Suite for config A")
    p_cmp.add_argument("--suite-b", required=True, help="Suite for config B (matched case order)")
    p_cmp.add_argument("--label-a", default="config_a")
    p_cmp.add_argument("--label-b", default="config_b")
    p_cmp.add_argument("--generator-model", default=None)
    p_cmp.add_argument("--judge-model", default=None)
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
