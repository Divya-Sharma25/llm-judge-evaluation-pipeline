# LLM-as-Judge Evaluation Pipeline

A pointwise LLM-judge pipeline with explicit bias mitigation and judge validation,
built on Groq (LLaMA as generator, Gemma as judge — two different model families
on the same provider, since only a Groq key was available).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your GROQ_API_KEY
```

## Usage

Judge a suite and run bias/validation checks:
```bash
python -m src.pipeline run --suite data/test_suite.json --probes data/probes.json
```

A/B compare two prompt configs (required deliverable — declares a winner):
```bash
python -m src.pipeline compare \
  --suite data/test_suite_v1.json --suite-b data/test_suite_v2.json \
  --label-a loose_prompt --label-b strict_prompt
```

Run the offline unit tests (no API key needed — mocked client):
```bash
python -m pytest tests/ -v
```

Results land in `results/`: `suite_report.json`, `ab_comparison.json`, and the full
auditable judge log (`judge_logs.jsonl` — every prompt + raw response + token usage).

## Judging mode

**Pointwise scoring** is the primary mode (`src/judge.py`): each case gets an
independent 1–5 score per rubric criterion, with a required evidence quote per
criterion. This is used for the main suite report because it scales linearly
(1 call/case vs. pairwise's O(n²) or O(n) with a fixed baseline) and gives
interpretable per-criterion diagnostics, not just a relative ranking.

**Pairwise comparison** (`src/bias/position_bias.py`) is used specifically for
the position-bias check and could be extended into the main A/B path — pairwise
is the stronger choice when you only care about "is B better than A" and want to
cancel out each judge's personal scale/anchor drift, at the cost of not telling
you *why* one is better on a per-criterion basis.

## Rubric

`correctness`, `faithfulness`, `completeness`, `instruction_following` — each
scored 1–5 with a mandatory evidence field (a quote/paraphrase from the model
output). A score above 2 without evidence is disallowed by the judge prompt.
Case-specific extra criteria can be added per test case via the `criteria` field.

## Bias handling — what was implemented and measured

| Bias | Mitigation implemented | Where |
|---|---|---|
| Position (A/B order) | Every pairwise comparison run in both orders; flip rate reported | `bias/position_bias.py` |
| Verbosity / length | Rubric explicitly instructs "length is not a quality signal"; probed with a padded-wrong vs. terse-correct pair | `bias/verbosity_probe.py` |
| Self-enhancement | Judge (Gemma) is a different model family from the generator (LLaMA) | `config.py` |
| Sycophancy / style | Evidence-quote requirement per criterion; probed with a confidently-wrong answer | `judge.py`, `bias/sycophancy_probe.py` |
| Score clustering | Few-shot numeric anchors (1–5, each anchor described) embedded directly in the judge system prompt | `judge.py` |

## Judge validation

**Test-retest consistency** (`validation/consistency.py`): re-runs the same
case `CONSISTENCY_RUNS` times (default 3) and reports how often the pass/fail
verdict flips. This was chosen over human-agreement or a full adversarial suite
because it needed no external gold-label dataset to still produce real evidence
of (in)stability — appropriate for the time available, and honestly reported as
the one validation method implemented rather than claimed as complete coverage.

## Discussion

**How biased was the judge before vs. after mitigation?**
Run `python -m src.pipeline run --suite data/test_suite.json --probes data/probes.json`
and check `results/suite_report.json` → `bias_report`. The `fooled_rate` in the
verbosity and sycophancy probe results is the direct "before mitigation would
have looked like X" signal — a probe case is scored using the *same* mitigated
rubric, so a nonzero fooled_rate means the mitigations in the prompt (length
instruction, evidence requirement) are not fully closing the gap, which is worth
reporting honestly rather than assuming the rubric wording alone solved it.
The position-bias `flip_rate` in `results/ab_comparison.json` is the direct
measure of how often order alone changes the verdict.

**Would I let this gate a release?**
Not on its own, and not yet. A single judge model, one validation method, and a
15-case suite is enough to catch gross regressions and get real signal on bias
direction, but it is not enough statistical power to trust as a hard release
gate. I'd treat a clearly failing pass_rate or a high position-bias/fooled_rate
as a strong stop-ship signal, but I would not treat a passing report alone as
sufficient — it should sit alongside human spot-checks, especially for
close A/B calls (small `score_diff_b_minus_a`) where the position-bias flip
rate suggests the judge itself is noisy at the margin.

**Was retrieval or generation the weaker link (for cases with an
`expected_output`)?** Not directly applicable to this problem (that framing is
for the RAG task) — but the analogous read here is: the *judge* was the
component most likely to be the weak link, not the generator, since the probes
are specifically designed to catch the judge failing rather than the generator.

## Project structure

```
src/
  config.py                  env-based config, no hardcoded secrets
  groq_client.py             Groq API wrapper + usage tracking
  judge.py                   rubric, prompt, structured parsing + JSON repair retry
  generator.py                fills in model_output when a suite doesn't supply one
  aggregate.py                suite-level rollup + A/B comparison
  pipeline.py                  CLI entrypoint (run / compare)
  bias/
    position_bias.py          pairwise both-orders flip-rate check
    verbosity_probe.py        padded-wrong vs terse-correct probe
    sycophancy_probe.py       confidently-wrong probe
  validation/
    consistency.py            test-retest flip-rate check
data/
  test_suite.json             15 pointwise cases, mixed correct/flawed
  test_suite_v1.json / v2.json  matched cases for the required A/B demo
  probes.json                 adversarial probe cases
tests/
  test_parsing.py             offline unit tests (mocked client, no API needed)
results/                       generated at runtime (report + full judge log)
```

## Known limitations (stated honestly, not glossed over)

- Only one validation method implemented (test-retest), not human-agreement or
  a full adversarial suite — time-constrained choice, noted rather than hidden.
- Self-enhancement mitigation uses two model families from the same provider
  (Groq), not two separate providers — a genuinely independent provider would
  be a stronger check.
- The 15-case suite gives directional signal, not statistically robust
  confidence intervals on pass rate or win rate.
