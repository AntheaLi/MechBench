# MechanismBench

MechanismBench is a controlled benchmark for testing whether an automated
research agent can identify the causal source of an apparent machine-learning
improvement.

The benchmark gives an agent a public claim such as "this model modification
improves validation accuracy." The agent can run controlled experiments under a
budget, submit a structured causal report, and predict hidden intervention
outcomes. Scoring is deterministic and schema-based: the primary score does not
require free-form LLM judgment.

## Current Status

This repository contains:

- the generic MechanismBench harness;
- public/private world schemas;
- deterministic budget accounting;
- an immutable evidence ledger;
- simulator worlds for five causal mechanisms;
- 100 certified compiled simulator episodes;
- baseline agents: oracle, scripted, naive, random;
- a Phase 2 tiny PyTorch vertical slice under `families/attn_branch`;
- an LLM scaffold with OpenAI, Anthropic, Gemini, and OpenAI-compatible backends;
- packaged Phase 4 study packs with cost guards and report generation.

No paid frontier-model study results are committed yet. The repository is ready
for a collaborator with API access to run them.

## Install

Use Python 3.10 or newer.

```bash
git clone <repo-url>
cd mechbench
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

If you only want the simulator and Phase 4 LLM scaffold, the base install is
enough. PyTorch is only needed for the `attn_branch` vertical slice.

## Verify The Repo

Run the unit tests:

```bash
python3 -m unittest discover -s tests
```

List available worlds:

```bash
python3 -m mechbench.cli list-worlds
```

Verify the simulator calibration gates:

```bash
python3 -m mechbench.cli calibrate-simulator \
  --worlds compiled \
  --output results/simulator_calibration.json

python3 -m mechbench.cli report-calibration \
  --input results/simulator_calibration.json \
  --output results/simulator_calibration_report.md
```

## Run A Free Phase 4 Smoke Test

This runs local baselines plus the deterministic mock LLM scaffold. It makes no
API calls and costs nothing.

```bash
python3 -m mechbench.cli phase4-study \
  --pack phase4_free_smoke \
  --output-dir results/phase4_free_smoke
```

Outputs:

- `results/phase4_free_smoke/study.json`
- `results/phase4_free_smoke/report.md`

## Run A Paid Study Safely

Always dry-run first. The dry-run resolves the worlds, repeats, enabled models,
and estimated API cost before any paid call.

```bash
python3 -m mechbench.cli phase4-study \
  --pack phase4_frontier_template \
  --dry-run
```

Set whichever API keys you have:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...

# Only needed for OpenAI-compatible Fable-style endpoints.
export FABLE_API_KEY=...
export FABLE_BASE_URL=https://your-openai-compatible-endpoint/v1
```

Run the frontier template:

```bash
python3 -m mechbench.cli phase4-study \
  --pack phase4_frontier_template \
  --output-dir results/phase4_frontier_template
```

If you only have some provider keys, skip missing providers:

```bash
python3 -m mechbench.cli phase4-study \
  --pack phase4_frontier_template \
  --skip-missing-credentials \
  --output-dir results/phase4_partial
```

Regenerate a report from saved JSON:

```bash
python3 -m mechbench.cli phase4-report \
  --input results/phase4_frontier_template/study.json \
  --output results/phase4_frontier_template/report.md
```

## Study Packs

Bundled packs live in `study_packs/`.

| Pack | Purpose |
|---|---|
| `phase4_free_smoke` | Free local check of the whole Phase 4 pipeline. |
| `phase4_budget_pilot` | Cheap real-LLM pilot on 10 simulator episodes. |
| `phase4_core_lab` | Recommended first lab run: one strong LLM, 25 worlds, 3 repeats. |
| `phase4_frontier_template` | GPT-5.5, GPT-5.4, Opus 4.8, Fable 5, Gemini, 25 worlds, 3 repeats. |
| `phase4_full_simulator_frontier` | Full 100-world simulator frontier sweep, 3 repeats. |

List packs from the CLI:

```bash
python3 -m mechbench.cli phase4-study --list-packs
```

Each paid pack has a `max_cost_usd` guard. If preflight cost exceeds the guard,
the run stops before making API calls. If model pricing is unknown, the run is
blocked unless the pack includes `price_per_million_tokens` or you pass
`--allow-unknown-cost`.

## What To Send Back After A Lab Run

Please send back the whole output directory for the run, especially:

- `study.json`
- `report.md`

The JSON contains per-episode scores, components, token usage, estimated API
costs, experiment sequences, traces when enabled, and repeated-run variance. The
Markdown report is the human-readable summary.

## CLI Reference

Evaluate one agent:

```bash
python3 -m mechbench.cli evaluate \
  --agent scripted \
  --worlds simulator/compiled_0001
```

Run a small ad hoc LLM scaffold study:

```bash
python3 -m mechbench.cli llm-study \
  --agent llm_openai \
  --model gpt-5.4-mini \
  --count 10 \
  --output results/llm_pilot.json
```

Run packaged Phase 4 study:

```bash
python3 -m mechbench.cli phase4-study \
  --pack phase4_core_lab \
  --output-dir results/phase4_core_lab
```

Render Phase 4 report:

```bash
python3 -m mechbench.cli phase4-report \
  --input results/phase4_core_lab/study.json \
  --output results/phase4_core_lab/report.md
```

## Repository Layout

```text
agents/                  Baseline agents and LLM provider backends
analysis/                Calibration reporting helpers
docs/                    Design notes and study instructions
families/simulator/      Factorized simulator benchmark family
families/attn_branch/    Tiny PyTorch vertical slice
mechbench/               Core harness, schemas, scoring, CLI, Phase 4 runner
study_packs/             Packaged Phase 4 run configurations
tests/                   Unit and integration tests
```

## Notes

- The simulator is the main Phase 4 target.
- The PyTorch vertical slice is useful for Phase 2/3 validation, but it can take
  longer locally.
- The `phase4_frontier_template` model names and prices are editable JSON
  fields. Update them if your lab account uses different aliases or contracted
  pricing.
- Do not publish API keys or raw provider credentials in result artifacts.
