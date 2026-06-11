# Phase 4 Agent Study

Phase 4 is now packaged as a reproducible study runner. A lab partner should be
able to dry-run a pack, inspect cost, set API keys, run the study, and read the
generated report without editing Python.

## Quick Start

List available packs:

```bash
python3 -m mechbench.cli phase4-study --list-packs
```

Run the free local smoke test:

```bash
python3 -m mechbench.cli phase4-study \
  --pack phase4_free_smoke \
  --output-dir results/phase4_free_smoke
```

Dry-run a paid pack before spending money:

```bash
python3 -m mechbench.cli phase4-study \
  --pack phase4_frontier_template \
  --dry-run
```

Run a paid pack after setting keys:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export FABLE_API_KEY=...
export FABLE_BASE_URL=https://your-fable-openai-compatible-endpoint/v1

python3 -m mechbench.cli phase4-study \
  --pack phase4_frontier_template \
  --output-dir results/phase4_frontier_template
```

If a collaborator only has keys for some providers, they can skip missing rows:

```bash
python3 -m mechbench.cli phase4-study \
  --pack phase4_frontier_template \
  --skip-missing-credentials \
  --output-dir results/phase4_partial
```

Regenerate a markdown report from saved JSON:

```bash
python3 -m mechbench.cli phase4-report \
  --input results/phase4_frontier_template/study.json \
  --output results/phase4_frontier_template/report.md
```

## Bundled Packs

- `phase4_free_smoke`: zero-cost baselines plus `llm_mock` on 10 balanced
  simulator episodes. Use this to check installation and reporting.
- `phase4_budget_pilot`: baselines plus one cheap OpenAI model on 10 balanced
  simulator episodes. Default guard: `$1`.
- `phase4_core_lab`: baselines, oracle, and one strong OpenAI model on 25
  balanced simulator episodes with 3 repeats. Default guard: `$10`.
- `phase4_frontier_template`: baselines plus GPT-5.5, GPT-5.4, Opus 4.8,
  Fable 5, and Gemini on 25 balanced simulator episodes with 3 repeats.
  Default guard: `$35`.
- `phase4_full_simulator_frontier`: same frontier sweep on all 100 compiled
  simulator episodes with 3 repeats. Default guard: `$150`.

The frontier model IDs are intentionally ordinary JSON fields. If a provider
uses different account-specific names, edit the pack or copy it to a new file.

## Cost Guards

Every paid pack has `max_cost_usd`. The runner estimates total spend before any
API call using the pack's token estimate and per-model prices. If the estimate
exceeds the guard, execution stops before spending.

You can override the guard:

```bash
python3 -m mechbench.cli phase4-study \
  --pack phase4_core_lab \
  --max-cost-usd 3 \
  --dry-run
```

Unknown paid model pricing is blocked by default. Add
`price_per_million_tokens` to the run spec or pass `--allow-unknown-cost`.

Runtime credentials are checked before any paid call. Missing API key or base
URL environment variables stop the run unless `--skip-missing-credentials` is
used.

## Outputs

Each real run writes:

- `study.json`: full machine-readable results, traces when enabled, token usage,
  cost accounting, per-episode scores, experiment sequences, and summary tables.
- `report.md`: human-readable report with total/component scores, per-world
  confusion, intervention-prediction accuracy, experiment sequences, budget use,
  and repeated-run variance.

## Minimal Lab Handoff

Send a lab partner this repository and ask them to run:

```bash
python3 -m unittest discover -s tests
python3 -m mechbench.cli phase4-study --pack phase4_free_smoke --output-dir results/free_check
python3 -m mechbench.cli phase4-study --pack phase4_frontier_template --dry-run
```

If those pass and the preflight cost is acceptable, they can set API keys and
run the frontier pack. The generated `results/.../study.json` and
`results/.../report.md` are the files to send back.
