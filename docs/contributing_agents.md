# Contributing Agents

Agent adapters implement:

```python
class MyAgent(MechBenchAgent):
    def investigate(self, api):
        ...
        return report, predictions
```

The API exposes:

- `inspect(path)`
- `list_files()`
- `list_actions()` / `list_experiments()`
- `run_experiment(config)`
- `analyze_artifact(experiment_id, analysis_spec)`
- `get_interventions()`
- `submit_predictions(predictions)`
- `submit_report(report)`

Agents should cite experiment IDs in the final report evidence ledger. The
generic scorer gives no evidence-grounding credit for uncited or fabricated IDs.

