# Design Decisions

The generic harness follows the codebase architecture document while respecting
the revised v0 plan:

- `Family` owns shared model/task/action infrastructure.
- `World` owns one certified causal configuration inside a family.
- Agents interact only through `AgentAPI`.
- Expensive work is mediated by `run_experiment`.
- Evidence is logged as immutable JSONL records.
- Primary v0 scoring is objective and structured; unrestricted shell access,
  honeypots, and adversarial integrity layers are deferred.

Config files are named `.yaml`, but the bootstrap currently writes
JSON-compatible YAML so the repository has no required runtime dependencies.
Installing PyYAML later enables broader YAML syntax automatically.

