# MechanismBench

MechanismBench is a benchmark harness for evaluating whether an automated
researcher can identify the causal source of an apparent ML improvement through
controlled experiments.

This repository currently contains the generic v0 harness:

- family/world discovery;
- controlled agent API;
- deterministic budget accounting;
- immutable experiment evidence ledger;
- structured report and intervention prediction schemas;
- composite scoring;
- CLI scaffolding for evaluation and verification;
- a tiny fixture family used for smoke tests.

The fixture family is intentionally not a real benchmark episode. It exists so
the generic harness can be tested before the simulator and PyTorch families are
implemented.

## Quick Start

```bash
python3 -m unittest discover -s tests
python3 -m mechbench.cli list-worlds
python3 -m mechbench.cli evaluate --agent oracle --worlds all
```

## Development Order

The revised v0 plan is simulator-first. The next layer after this generic
codebase is the factorized simulator and offline episode compiler, followed by
the tiny Transformer retrieval family.

