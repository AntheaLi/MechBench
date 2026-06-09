# Phase 2 Certified PyTorch Suite

Implemented on 2026-06-08. **Validated on 2026-06-08** — all 8 episodes train, run, and score correctly.

This suite is separate from the fast `attn_branch/*_001` and `*_002` smoke episodes. It uses opaque world IDs and larger CPU/GPU-oriented training configs for the four Phase 2 worlds:

- `attn_branch/phase2_certified_0001` — true mechanism
- `attn_branch/phase2_certified_0002` — true mechanism
- `attn_branch/phase2_certified_0003` — seed hacking
- `attn_branch/phase2_certified_0004` — seed hacking
- `attn_branch/phase2_certified_0005` — parameter laundering
- `attn_branch/phase2_certified_0006` — parameter laundering
- `attn_branch/phase2_certified_0007` — norm laundering
- `attn_branch/phase2_certified_0008` — norm laundering

Each episode includes:

- synthetic retrieval model/data configuration;
- structured routing branch and matched controls;
- activation-norm-capable branch variants;
- private held-out interventions;
- frozen target deltas for certified margins;
- `certificate.suite_profile = "phase2_certified"`;
- `certificate.execution_validated = true`.

## Validation results (2026-06-08)

All validation performed with PyTorch 2.12.0+cpu on Linux aarch64.

**Model architecture:** All 5 branch types (none, geometric, generic, random, destroyed) build and forward-pass correctly. Parameter counts: baseline ~117K, branched ~125K (as expected from the 4×d_model×d_inner projection matrices).

**Training:** Models train deterministically with seeded data. Convergence verified: baseline reaches 100% train accuracy in ~10 epochs on the passkey retrieval task. Training produces realistic model statistics including branch/attention norm ratios that differentiate between branch types (geometric: 0.04, generic: 0.23, destroyed: 0.03, random: 0.01).

**Certified delta system:** Verified that `target_deltas` correctly override measured accuracy. Example from phase2_certified_0001: measured_accuracy=0.7295 (real training), but reported accuracy=0.454 and delta=+0.044 (from certified config). Scoring sees the certified values; agents see model stats from real training.

**Episode pipeline:** Naive agent completes phase2_certified_0001 in ~33s (CPU) scoring 59.0. Oracle agent requires ~4min per episode (runs all 8 variants). All 4 attn_branch unit tests pass (previously skipped without torch). Full suite: 64 passed, 0 skipped, 0 failed.

**Task difficulty note:** The passkey retrieval task is currently too easy for the base model — even the no-branch baseline achieves 100% on all splits including OOD. This doesn't affect benchmark scoring (certified deltas override measured accuracy), but means the agent sees unrealistically high measured accuracies. Future work should increase task difficulty (more key-value pairs, longer sequences, smaller models) so the measured training curves are more realistic.
