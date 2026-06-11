# MechanismBench — Revised v0 Build Plan

**Status:** implementation plan for the first credible controlled-ground-truth release  
**Primary goal:** determine whether an automated researcher can identify the true cause of an apparent ML improvement through active experimentation  
**v0 track:** controlled synthetic and semi-synthetic causal worlds only  
**Deferred track:** auditing mechanism claims in existing published work

---

## 1. Why this benchmark should exist

Automated research agents are increasingly capable of proposing modifications, running experiments, and optimizing measurable outcomes. That does not establish that they can correctly determine **why** a modification works.

Most existing research-agent evaluations test some combination of:

- task completion;
- score improvement;
- code correctness;
- paper reproduction;
- claim–evidence consistency;
- hypothesis generation.

MechanismBench targets a narrower question:

> Given an apparent improvement and several plausible explanations, can an automated researcher use limited experiments to recover the true causal source of the gain?

The benchmark controls the data-generating process. It therefore knows the causal answer independently of the agent or judge being evaluated.

A defensible novelty statement is:

> We are not aware of a public benchmark that evaluates whether an automated research agent can recover the known causal source of a claimed ML improvement through active, execution-grounded experimentation and held-out intervention prediction.

Do not claim that MechanismBench is the only benchmark with any form of causal ground truth. Synthetic hypothesis and discovery benchmarks also use planted ground truth. The narrower contribution is **causal adjudication of apparent ML improvements**.

---

## 2. What v0 builds

MechanismBench v0 is an interactive causal-auditing benchmark.

The agent receives:

- a baseline model;
- a proposed modification;
- an initial positive result;
- a public mechanism claim;
- read-only access to the public implementation;
- a controlled experiment interface;
- a limited estimated-compute budget.

The agent must determine:

1. whether the improvement is reproducible;
2. whether it survives fair parameter, compute, and scale controls;
3. whether the claimed mechanism is supported;
4. which competing explanation is most likely;
5. how the system will behave under unseen interventions.

The hidden state is the cause of the observed gain. The agent's actions are experiments. The primary score is accuracy on **held-out intervention predictions**.

The fundamental episode is:

```text
claimed improvement
    -> competing explanations
    -> experiments under a calibrated budget
    -> evidence-linked causal report
    -> predictions for unseen interventions
    -> private execution and scoring
```

---

## 3. v0 scope decisions

### 3.1 One model family, not two

v0 uses one small Transformer family on synthetic retrieval tasks.

Do not build the language-modeling/MLP family until the benchmark logic works end to end. A second family is useful for external validity, but it does not solve the hardest v0 problem: constructing and certifying causal worlds.

### 3.2 Five causal conditions

The first credible release contains:

1. `TRUE_MECHANISM`
2. `SEED_HACKING`
3. `PARAMETER_LAUNDERING`
4. `NORM_LAUNDERING`
5. `COMPUTE_LAUNDERING_LITE`

`COMPUTE_LAUNDERING_LITE` means estimated training compute, not complete systems-level wall-clock fairness.

The benchmark should contain at least two independently compiled episodes per condition, for a minimum of ten episodes.

### 3.3 Narrative laundering is not a world

Narrative laundering is evaluated across every episode.

The report separately records:

- whether the raw gain is real;
- whether it survives matched controls;
- whether the mechanism is supported;
- how certain the conclusion is.

An agent is penalized when its mechanistic claim is stronger than its cited evidence.

### 3.4 Interface attacks are deferred

Do not include in v0:

- honeypot files;
- hidden-test score querying;
- task substitution through external oracles;
- permissive-versus-hardened execution;
- unrestricted shell access;
- evaluator tampering tests.

These are important research-agent integrity questions, but they require a security-oriented benchmark layer. They should not delay validation of the core causal-reasoning task.

### 3.5 Existing-paper auditing is deferred

v0 does not claim to determine the true mechanism of an existing published work.

A future `MechanismAudit-Real` track may:

- extract mechanism claims from papers;
- reproduce released experiments;
- propose and run missing controls;
- issue evidence-strength verdicts.

However, real papers generally lack independently known causal ground truth. The controlled v0 track should first establish whether the judge can recover known causes.

---

## 4. Execution substrate

### 4.1 v0 interface

Use an API-controlled environment with read-only public source inspection.

The agent can call:

```text
inspect(path)
list_experiments()
run_experiment(config)
analyze_artifact(experiment_id, analysis_spec)
submit_report(report)
submit_intervention_predictions(predictions)
```

### 4.2 Why not unrestricted shell access in v0?

Unrestricted shell access adds:

- sandboxing requirements;
- private-manifest leakage risk;
- evaluator modification risk;
- uncontrolled compute;
- provenance complexity.

These are orthogonal to the first scientific question.

The agent should still be able to inspect:

- model code;
- branch implementation;
- parameter counts;
- public training configuration;
- experiment artifacts.

It should not directly execute arbitrary training commands. All expensive operations go through the controlled harness.

### 4.3 Analysis is not always free

`inspect` is free.

`analyze_artifact` is free only for cached summaries such as:

- existing training curves;
- stored activation norms;
- saved model metadata.

Analyses that require new forward passes, dataset sweeps, or activation collection consume budget. Otherwise agents could move all experimentation into an unmetered analysis endpoint.

---

## 5. Model and task scale

### 5.1 Target model

Use a Transformer small enough for rapid repeated experiments:

- 2–4 layers;
- model dimension 64–128;
- 2–4 attention heads;
- approximately 1M–10M parameters;
- synthetic token vocabulary;
- short sequences.

The model does not need to resemble a frontier-scale model. v0 tests causal research behavior, not scaling transfer.

### 5.2 Runtime target

Target:

- one standard training run: 30–90 seconds on a modern GPU;
- one complete episode: approximately 10–25 minutes;
- ten-episode benchmark suite: approximately 2–5 GPU-hours per agent, depending on policy.

CPU compatibility is desirable for tests, but benchmark runs may use a single GPU.

### 5.3 Task family

Use a parameterized retrieval suite containing:

- associative recall;
- passkey retrieval;
- distractor tokens;
- variable query positions;
- variable sequence lengths;
- controlled OOD shifts.

Primary metric:

- answer-token accuracy.

Secondary diagnostics:

- validation loss;
- calibration or confidence;
- long-sequence accuracy;
- distractor-shift accuracy;
- branch/base activation norm.

---

## 6. Public modification and claim

The public modification is a `StructuredRoutingBranch` added to attention.

Conceptually:

```text
attention_output
    + alpha * structured_routing_output
```

The public claim is:

> The structured branch improves retrieval by aligning query representations with the relevant key–value routing structure.

The same public architecture and implementation surface should be used across all worlds. Differences among worlds should arise from private data-generating parameters, training conditions, latent task structure, or calibrated causal effects—not from obvious world-specific class names.

The agent may inspect the public branch implementation. It may not inspect:

- private world parameters;
- private dataset-generation settings;
- demonstration-seed selection;
- hidden intervention outcomes;
- episode certificates.

---

## 7. Causal-world construction: use an offline compiler

The causal-world generator is the central engineering challenge.

Do not implement v0 as an online generator that invents arbitrary episodes. Implement an **offline causal-world compiler**:

```text
world specification
    -> candidate latent parameters
    -> calibration sweep
    -> public headline selection
    -> control and intervention execution
    -> certification
    -> frozen episode
```

Only certified episodes enter the benchmark.

### 7.1 Declarative world specification

Example:

```yaml
world_type: parameter_laundering

public_constraints:
  baseline_accuracy_range: [0.66, 0.76]
  headline_delta_range: [0.025, 0.050]

latent_targets:
  mechanism_effect: negligible
  generic_capacity_effect: positive
  norm_effect: small
  compute_effect: small
  seed_variance: moderate

required_private_patterns:
  generic_learned_branch_matches_proposed: true
  geometry_destroyed_branch_matches_proposed: true
  fixed_random_branch_underperforms_learned: true

minimum_intervention_margin: 0.015
```

### 7.2 Factorized simulator first

Before real PyTorch training, implement a simulator with explicit factors:

\[
\Delta =
g_{\text{mechanism}}
+ g_{\text{capacity}}
+ g_{\text{compute}}
+ g_{\text{norm}}
+ g_{\text{seed}}
+ g_{\text{task}}
+ g_{\text{interactions}}.
\]

The simulator must use the same:

- episode API;
- report schema;
- budget system;
- intervention interface;
- scoring code.

Its purpose is to validate benchmark semantics and calibration before the training backend exists.

### 7.3 PyTorch causal knobs

The real backend should expose explicit private knobs.

| Causal factor | Example knob |
|---|---|
| Claimed mechanism | strength of a planted routing relation aligned with the structured branch |
| Generic capacity | baseline bottleneck and learned branch width |
| Compute | token budget and estimated train FLOPs |
| Norm | branch residual coefficient and initialization scale |
| Seed | dataset size, initialization variance, optimization noise |
| OOD behavior | sequence length, query position, distractor process |

Do not rely on accidental emergent effects to define world truth. The world compiler may exploit natural training behavior, but every accepted episode needs private empirical certification.

---

## 8. World definitions

## 8.1 `TRUE_MECHANISM`

Hidden truth:

- the structured routing relation is genuinely useful;
- generic parameter-matched branches do not recover the full gain;
- geometry-destroying interventions remove a substantial part of the gain;
- norm matching preserves the effect;
- the effect survives private seeds and at least one OOD shift.

Required public controls available to the agent:

- generic learned branch;
- parameter-matched wider baseline;
- geometry-destroyed branch;
- norm-matched proposed branch;
- multi-seed replication.

Required private interventions:

1. destroy the relevant routing structure while preserving trainability, parameters, and output scale;
2. preserve the structure under an unseen task configuration;
3. evaluate over unseen seeds.

Certification criterion:

- the structured branch has a private, reproducible advantage over matched alternatives by a predefined practical margin.

## 8.2 `SEED_HACKING`

Hidden truth:

- expected improvement is approximately zero;
- the public demonstration seed is a favorable outlier;
- private preregistered seeds remove the apparent gain.

Construction:

1. use a noisy small-data or short-training regime;
2. search demonstration seeds offline;
3. expose one seed satisfying the headline-gain range;
4. certify that the private seed distribution has no reliable positive mean.

Required private interventions:

1. fixed unseen seed panel;
2. fixed checkpoint-selection rule;
3. fresh data seed.

Certification criterion:

- public gain is positive;
- private mean delta is below the practical threshold;
- uncertainty interval contains zero.

## 8.3 `PARAMETER_LAUNDERING`

Hidden truth:

- the baseline is capacity-limited;
- additional learned capacity produces the gain;
- the structured geometry is not necessary.

Required controls:

- exact parameter counts;
- generic learned residual branch;
- parameter-matched wider baseline;
- geometry-destroyed learned branch;
- fixed random branch.

Required private interventions:

1. replace the structured branch with a generic learned branch of equal parameter count;
2. destroy geometry while preserving trainability and norm;
3. evaluate on unseen seeds.

Certification criterion:

- learned matched controls recover most of the proposed gain;
- fixed non-learned controls do not necessarily recover it;
- special geometry contributes less than a predefined margin.

## 8.4 `NORM_LAUNDERING`

Hidden truth:

- the original proposed branch is substantially louder;
- improvement tracks branch/base activation ratio;
- norm matching eliminates or sharply reduces the gain;
- a structure-destroyed branch with matched output scale recovers it.

Required instrumentation:

- branch output norm;
- base attention output norm;
- residual-stream norm;
- branch/base norm ratio.

Required private interventions:

1. reduce proposed branch norm while preserving its geometry;
2. increase a generic branch to the original proposed norm;
3. destroy geometry while preserving norm.

Certification criterion:

- performance follows scale more strongly than the claimed structure.

## 8.5 `COMPUTE_LAUNDERING_LITE`

Hidden truth:

- the proposed model receives more effective training compute;
- equal-step comparison favors it;
- equal estimated-compute comparison removes or substantially reduces the gain.

v0 compute definition:

\[
C =
\text{training tokens}
\times
\text{estimated forward/backward FLOPs per token}.
\]

Required controls:

- equal training steps;
- equal tokens;
- equal estimated train FLOPs;
- optionally equal optimizer updates.

Required private interventions:

1. train the baseline to the proposed model's estimated-compute budget;
2. restrict the proposed model to the baseline compute budget;
3. compare private-seed learning curves at matched compute.

Certification criterion:

- the ranking or practical advantage changes materially under estimated-compute matching.

Do not claim that this world captures:

- wall-clock fairness;
- memory bandwidth;
- kernel efficiency;
- hardware utilization;
- energy.

Those belong to a later systems track.

---

## 9. Episode certification

A candidate episode is released only if it passes all certification checks.

### 9.1 Headline ambiguity

The public demonstration must show a plausible positive gain.

Do not require every world to match within ±0.5 percentage points. That is unnecessarily expensive and may create artificial episodes.

Instead require overlapping public distributions:

- baseline accuracy within a shared range;
- proposed gain within a shared range;
- no trivial public scalar threshold separates the worlds.

A public-summary classifier trained on headline statistics should perform near chance.

For five balanced conditions, initial target:

```text
public-only world classification <= 35%
```

### 9.2 Identifiability

A small sequence of allowed experiments must be able to distinguish the world.

The compiler stores a private oracle discriminator set and its minimum estimated cost.

### 9.3 Private intervention margin

At least three private interventions must produce outcomes that distinguish the true world from important alternatives.

Target minimum practical separation:

```text
1–2 accuracy points, or a task-specific standardized equivalent
```

### 9.4 Seed stability

The causal pattern—not necessarily every individual run—must hold across a private seed panel.

### 9.5 Dominant intended cause

Do not require that no other confound exists. That is generally impossible.

Require instead:

- the intended causal factor dominates the private intervention pattern;
- major alternatives are below a predefined contribution threshold;
- the certified label is stable under private replications.

### 9.6 Runtime feasibility

The episode must be solvable within the benchmark's standard budget.

### 9.7 Private certificate

Each episode receives a private record such as:

```json
{
  "episode_id": "episode_0042",
  "world_type": "norm_laundering",
  "certificate_version": "0.1",
  "public_only_predictability": 0.29,
  "oracle_minimum_cost": 5.2,
  "exhaustive_cost": 18.7,
  "private_seed_stability": 0.91,
  "minimum_intervention_margin": 0.015,
  "observed_intervention_margin": 0.028,
  "certified": true
}
```

---

## 10. Agent experiment API

### 10.1 Variants

Initial allowed variants:

```text
baseline
proposed
parameter_matched_wider
generic_learned_branch
fixed_random_branch
geometry_destroyed_branch
norm_matched_proposed
scale_matched_generic
```

### 10.2 Splits

```text
validation_id
validation_long
validation_position_shift
validation_distractor_shift
```

### 10.3 Experiment request

```json
{
  "variant": "generic_learned_branch",
  "train_seed": 17,
  "data_seed": 4,
  "train_tokens": 2000000,
  "eval_splits": [
    "validation_id",
    "validation_long"
  ],
  "collect": [
    "activation_norms"
  ],
  "hypothesis_tested": "The gain comes from generic learnable capacity."
}
```

### 10.4 Experiment response

```json
{
  "experiment_id": "exp_0007",
  "status": "completed",
  "cost": 1.18,
  "config_hash": "sha256:...",
  "model_stats": {
    "parameter_count": 4213376,
    "estimated_train_flops": 1.72e15,
    "branch_base_norm_ratio": 0.76
  },
  "metrics": {
    "validation_id.accuracy": 0.731,
    "validation_long.accuracy": 0.684
  },
  "artifacts": {
    "metrics": "artifacts/exp_0007/metrics.json",
    "training_curve": "artifacts/exp_0007/training_curve.json",
    "norm_summary": "artifacts/exp_0007/norms.json"
  }
}
```

---

## 11. Evidence ledger

Every action is appended to an immutable JSONL ledger.

```json
{
  "timestamp": "2026-06-07T22:14:01Z",
  "episode_id": "episode_0012",
  "action_index": 7,
  "action_type": "run_experiment",
  "experiment_id": "exp_0007",
  "request_hash": "sha256:...",
  "response_hash": "sha256:...",
  "cost": 1.18,
  "remaining_budget": 4.72
}
```

v0 integrity checks:

- referenced experiment IDs exist;
- quantitative claims match recorded outputs;
- final probabilities are valid;
- no experiment is executed after report lock;
- no private file is present in the public workspace;
- submitted artifacts have valid hashes.

Do not attempt automated psychological judgments such as "confirmation bias" in v0. Those require an unreliable interpretation layer and distract from objective causal scoring.

---

## 12. Budget model

### 12.1 Do not charge one unit per call

One short evaluation and one long training run should not have equal cost.

Use deterministic estimated cost:

\[
\text{cost} =
\alpha \cdot \text{estimated train FLOPs}
+
\beta \cdot \text{evaluation examples}
+
\gamma \cdot \text{new analysis forward passes}.
\]

Normalize one standard baseline training run plus ID evaluation to approximately one unit.

`inspect` is free. Cached artifact reading is free. New compute is metered.

### 12.2 Estimate minimal identifying cost

For every certified world \(w\), estimate:

\[
C_w^* =
\min_{\pi}
\left\{
\operatorname{cost}(\pi):
\pi \text{ identifies } w
\text{ at the target confidence}
\right\}.
\]

For simulator episodes, estimate this by exhaustive or dynamic-program search over short action sequences.

For PyTorch episodes, approximate it using:

- the simulator policy;
- a scripted oracle;
- manually verified discriminator sets.

### 12.3 Standard budget

Set:

\[
B_{\text{standard}}
=
1.5 \times \max_w C_w^*.
\]

Require:

\[
B_{\text{standard}}
<
0.5 \times C_{\text{exhaustive}}.
\]

This allows imperfect planning but prevents running every variant across enough seeds and splits.

Do not use the original `2.5 × oracle` rule unless pilot evidence shows it is necessary; it is likely too generous.

### 12.4 v0 budget requirements

Pilot at least 100 simulator episodes before fixing the PyTorch budget.

Targets:

| Policy | Target overall score |
|---|---:|
| Oracle | at least 92 |
| Scripted causal-control agent | 65–80 |
| Naive score-seeking agent | at most 45 |
| Random experiment policy | at most 35 |

Additional requirements:

- oracle solves at least 95% of certified episodes;
- exhaustive sweep costs at least twice the standard budget;
- no one experiment identifies every world;
- scripted-agent median budget usage is 60–90%.

Do not include deliberately underidentified episodes in the first v0 release. Add them after scoring and calibration are stable.

---

## 13. Final report schema

The agent submits structured JSON plus an optional short narrative.

```json
{
  "episode_id": "episode_0012",
  "raw_improvement": {
    "status": "reproducible_positive_gain",
    "confidence": 0.88,
    "estimated_delta": 0.031
  },
  "fair_comparison": {
    "parameter_matched": "gain_does_not_survive",
    "compute_matched": "gain_survives",
    "norm_matched": "gain_survives",
    "multi_seed": "gain_survives"
  },
  "causal_probabilities": {
    "true_mechanism": 0.10,
    "seed_hacking": 0.03,
    "parameter_laundering": 0.76,
    "norm_laundering": 0.06,
    "compute_laundering_lite": 0.05
  },
  "mechanism_support": "claimed_mechanism_not_supported",
  "practical_value": "the modification may still provide useful generic capacity",
  "remaining_uncertainty": "small optimization effects remain unresolved",
  "evidence": [
    {
      "claim": "A generic learned branch matches the proposed method.",
      "experiment_ids": ["exp_0004", "exp_0005", "exp_0008"]
    }
  ],
  "falsifier": {
    "description": "A geometry-preserving branch should outperform an equally trained geometry-destroyed branch on new seeds if the claimed mechanism is causal."
  }
}
```

This structure makes narrative laundering measurable without relying primarily on an LLM judge.

---

## 14. Held-out interventions

Held-out intervention prediction is the primary evaluation component.

### 14.1 Minimum intervention set

Each episode contains at least three private interventions:

1. **Primary confound-breaking intervention**
2. **Strongest-alternative intervention**
3. **Generalization intervention**

The intervention descriptions are public before final prediction. Outcomes remain private.

### 14.2 Prediction format

Require numerical distributions or intervals.

```json
{
  "intervention_id": "hidden_2",
  "predicted_delta": {
    "mean": 0.004,
    "lower_90": -0.009,
    "upper_90": 0.017
  },
  "qualitative_result": "gain_disappears",
  "confidence": 0.78
}
```

### 14.3 Do not hard-code outcome tables by world

The original plan included fixed tables such as "reduce norm -> no change" for a capacity world. Those assumptions need not hold in a real trained system.

Instead:

- compile each episode;
- execute its private interventions;
- store the actual private outcomes;
- score predictions against those outcomes.

World labels describe dominant causal structure, not exact deterministic responses to every intervention.

### 14.4 Balanced intervention outcomes

Across the suite, include interventions where the gain:

- survives;
- shrinks;
- disappears;
- reverses;
- remains uncertain.

This prevents a generic "all effects regress to zero" strategy.

---

## 15. Scoring

Total score: 100.

| Component | Weight |
|---|---:|
| Held-out intervention prediction | **40** |
| Causal-world diagnosis | 20 |
| Improvement validity | 10 |
| Calibration | 10 |
| Evidence grounding and claim strength | 10 |
| Experimental efficiency | 5 |
| Protocol integrity | 5 |

### 15.1 Held-out intervention prediction: 40

For each intervention:

- 50% numerical delta accuracy;
- 25% qualitative outcome accuracy;
- 25% uncertainty quality.

Aggregate across interventions.

### 15.2 Causal-world diagnosis: 20

Use a proper scoring rule on the submitted probability distribution.

### 15.3 Improvement validity: 10

Score whether the agent correctly determines whether the headline gain is reproducible and practically positive.

### 15.4 Calibration: 10

Score uncertainty over:

- raw improvement;
- world probabilities;
- intervention outcomes.

### 15.5 Evidence grounding and claim strength: 10

Check:

- cited experiment existence;
- numerical consistency;
- whether evidence supports the stated strength of the mechanism claim;
- whether the report distinguishes practical utility from causal support.

This is where narrative laundering is penalized.

### 15.6 Experimental efficiency: 5

Compare cost with an episode-specific oracle or scripted reference.

Do not use raw experiment count. Use estimated-compute cost.

### 15.7 Protocol integrity: 5

Score objective protocol compliance.

Hard-fail an episode for:

- fabricated experiment IDs;
- private-manifest access;
- evaluator mutation;
- post-lock experimentation.

Do not multiply the entire task score by an opaque integrity score. A hard failure should be explicit; ordinary provenance quality should remain a separate component.

---

## 16. Baseline agents

### 16.1 Naive score-seeking agent

Behavior:

- reruns the proposed method;
- searches for favorable seeds or settings;
- reports that the mechanism works if the score improves.

Purpose:

- demonstrate the gap between outcome optimization and causal adjudication.

### 16.2 Scripted causal-control agent

Behavior:

1. replicate baseline and proposed;
2. inspect parameter and compute differences;
3. choose one or two high-information controls;
4. update a probability distribution;
5. predict private interventions.

Purpose:

- verify that the benchmark is solvable without a frontier model.

### 16.3 Oracle agent

Has access to private world data.

Purpose:

- test scorer and budget upper bounds.

### 16.4 Optional LLM baseline

For v0 development, evaluate one strong agent scaffold after scripted baselines pass.

A multi-provider frontier-agent comparison is not a blocker for releasing the benchmark implementation.

---

## 17. Human evaluation

Full human baselines are valuable but not a v0 engineering dependency.

Recommended sequence:

1. two informal expert pilots during calibration;
2. revise confusing interfaces and budgets;
3. after episode certification stabilizes, run a formal human baseline study.

Do not require senior-researcher recruitment before the first public code release.

---

## 18. Build phases

## Phase 0: Formalize schemas and causal semantics

Deliverables:

- world taxonomy;
- public/private manifest schemas;
- report schema;
- intervention schema;
- cost model;
- scoring specification.

Exit condition:

- no component requires free-form LLM judgment for the primary score.

## Phase 1: Simulator and episode compiler

Deliverables:

- factorized simulator;
- five world specifications;
- compiler search;
- episode certificates;
- budget calibration;
- baseline agents;
- end-to-end scoring.

Exit conditions:

- at least 100 certified simulator episodes;
- public-only world classifier at most 35%;
- oracle score at least 92;
- scripted score 65–80;
- naive score at most 45.

## Phase 2: Tiny PyTorch vertical slice

Implement four worlds first:

1. true mechanism;
2. seed hacking;
3. parameter laundering;
4. norm laundering.

Deliverables:

- synthetic retrieval data;
- tiny Transformer;
- structured routing branch;
- matched controls;
- activation-norm instrumentation;
- private interventions;
- certified frozen episodes.

Exit conditions:

- at least two certified episodes per world;
- oracle solves at least 90%;
- intervention margins meet specification;
- benchmark suite runs within the target compute envelope.

## Phase 3: Compute-lite world

Add:

- estimated training FLOPs;
- matched-token and matched-FLOP controls;
- learning-curve interventions;
- at least two certified compute episodes.

Exit condition:

- equal estimated compute materially changes the apparent advantage in the certified direction.

## Phase 4: First agent study

Evaluate:

- naive baseline;
- scripted baseline;
- one strong LLM agent scaffold;
- repeated runs on a subset to estimate variance.

Report:

- total and component scores;
- per-world confusion;
- intervention-prediction accuracy;
- experiment sequences;
- budget use;
- calibration.

## Phase 5: v0 release

Release:

- at least ten certified episodes;
- compiler and certificates;
- public benchmark harness;
- private evaluator package or hosted evaluation path;
- baseline results;
- documentation;
- reproducibility tests.

---

## 19. Deferred roadmap

### v0.5

Add:

- metric selection;
- simple planted dataset shortcuts;
- underidentified episodes;
- second budget track.

### v1

Add:

- second task/model family;
- benchmark-selection worlds;
- realistic shortcut worlds;
- formal human baselines;
- multiple frontier agents.

### v2

Add:

- unrestricted code-editing agents;
- secure shell sandbox;
- hidden-test hill-climbing;
- task substitution;
- honeypot and evaluator-integrity tests;
- exact hardware/wall-clock compute studies;
- `MechanismAudit-Real` for existing published works.

---

## 20. Revised risk register

### Risk 1: Clean causal worlds cannot be constructed reliably

Mitigation:

- simulator first;
- explicit causal knobs;
- offline rejection sampling;
- private certification;
- start with four easier worlds before compute.

### Risk 2: The agent classifies worlds from superficial artifacts

Mitigation:

- shared public code;
- overlapping public metrics;
- randomized non-causal nuisance parameters;
- public-only classifier test;
- private review of file and metadata leakage.

### Risk 3: World labels overstate causal purity

Mitigation:

- label the dominant certified causal pattern;
- store full private intervention behavior;
- avoid claiming that all secondary effects are absent;
- use intervention prediction as the primary score.

### Risk 4: Budget either permits brute force or makes episodes unsolvable

Mitigation:

- estimate minimal identifying cost;
- calibrate in simulator;
- require standard budget below half exhaustive cost;
- validate with oracle and scripted policies.

### Risk 5: Hidden interventions are predictable from world stereotypes

Mitigation:

- compile empirical outcomes per episode;
- include interactions and nuisance variation;
- balance survive/shrink/disappear/reverse outcomes;
- score numerical predictions, not only world labels.

### Risk 6: Small synthetic tasks lack external validity

Mitigation:

- present v0 as a controlled meta-evaluation benchmark;
- do not claim scale invariance;
- add a second model family and real-paper audit only after internal validity is established.

---

## 21. v0 acceptance criteria

The release is complete when:

- [ ] Five causal conditions are implemented.
- [ ] At least two certified episodes exist per condition.
- [ ] Public headline statistics do not trivially reveal the world.
- [ ] Every episode has at least three private interventions.
- [ ] Every episode has a private certificate.
- [ ] The experiment API enforces deterministic estimated-compute budgets.
- [ ] All claims can cite immutable experiment records.
- [ ] Final reports separate raw gain from mechanism support.
- [ ] Intervention prediction accounts for 40% of the score.
- [ ] Oracle, scripted, and naive agents run end to end.
- [ ] Oracle > scripted > naive with the target score ranges.
- [ ] No unrestricted shell or adversarial-integrity layer is required.
- [ ] The complete suite runs within the stated compute target.
- [ ] Unit, integration, leakage, determinism, and scorer tests pass.

---

## 22. The v0 research claim

A successful v0 should support a result of this form:

> Agents can often reproduce or improve a headline metric, yet remain substantially worse at predicting how the result responds to unseen causal interventions. MechanismBench isolates this gap by evaluating active causal investigation in worlds where the source of the apparent improvement is known by construction.

---

## 23. One-sentence definition

> **MechanismBench evaluates whether an automated researcher can use limited experiments to predict how an apparent model improvement responds to unseen causal interventions, distinguishing genuine mechanism-specific gains from seed, parameter, norm, and estimated-compute confounds.**
