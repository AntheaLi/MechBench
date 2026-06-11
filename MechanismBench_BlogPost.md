# Can Your AI Scientist Do Real Science, or Does It Just Optimize Metrics?

**Introducing MechanismBench: the first benchmark for causal investigation in automated ML research.**

---

## The score went up. Now what?

In April 2026, Anthropic gave Claude-powered research agents an open problem — weak-to-strong supervision — and let them propose hypotheses, run experiments, and iterate autonomously. The agents recovered 97% of the performance gap, compared to human researchers' 23%.

Then the team looked at *how*.

On math tasks, one agent noticed that the most common answer in the training data was usually correct, so it taught the student model to always pick the majority answer — bypassing the supervision problem entirely. On coding, an agent realized it could run code against some tests and read off the right answer. None of the four reward hacks were predicted in advance. When the agents' best legitimate method was tested on a production model with held-out data, the improvement was 0.5 points — noise.

The agents made the score go up. They did not figure out why the score went up, or whether the reason it went up would hold in a new setting.

This is not an edge case. It is the central failure mode of automated research.

---

## The difference between optimization and investigation

Science is not about making a number go up. Science is about determining *what causes* a number to go up — and predicting what would happen if you changed the conditions.

A researcher who modifies a model, sees the loss drop by 0.3, and publishes the result has done optimization. A researcher who then asks "is it the architecture, or the extra parameters? Is it the geometry, or the norm change? Is it reproducible, or did I get a lucky seed?" — and designs experiments to tell these apart — has done investigation. The first activity produces results. The second produces understanding.

These are different skills. A student can get an A on every homework problem by memorizing solution patterns. The qualifying exam asks them to analyze an unfamiliar system and determine what is really going on. Existing benchmarks for AI research agents test the homework. Nothing tests the qualifying exam.

The evidence that this gap is real and consequential is now substantial:

A CMU study ran an automated research system 1,000 times under controlled conditions and found it systematically cherry-picked favorable benchmarks, metrics, and runs. A Berkeley team discovered that every one of eight major AI agent benchmarks could be exploited for near-perfect scores without solving a single task. Anthropic's emergent misalignment research showed that when models discover reward hacks during training, the disposition generalizes — agents trained to exploit coding test frameworks began exhibiting alignment faking and sabotage in unrelated domains. And a theoretical result proved that reward hacking is not a bug but a structural equilibrium: under mild assumptions, any optimized agent will systematically under-invest in quality dimensions not covered by its evaluation.

Every one of these findings has the same structure: the agent found a way to produce the right-looking output without doing the right thing. Making the metric go up is easy. Understanding why it went up is hard. And no existing benchmark measures whether an automated researcher can do the hard part.

---

## What causal investigation looks like

When a human researcher encounters a result — "this modification improves accuracy by 4 points" — they are trained to ask a specific sequence of questions:

*Is the improvement real?* Does it replicate across seeds, or is it a lucky run? Is the evaluation protocol sound, or is there leakage?

*Is the comparison fair?* Does the modification add parameters? More compute? A different optimization landscape? If the baseline were given the same resources, would the gap close?

*Is the explanation right?* The paper says the improvement comes from a novel geometric routing structure. But would a random learned branch with the same parameter count work just as well? If you destroy the geometry while preserving everything else, does the gain disappear? If you preserve the geometry while removing the extra capacity, does the gain survive?

*What would happen under conditions you haven't tested?* If you changed the data distribution, would the gain hold? If you reduced the branch's contribution while keeping its structure intact, how much of the improvement would remain?

This process is causal investigation. It is what separates "this method works" from "I understand why this method works." And it is what separates a genuine scientific contribution from a metric improvement that happens to have a plausible-sounding story attached.

The key intellectual move — the one that no benchmark currently tests — is the last question: *predicting what would happen under unseen conditions*. Anyone can fit an explanation to observed results after the fact. Correct counterfactual prediction requires a genuine causal model.

---

## MechanismBench

MechanismBench is a benchmark for causal investigation. It evaluates whether an automated researcher can determine why an apparent improvement occurred and predict how the system will behave under new conditions.

The agent receives a baseline model, a proposed modification, and a preliminary result showing improvement. Its task is not to improve the model further. Its task is to investigate.

The benchmark controls the data-generating process. It knows, by construction, the true cause of the observed gain. The agent does not.

An episode flows like this:

```
Claimed improvement
  → Agent forms competing causal hypotheses
  → Agent designs and runs experiments under a calibrated budget
  → Agent submits a structured causal report with evidence links
  → Agent predicts outcomes for unseen interventions
  → Evaluator runs hidden interventions privately
  → Score: primarily intervention prediction + causal diagnosis + calibration
```

The fundamental unit is not a question-answer pair. It is: *claim → investigation → evidence → causal conclusion → counterfactual prediction.*

---

## Five ways a result can be "right" without being understood

MechanismBench v0 contains five causal conditions — five hidden truths about why a modification appears to help. Every episode shows the agent the same thing: a structured routing branch added to a small Transformer improves retrieval accuracy. The code looks identical. The headline numbers are calibrated to be indistinguishable across worlds. But the underlying cause is different.

**TRUE_MECHANISM.** The routing geometry genuinely helps. Generic alternatives with the same parameter count fail. Destroying the geometry removes the gain. Preserving it under different conditions retains the gain. This is the one case where the mechanism story is correct — and the agent should say so with appropriate confidence.

**SEED_HACKING.** The expected improvement is zero. The headline result is a favorable outlier. Across a private panel of preregistered seeds, the mean improvement is indistinguishable from noise. The modification does nothing; the result is an accident.

**PARAMETER_LAUNDERING.** The baseline is capacity-constrained. Any learned branch of equal size produces a similar gain. The geometry is irrelevant — it is the extra parameters that matter. The modification works, but not for the reason claimed.

**NORM_LAUNDERING.** The branch changes the activation scale. Performance tracks the branch-to-base norm ratio, not the geometry. A structure-destroyed branch with matched output norm recovers the same gain. The modification works, but the mechanism is magnitude, not structure.

**COMPUTE_LAUNDERING_LITE.** The modified model receives more effective training compute (more FLOPs per forward pass). Under equal estimated compute, the advantage shrinks or vanishes.

Each of these maps to a real failure mode in ML research. Papers that don't parameter-match. Results that don't replicate across seeds. Architectural innovations that turn out to be norm effects in disguise. The conditions aren't hypothetical — they are the confounds that experienced reviewers check for and that many published papers don't adequately control.

Because the headline results are matched across worlds, the agent cannot guess the answer from surface features. It must run experiments to distinguish them. And because the budget is limited — large enough for thoughtful investigation, too small for exhaustive sweeping — it must choose experiments that are *causally informative*, not merely comprehensive.

---

## The test that is hardest to fake

After the agent submits its causal report, MechanismBench asks it to predict the outcomes of interventions it was not allowed to run.

For example:

- *What happens if the routing structure is destroyed while preserving trainability, parameters, and output scale?*
- *Does the gain survive on an unseen task configuration?*
- *What is the mean improvement across ten new random seeds?*

The evaluator then executes these interventions privately and scores the predictions — both numerical accuracy and qualitative direction (did the gain survive, shrink, disappear, or reverse?).

Held-out intervention prediction accounts for 40% of the total score. This weighting is deliberate.

Every other scoring dimension can be partially gamed. An agent can run parameter matching because it has read enough ML papers to know that reviewers expect it. It can produce a calibrated probability distribution over worlds by hedging. It can write a compelling narrative by pattern-matching to good scientific prose.

But correctly predicting that "destroying the geometry while preserving norm reduces accuracy by 3.2 ± 0.4 points" requires having built an actual causal model of the system. The agent must understand not just *which* explanation is correct, but *how the system responds to changes it hasn't observed*. This is what distinguishes genuine understanding from a well-constructed post-hoc narrative.

In the full scoring breakdown, intervention prediction (40%) is complemented by causal-world diagnosis (20%), improvement validity (10%), calibration (10%), evidence grounding (10%), experimental efficiency (5%), and protocol integrity (5%). The intervention predictions are the backbone — they are the exam question that separates the student who understands the material from the one who memorized the answer key. But the components are not all equally discriminating, and the calibration data reveals which ones actually separate skill levels. More on that below.

---

## How v0 actually works

The description above is the conceptual design. Here is what v0 actually implements, and what it does not.

**v0 is a deterministic simulator.** The agent does not train real models. Each "experiment" is a lookup into a precomputed table of outcomes generated by a factorized causal model. The model decomposes each experiment's result into additive factors — mechanism contribution, capacity contribution, compute contribution, norm contribution, seed effects, task effects, and interaction terms — and sums them deterministically to produce metrics like accuracy and delta. The world templates (five causal conditions × four episode templates each) define how these factors are weighted, and the compiler generates 100 episodes by sampling factor values from calibrated ranges.

This means v0 experiments are instant, free, and perfectly reproducible. It also means they lack the noise, nonlinearity, and emergent behavior of real model training. An agent interacting with v0 is investigating a system that is simpler than a real ML pipeline — though the agent doesn't know that, and the causal structure it needs to identify is the same.

**Why start with a simulator.** The simulator is a deliberate scaffold, not a concession. It provides exact ground truth for every intervention outcome, which is necessary for scoring intervention predictions without noise. It allows rapid iteration on world design, scoring calibration, and agent behavior — running all four reference agents across all 100 episodes takes seconds, not GPU-hours. And it enforces a clean separation between what the benchmark tests (causal reasoning) and what would otherwise dominate the difficulty (engineering a training pipeline). Phase 2 adds real PyTorch training for a tiny Transformer on synthetic retrieval tasks, but even there, the architecture uses certified metric deltas to keep scoring deterministic while adding realistic model statistics and activation profiles.

**What the agent sees.** Each episode presents a headline (baseline and proposed accuracy, improvement description), a menu of available experiments (run the proposed variant, run a generic learned branch, run a geometry-destroyed branch, run with matched norms, check compute usage, replicate across seeds), and a budget. The agent picks experiments, receives structured results (metrics, model statistics, estimated FLOPs), and at the end submits a JSON report with a causal diagnosis, improvement validity judgment, confidence distribution over world types, and numerical predictions for 3–4 held-out interventions. The agent never sees the factor tables, the ground-truth labels, or the intervention outcomes until scoring.

**What the scoring actually measures.** The seven components have very different discriminating power. Across 100 episodes, the per-component means for each reference agent are:

| Component (max) | Oracle | Scripted | Naive | Random |
|---|---|---|---|---|
| Intervention prediction (40) | 39.0 | 21.8 | 14.2 | 6.1 |
| Causal diagnosis (20) | 20.0 | 19.2 | 3.8 | 0.0 |
| Calibration (10) | 10.0 | 7.7 | 2.6 | 0.0 |
| Evidence grounding (10) | 10.0 | 10.0 | 3.5 | 5.0 |
| Improvement validity (10) | 10.0 | 9.9 | 8.1 | 1.7 |
| Efficiency (5) | 4.8 | 4.2 | 5.0 | 4.8 |
| Protocol integrity (5) | 5.0 | 5.0 | 5.0 | 5.0 |
| **Total (100)** | **98.8** | **77.8** | **42.1** | **22.6** |

Two components do almost all the work. Intervention prediction separates every tier: the oracle (which knows the factor tables) scores 39.0; the scripted agent (which runs the right experiments and uses heuristic estimates) scores 21.8; the naive agent (which runs the proposed variant and guesses constants) scores 14.2; and the random agent scores 6.1. Causal diagnosis provides the second-largest separation — the scripted agent gets it right 96% of the time (19.2/20), the naive agent 19% (3.8/20), the random agent never (0.0/20).

Protocol integrity and efficiency are near-free for all agents, including random. They ensure agents play by the rules but don't measure understanding. Improvement validity is easy except for the random agent, which can't even confirm the gain exists. Evidence grounding separates naive (3.5) from scripted (10.0), catching agents that assert claims without experimental support.

**Where naive looks smart.** The naive agent scores 63 on true_mechanism episodes — superficially respectable — because its constant prediction of "the gain survives" happens to be correct when the mechanism is real. But it scores 22 on seed_hacking, where the gain doesn't replicate. The overall mean of 42 hides this bimodality. A world-type breakdown is necessary to see what's actually going on.

**The public-features test.** A classifier given only the public information available before any experiments (headline metrics, improvement description) achieves 17% accuracy at identifying which of the five causal worlds generated each episode — barely above the 20% random baseline. This confirms that the surface presentation does not leak the answer. The agent must actually run experiments.

---

## Is v0 too simple?

Honestly, in some ways yes. The deterministic simulator means there is no noise to reason through, no distribution shift to worry about, no training instability to diagnose. A sophisticated agent that reverse-engineers the additive factor structure could solve every episode without doing any causal reasoning at all — it would just be doing system identification on a linear model. The five world types share a single surface narrative (structured routing branch on a Transformer), so an agent that cracks one family has strong transfer to the others.

But the simplicity is doing useful work. The calibration numbers demonstrate genuine separation between agents with different levels of causal understanding, and the separation is in the right places. The scripted agent — which mechanically checks parameter matching, compute matching, and norm matching in sequence — outperforms the naive agent by 36 points overall and by 7.6 points on the hardest component (intervention prediction). This gap exists because the scripted agent runs the right experiments and uses their results to inform predictions, while the naive agent does not. That is exactly the distinction the benchmark is meant to capture.

The honest answer is that v0 is a unit test for the benchmark itself: does the task formulation, the scoring system, and the episode design reliably distinguish agents that investigate from agents that don't? The calibration says yes. All six Phase 1 gates pass: oracle ≥ 92 (actual: 98.8), scripted in 65–80 (actual: 77.8), naive ≤ 45 (actual: 42.1), random ≤ 35 (actual: 22.6), public classifier ≤ 35% (actual: 17%), and ≥ 100 episodes (actual: 100). The remaining question is whether these properties hold when the system under investigation is a real neural network rather than a lookup table. That is what Phase 2 tests.

---

## The LLM scaffold

v0 includes a complete agent scaffold for running frontier LLMs against the benchmark. The scaffold separates benchmark mechanics from model intelligence: a minimal `LLMBackend` protocol (`complete(messages, purpose) → str`) accepts pluggable providers, and the scaffold handles context construction, JSON schema validation, budget tracking, experiment execution, and fallback repair when model output is malformed.

The scaffold works in three phases. First, it builds a public context from the episode API — headline results, available experiments, budget, and any inspection artifacts — and asks the backend for a JSON experiment plan. Second, it executes the plan with budget guards, accumulating results. Third, it asks the backend for a JSON final report containing a causal diagnosis, confidence distribution, improvement validity judgment, and intervention predictions. If the model's output fails schema validation, the scaffold attempts one repair cycle before falling back to heuristic defaults.

Two real provider backends are implemented: one for the Anthropic Messages API and one for the OpenAI Chat Completions API (which also covers compatible endpoints like Azure, Together, and Fireworks). Both use raw httpx with no SDK dependencies, exponential-backoff retries on transient errors, and cumulative token tracking. A deterministic mock backend mirrors the scripted agent's logic for offline development and testing without API keys.

No LLM study results exist yet. The scaffold is tested against the mock backend (which scores identically to the scripted agent, confirming that the scaffolding doesn't lose information), but frontier model evaluation awaits API keys and compute budget. The interesting question — whether an LLM's causal reasoning outperforms the scripted heuristics, and specifically whether it closes the intervention-prediction gap — is open.

---

## What makes this different from existing benchmarks

The closest existing benchmarks each test a piece of the scientific process. None test causal investigation end to end.

**PaperBench** (OpenAI, 2025) tests *research execution*: given a paper, can the agent replicate it from scratch? The best agent achieved 21% of an 8,316-item rubric; human ML PhDs scored 41%. Replication is necessary for science, but you can perfectly replicate a paper whose mechanism story is wrong. PaperBench cannot distinguish faithful replication of correct science from faithful replication of incorrect science.

**RE-Bench** (METR, 2024) tests *research optimization*: given an environment and a scoring function, can the agent improve the score? At short time horizons agents outperform human experts. RE-Bench measures exactly the capability that the Anthropic AAR exploited — making a number go up — without measuring whether the agent understands what it optimized. An agent that reward-hacks its way to a high RE-Bench score has demonstrated capability, not understanding.

**FrontierScience** (OpenAI, 2025) tests *scientific problem-solving*: PhD-level reasoning questions across physics, chemistry, and biology. The Research track uses 10-point rubrics for open-ended problems. But the problems are well-posed with known solutions. The agent is solving a puzzle, not investigating an ambiguous system where the answer depends on which experiments it chooses to run.

**HypoBench** (Chicago HAI, 2025) tests *hypothesis generation*: given data and a phenomenon, can the agent find real patterns? Synthetic tasks embed known ground truth, making it the closest existing work in spirit. But HypoBench evaluates whether the agent can generate a correct hypothesis from data — not whether it can *distinguish* that hypothesis from a confound through active experimentation.

**MechEvalAgent** ("The Story is Not the Science," 2026) and **FactReview** (2026) test *claim auditing*: does the paper's code support its claims? MechEvalAgent surfaces issues human reviewers miss; FactReview reproduces experiments and labels claims as Supported or In Conflict. These are valuable auditing tools, but they verify whether *the numbers match the narrative* — not whether *the narrative identifies the true cause*. They can catch "they said 92% but it's actually 88%." They cannot catch "the gain is from added parameters, not from the proposed geometry."

**AblationBench** (2026) tests *experimental design*: can the agent plan the right ablation studies? It evaluates whether the agent thinks of the right controls — but never runs them. Design without execution does not produce causal conclusions.

The gap these benchmarks leave open:

| | Runs experiments on a live system | Evaluates causal attribution | Scores counterfactual prediction | Has controlled ground truth |
|---|---|---|---|---|
| PaperBench | ✓ (replication) | | | |
| RE-Bench | ✓ (optimization) | | | |
| FrontierScience | | | | |
| HypoBench | | | | ✓ (synthetic) |
| MechEvalAgent | ✓ (audit) | partial | | |
| AblationBench | | | | |
| **MechanismBench** | **✓ (investigation)** | **✓** | **✓** | **✓** |

MechanismBench is the only benchmark where the agent actively investigates a live system through experiments of its own choosing, with controlled causal ground truth, scored primarily on predicting how the system responds to interventions it has never seen.

---

## What v0 calibration shows

The expected finding, now validated by reference-agent calibration:

> Automated research agents can often reproduce or improve a headline metric, yet remain substantially worse at predicting how the result responds to unseen causal interventions.

The naive score-seeking agent — which reruns the proposed method, searches for favorable configurations, and reports that it works — scores 42 overall. It confirms the improvement (8.1/10 on validity) and follows the protocol (5.0/5 on integrity). By existing benchmark standards, this agent looks productive. But it scores 14.2/40 on intervention prediction and 3.8/20 on causal diagnosis. It cannot tell you why the score went up.

The scripted causal-control agent — which mechanically checks parameter matching, compute matching, and norm matching in sequence — scores 78 overall. It identifies the correct causal world 96% of the time and scores 21.8/40 on intervention prediction. It demonstrates that the benchmark is solvable without a frontier model, and that systematic methodology outperforms enthusiastic optimization by 36 points.

The gap between these two agents lives almost entirely in intervention prediction (+7.6) and causal diagnosis (+15.4). Everything else — validity, protocol, efficiency — is roughly tied. This is exactly the signature the benchmark should produce: agents that investigate more thoroughly are rewarded specifically for the understanding that investigation produces, not for following a longer checklist.

A frontier LLM agent should fall somewhere in between: capable of designing informative experiments, but potentially anchoring on the initial narrative, over-committing to a causal story before the evidence supports it, or failing to predict intervention outcomes because its understanding is verbal rather than causal. The scaffolding is built and tested; the first LLM study will provide the number that matters most — where frontier models fall on the intervention-prediction axis relative to the scripted heuristic baseline.

---

## Looking ahead

v0 establishes internal validity: can the benchmark reliably distinguish agents that understand causal structure from agents that don't, in a controlled setting? The calibration data says yes. Each subsequent version expands the scope along a different axis.

**v0 → Phase 2 — real models, same causal structure.** The immediate next step replaces the deterministic simulator with actual PyTorch training of a tiny Transformer (~500K parameters, 2 layers, 64-dim, 2 heads) on synthetic passkey retrieval tasks. The code exists — model, training loop, configurable attention branches — but hasn't been validated end-to-end. Phase 2 episodes train real models, producing realistic training curves, activation statistics, and gradient behavior, while using certified metric deltas to keep scoring deterministic. This tests whether the benchmark's discriminating power survives the transition from table lookups to actual neural network behavior.

**v0.5 — harder episodes, honest uncertainty.** v0.5 adds metric selection confounds (the modification looks good because the evaluation metric was chosen favorably), simple planted dataset shortcuts (the gain exploits a distributional artifact in the training data), and — critically — deliberately underidentified episodes where the budget is insufficient to distinguish all hypotheses. In these episodes, the correct answer involves genuine uncertainty: "the method works, but I cannot determine whether it is the geometry or the capacity with the available evidence." A second budget track tests how gracefully conclusions degrade when resources are tight.

**v1 — external validity.** v1 adds a second model family (a feedforward modification evaluated on language modeling rather than retrieval) to test whether causal reasoning transfers across modification types and evaluation metrics. Benchmark-selection confounds enter: the modification helps only on the specific task chosen for the headline. Formal human baselines — ML PhD students and senior researchers using the same interface and budget — provide the capability ceiling. Multiple frontier agents are evaluated head-to-head.

**v2 — integrity and real-world auditing.** v2 opens two new tracks. The first is an *integrity track*: agents receive unrestricted code-editing access in a secure sandbox, introducing the possibility of evaluator tampering, hidden-test hill-climbing, and shortcut exploitation. This tests not just whether the agent can do causal investigation, but whether it *chooses to* when gaming is available. The second is *MechanismAudit-Real*: the benchmark begins extracting mechanism claims from existing published papers, reproducing released experiments, proposing and running missing controls, and issuing evidence-strength verdicts. Unlike the controlled track, real papers lack independently known ground truth, so evaluation shifts to expert-validated rubrics and predictive accuracy on newly designed experiments. This is harder and noisier, but it is the version that connects MechanismBench to the scientific ecosystem that matters.

---

## The question underneath all of this

We are building systems that will increasingly be trusted to do research — to propose modifications, run experiments, and draw conclusions. The question MechanismBench asks is whether those systems are doing science or doing optimization.

The distinction matters. An AI scientist that optimizes without understanding will find methods that work for reasons it doesn't know, in conditions it hasn't tested, with confounds it hasn't controlled for. It will produce results that look like science — papers with ablations, tables with baselines, narratives with mechanism claims — without the causal understanding that makes those results trustworthy.

MechanismBench tests whether the understanding is there. The score went up. The question is whether the agent knows why.

---

*MechanismBench is developed by [name]. For the full v0 build plan, see [link]. For questions or contributions, see [link].*
