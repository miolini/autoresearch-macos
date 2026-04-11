# Pre-mission briefing

## Purpose

This document is the final briefing for the coding pass before Azure-oriented orchestration, packaging, and deployment work begins.

Its purpose is to keep the final implementation aligned with the scientific intent of the project and to prevent the codebase from drifting back toward a generic coordination benchmark. The target is a compact Theory-of-Mind benchmark in which **belief inference must change action selection in the right way, at the right time, under ambiguity and context**.

This briefing applies especially to:

- `revised_program.md`
- `mini_benchmark_variant1_ambiguous_bottleneck.md`
- `mini_benchmark_variant2_contextual_right_of_way.md`

---

# Core mission reminder

The project is not trying to prove only that:

- latent intent can be classified
- belief quality can be improved
- coordination reward can be increased by any means

The real target is narrower and more demanding:

**Can a lightweight ToM mechanism improve behaviour by changing policy appropriately under partial observability, ambiguity, and social context?**

The key operational question is:

- when should the learner wait?
- when should it probe?
- when should it yield?
- when should it assert?
- when should it switch once evidence accumulates?

This should remain the centre of the final coding pass.

---

# Bottom-line priorities

These are the most important things to preserve or strengthen.

## 1. Belief-to-policy switching matters more than belief quality alone

Do not treat a better latent-state predictor as a win unless it produces better action.

The benchmark should make it possible to tell the difference between:

- good intent inference with poor control
- good control without meaningful ToM
- genuine belief-guided behavioural adaptation

This distinction is central to the entire project.

## 2. Ambiguity handling must be real

The benchmark should include situations where:

- early evidence is ambiguous
- more than one hidden partner style fits the early trajectory
- premature commitment is dangerous
- waiting too long is also dangerous

If ambiguity is weak or cosmetic, the benchmark collapses into ordinary coordination.

## 3. Context-sensitive action selection must be explicit

Especially in Variant 2, the same inferred partner state should not always imply the same action.

Context should visibly matter, for example through:

- urgency
- safety margin
- norm expectation
- timeout pressure

The final code should make it possible for the policy to use context alongside belief.

## 4. No-progress behaviour is a first-class failure mode

Deadlock, hesitation, and mutual waiting are not side issues. They are a core scientific signal.

The final pass should preserve or strengthen mechanisms that can detect or respond to:

- stagnation
- repeated mutual waiting
- costly over-politeness
- late or absent switching out of no-progress states

## 5. Simplicity is still a hard constraint

The benchmark should remain small and interpretable.

Do not solve the scientific problem by adding too much mechanism. The repo still needs:

- a fixed evaluation boundary
- a single editable surface in `train.py`
- disciplined small changes
- short comparable runs
- interpretable logs

The benchmark should become behaviourally richer, not infrastructurally bloated.

---

# Recommended search directions for the final coding pass

These are the most relevant implementation directions for the two mini benchmarks.

## Highest priority

### Auxiliary next-action prediction for the partner
This is one of the best lightweight ways to improve belief quality in a form that may actually help action selection.

Why it matters:
- supports short-horizon behavioural inference
- aligns with both ambiguity and switching
- is useful in both mini variants

### Belief stabilization
Use KL-style or similar regularization if appropriate to avoid noisy or overconfident early updates.

Why it matters:
- Variant 1 depends on resisting premature commitment
- Variant 2 depends on calibrated context use
- helps separate informative updates from noisy oscillation

### Longer observation history
A modest increase in observation history may help in late-disambiguation and partner-style inference.

Why it matters:
- hidden styles may only separate over time
- the benchmark depends on behavioural evidence, not one-step cues

### Belief-to-policy gating improvements
This is likely more important than many raw architecture changes.

Why it matters:
- the project is about whether inferred belief changes action
- the right question is not only “what does the model believe?” but “how does the policy use that belief?”

### No-progress / pathological hesitation handling
Introduce features or logic that let the policy recognize when waiting has stopped being useful.

Why it matters:
- both mini variants are vulnerable to deadlock
- politeness can become a failure mode
- this is directly relevant to conflict resolution

## Medium priority

### Context-belief fusion
Especially important for Variant 2.

Why it matters:
- the same inferred partner type may call for different actions depending on context
- this is the cleanest bridge to the broader PhD brief

### Small latent intention head
Use this if it maps clearly to the benchmark’s fixed hidden partner styles.

Why it matters:
- useful for structured inference
- helpful if kept interpretable
- strongest in Variant 1

### Small recurrent capacity increase
A modest GRU depth or hidden-size change is reasonable as a local search direction.

Why it matters:
- may improve sequence representation
- should not be mistaken for the main scientific contribution

---

# Lower-priority or conditional directions

## Gaussian latent belief
Only explore this if simpler categorical or structured beliefs plateau.

Reason:
- likely to reduce interpretability
- not clearly necessary for the current benchmark design

## Communication-related prediction heads
Only relevant if communication is already present in the observation space.

Reason:
- otherwise this adds noise and distracts from the main behavioural question

---

# Cautions for the final coding pass

## Do not confuse lower motion with better coordination
A more passive policy may reduce collisions while making the system worse overall.

Watch for:
- paralysis
- deadlock inflation
- excessive delay
- fake safety through inactivity

## Do not confuse better F1 with better ToM use
A model that predicts partner style better but still takes the wrong action is not a meaningful success.

## Do not penalize all waiting
Waiting is often correct under ambiguity.

Penalize:
- avoidable waiting
- repeated no-progress waiting
- waiting that continues after evidence should trigger commitment

Do not penalize:
- short useful caution
- deliberate probing pauses
- genuinely safety-preserving restraint

## Do not let Variant 2 learn a static mapping
If the model learns rules like:
- “assertive partner means yield”
- “cooperative partner means go”

without regard to context, then the benchmark has failed to test the intended question.

## Do not hide regressions inside averages
Scenario-family inspection remains essential.

A change can improve a mean score while failing badly in:
- `false_friend`
- `late_disambiguation`
- `same_belief_different_action`
- `no_progress_switch`

---

# Variant-specific reminders

## Variant 1: Ambiguous Bottleneck Negotiation

Primary scientific target:
- belief-guided switching under ambiguous early evidence

What matters most:
- resisting premature commitment
- avoiding indefinite waiting
- switching at the right moment
- making belief turning points visible in logs

Best implementation emphasis:
- belief smoothing
- partner next-action prediction
- history length
- no-progress handling
- clear action-switch logging

Best narrative affordance:
- this variant is ideal for STORYTELLER because run logs can naturally be converted into compact human-readable incidents after the fact

## Variant 2: Contextual Right-of-Way Negotiation

Primary scientific target:
- socially contingent action selection

What matters most:
- using belief differently across contexts
- handling urgency and norm shifts
- avoiding static partner-type-to-action mappings
- making context-sensitive regret visible in logs

Best implementation emphasis:
- context-belief fusion
- calibrated switching
- no-progress and timeout-aware response
- scenario tags that clearly differentiate context use

Best scientific bridge:
- this variant is the stronger bridge to the original PhD brief because it is closer to transport-like and shared-resource decision settings

---

# Logging requirements to preserve before Azure

Before moving into Azure-oriented orchestration, make sure local outputs are rich enough to survive remote execution and downstream analysis.

Each run should preserve:

- run ID
- benchmark variant
- hypothesis
- exact code change
- main metrics
- scenario-family deltas
- keep/discard decision
- short interpretation

For notable episodes, also preserve:

- scenario family
- partner style
- context tags
- belief turning point
- action switch point
- outcome
- one-line interpretation

This is required for:

- `RESEARCHER.md` timeline building
- `STORYTELLER.md` incident reconstruction
- release-note generation
- Azure artifact traceability

---

# What the “morning output” should now become

Retain the original spirit, but adapt it to the mini benchmarks.

## Keep
- top 10 configs by `ToMCoordScore`
- best checkpoint
- experiment log of accepted and rejected code edits
- comparison plots versus baseline

## Strengthen
Instead of only a generic replay or demo output, prefer:

- a compact replay viewer or artifact set showing:
  - trajectories
  - inferred beliefs
  - belief turning points
  - action switch points
  - scenario family labels
  - context labels where applicable

That is more aligned with the actual scientific purpose of the mini benchmark suite.

---

# Definition of success for the final coding pass

The final coding pass is successful if the repo clearly supports the following claim:

**This benchmark can distinguish between merely predicting another agent and actually using that prediction to make better, safer, and more context-appropriate decisions under uncertainty.**

It is especially successful if:

- Variant 1 reveals whether the model can switch well under ambiguity
- Variant 2 reveals whether the model can act differently under different contexts using the same inferred belief
- logs are strong enough for scientific interpretation and narrative translation
- the implementation remains compact enough for disciplined overnight search and Azure packaging

---

# Final instruction to the coding agent

Before touching Azure-facing packaging or deployment concerns:

- keep the benchmark scientifically sharp
- keep the benchmark small
- preserve comparability
- preserve interpretability
- prefer belief-guided behavioural improvement over abstract sophistication
- make switching, ambiguity, and social contingency easy to evaluate
- ensure logs support both research analysis and downstream storytelling
