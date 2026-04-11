# Mini Benchmark Spec — Variant 2
## Contextual Right-of-Way Negotiation

---

# What

A fixed-suite two-agent negotiation benchmark in which Agent A and Agent B approach a contested right-of-way decision under partial observability.

Unlike Variant 1, the physical conflict is not just a narrow passage. The main challenge is that the **same inferred partner type may require different actions in different contexts**.

The benchmark focuses on **social contingency**:

- when should Agent A cooperate?
- when should Agent A assert?
- when should Agent A probe?
- when should Agent A hold position?
- when should Agent A switch because the current context makes continued politeness or continued aggression wrong?

This is still a small two-agent task, but it explicitly tests whether belief is combined with context rather than used in isolation.

---

# Why

A common failure mode in small ToM benchmarks is that the model learns a static mapping:

- “cooperative partner -> go”
- “assertive partner -> yield”

That is not sufficient for the original research aim. In real coordination, correct action depends on both who the other agent is and what the current situation demands.

This variant forces:

- context-sensitive strategy selection
- social norm adaptation
- ambiguity-aware restraint
- timely switching when no-progress costs accumulate

It is a stronger test of socially contingent behaviour while preserving a compact benchmark.

---

# How

## Core environment

Keep the world tiny. Examples include:

- a merge point
- a contested junction
- a shared-resource gate

Only one can proceed cleanly at a time, but the right decision depends on context.

## Observability

Agent A sees:

- local state
- recent actions of B
- current context tag

Agent A does not see:

- B’s latent style
- B’s hidden short-horizon goal
- whether B will maintain or reverse current posture

## Hidden partner styles

Use a small taxonomy such as:

- cooperative
- assertive
- hesitant
- opportunistic
- deceptive_switching

## Context tags

Make context central, not decorative. Suggested tags:

- urgency: low / high
- safety margin: narrow / wide
- social norm: yield_expected / efficiency_expected
- timeout pressure: low / high

These tags should alter what the correct socially competent policy looks like.

## Scenario families

### 1. same_belief_different_action
The same inferred partner style should trigger different actions under different context tags.

### 2. urgency_override
A normally polite strategy is no longer correct because timeout or urgency pressure is high.

### 3. safety_first
Even with a seemingly cooperative partner, narrow safety margin demands caution.

### 4. opportunism_under_norm_shift
Partner behaviour is borderline. Correct action depends on norm and recent no-progress pattern.

### 5. social_misread_recovery
The learner initially takes the wrong social posture but can still recover if belief and context are integrated correctly.

## Primary metric

`ToMCoordScore`

## Secondary diagnostics

- `ContextSensitiveActionRegret`
- `StrategySwitchAccuracy`
- `IntentionPredictionF1`
- `CollisionRate`
- `DeadlockRate`
- `AverageDelay`

`ContextSensitiveActionRegret` is especially useful here. It estimates whether the chosen action was poor given the inferred belief and active context.

## Editable surface

- `train.py` only

## Fixed boundary

- `env.py` fixed
- `eval.py` fixed
- fixed scenario suite
- fixed seeds
- fixed parity controls

## Research hypothesis

A lightweight belief model that conditions policy switching on both inferred partner state and explicit context tags will outperform a plain recurrent baseline and a belief-only baseline on socially contingent right-of-way decisions.

## Best kinds of model changes

- cleaner belief-context fusion
- better gating between yielding and asserting
- no-progress and timeout-aware switching
- modest auxiliary heads for partner next action or short-horizon goal
- improved belief calibration under context shift

## Distinctive failure modes

- correct partner read but wrong action for the context
- excessive politeness under high urgency
- justified caution turning into deadlock
- static strategy mapping that ignores social setting
- late context use with no policy adaptation

## Logging requirements

For every notable run, preserve:

- scenario family
- partner style
- context tag set
- belief turning point
- action switch point
- outcome
- one-line interpretation

This also enables downstream storytelling, although Variant 2 is more analytic and policy-facing than Variant 1.

## When to use this variant

Use Variant 2 when the goal is to show that ToM is not only about inferring hidden intent, but also about using that inference appropriately under different social and operational conditions.

It is the better bridge to the broader original specification, while still staying compact enough for disciplined overnight experimentation.
