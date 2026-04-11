# Research Family Triage Policy

## Purpose

This note defines how to treat experiment families after positive, mixed, neutral, or negative evidence.

A "family" means a recurring class of train.py-only ideas such as:
- belief stabilization
- post-evidence commitment timing
- belief-to-policy gating
- deadlock-specific micro-shaping
- lightweight auxiliary losses

Known recurring scenario weakness to track independently:
- `false_friend`
  - treat this as a named stress family when reviewing results
  - especially check whether the model is confusing cooperative softness with opportunistic softness after evidence release

## Default evidence ladder

Use this order:
1. smoke run for breakage only
2. 1-seed 800-episode gate for initial signal
3. 3-seed quick gate for direction check
4. 5-seed promotion gate for incumbent decisions

## Family outcomes

### Promote family
Use when:
- 5-seed gate passes cleanly
- deadlock is not worse on any seed
- mean ToMCoordScore is higher
- mean collision is lower or equal
- mean success is higher or equal
- no seed is an obvious selector discard for core metrics

### Keep family active
Use when:
- 1-seed or 3-seed evidence is positive
- or 5-seed evidence is mixed but still scientifically interesting
- deadlock is not worse, but stability or consistency is not strong enough for promotion

### Deprioritize family
Use when:
- repeated experiments are effectively neutral
- gains are too small to separate from existing incumbent behavior
- the family appears safe but not high-leverage

This is the default label for ideas that do not hurt but also do not clearly move the benchmark.

### Retire family for now
Use when:
- repeated experiments worsen deadlock
- or cause repeated selector discards
- or lose on the same seed patterns more than once

Retire does not mean never revisit; it means stop spending near-term iteration budget there.

## Straightforward methods to clarify direction using existing artifacts

These methods should use only existing run artifacts such as:
- `selection.json`
- `metrics.json`
- `choice_analysis.json`
- `scenario_summaries`
- experiment-mask firing counts/rates

### 1. Seed-level win/loss matrix
For each seed, mark whether the candidate beat baseline on:
- ToMCoordScore
- DeadlockRate
- CollisionRate
- SuccessRate
- AmbiguityEfficiency
- IntentionPredictionF1

Purpose:
- distinguishes broad improvement from fragile one-metric wins
- rules families down when they repeatedly win on secondary metrics but lose core deployment metrics

### 2. Leave-one-seed-out mean check
Compute 5 means, each leaving out one seed.

Purpose:
- tests whether the apparent gain depends on one unusually favorable seed
- rules families down if the conclusion flips when one seed is removed

### 3. Scenario-family delta table
Aggregate `scenario_summaries` by `scenario_family` and compare candidate vs baseline on:
- terminal outcome counts
- switch delay after evidence
- belief shift moment
- human-interpretation categories if useful

Purpose:
- shows whether a family helps only one scenario family while hurting another
- rules families down when a broad-looking gain is actually a narrow scenario-specific artifact

### 4. Contested-state behavior mix
From `choice_context_counts` and `choice_context_rates`, compare action mixes in:
- `high_conflict_bottleneck`
- `ambiguous_early`
- `high_urgency`
- `normal_flow`

Purpose:
- clarifies whether a change is actually shifting policy where intended
- rules families down if the claimed mechanism is not visible in the relevant contexts

### 5. Mask-rate to outcome check
Use `experiment_mask_firing_rates` and terminal outcomes.

Purpose:
- asks whether the mechanism that was added is firing in runs that actually improve results
- rules families down if the new mask fires often but does not correlate with better outcomes

### 6. Regret and switch-timing sanity check
Use:
- `mean_context_sensitive_action_regret`
- `switch_delay_after_evidence`
- `action_switch_moment`

Purpose:
- separates true coordination gains from noisy metric wins
- rules families down when ToMCoordScore rises but switch timing or regret gets worse

## Observability constraint: exit pressure is not directly visible

For the frozen current benchmark, true exit pressure / remaining steps are **not directly present in the observation vector**.

Implication:
- any "probe gently on exit" behavior learned under the current benchmark must be approximated indirectly
- direct exit-aware routing would require benchmark or observation-surface changes and should therefore be treated as a separate future design choice, not quietly slipped into train.py-only iterations

Current train.py-only work should assume that exit-aware behavior can only come from:
- recurrent state / sequence memory
- bottleneck geometry
- evidence-released state
- partner softness / pressure cues
- belief comparisons

## Policy for neutral results

If a family yields results that are effectively identical to the current incumbent on a clean 1-seed or 3-seed gate:
- mark the family **deprioritized**, not promoted
- save a note describing what the neutral result suggests mechanistically
- do not spend a 5-seed promotion gate on that family unless a new refinement changes the activation logic materially

## Policy for mixed results

If a family wins on mean metrics but has one clear weak seed:
- do not promote
- save a review note
- identify whether the weak seed is tied to:
  - one scenario family
  - one context type
  - one behavior mode (for example, post-evidence overcommit or soft-soft stalling)
- the next follow-up should narrow the mechanism specifically around the failure mode

If the weak seed repeatedly maps back to the same scenario family (for example `false_friend`):
- treat that scenario family as a standing stress test
- mention it explicitly in future review notes
- prefer targeted branch-selection fixes over broad threshold retuning
