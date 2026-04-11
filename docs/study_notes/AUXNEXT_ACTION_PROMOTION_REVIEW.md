# Lightweight Auxiliary-Loss Next-Action Comparison Review

## Status

Reviewed and saved for later reference.

Current verdict from fresh 5-seed promotion test:
- **Do not promote** this aux-loss-only variant as the incumbent.

## What was being tested

A train.py-only idea aimed at improving post-evidence commitment timing **without changing the benchmark or decision-prior logic directly**.

### Recommended lightweight aux-loss-only direction

Core idea:
- keep the existing behavior auxiliary loss against the teacher's **current** action
- lightly blend in supervision toward the teacher's **next-step** action when the current state suggests the teacher is about to switch from a soft action to a decisive action

Initial version:
- after evidence release
- non-narrow margin
- current teacher action in `WAIT / YIELD / PROBE`
- next teacher action in `PROCEED / ASSERT`
- blend:
  - `0.75 * current_teacher_CE + 0.25 * next_teacher_CE`

Tightened version that was actually promotion-tested:
- same blended aux-loss idea, but the next-action comparison only activates when the current state is:
  - post-evidence
  - contested
  - partner-soft
  - urgency- or throughput-pressured
  - non-narrow margin
  - current teacher action soft, next teacher action decisive

Why this direction was considered:
- it is **aux-loss-only**
- it avoids direct edits to the benchmark, env, eval boundary, or local runner
- it tries to encourage earlier decisive switching only in the narrow class of situations where residual soft-soft stalling seemed plausible

## Fresh promotion-gate evidence

Fresh output roots used to avoid mixing evidence:
- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/logs/v3omx_auxnextcmp_promo_seed7`
- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/logs/v3omx_auxnextcmp_promo_seed11`
- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/logs/v3omx_auxnextcmp_promo_seed17`
- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/logs/v3omx_auxnextcmp_promo_seed23`
- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/logs/v3omx_auxnextcmp_promo_seed29`

## 5-seed results table

| Seed | Decision | Baseline ToMCoordScore | Candidate ToMCoordScore | Baseline Deadlock | Candidate Deadlock | Baseline Collision | Candidate Collision | Baseline Success | Candidate Success |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7  | keep    | 0.0994 | 0.2542 | 0.10 | 0.10 | 0.35 | 0.25 | 0.55 | 0.60 |
| 11 | keep    | 0.0613 | 0.2817 | 0.10 | 0.10 | 0.40 | 0.20 | 0.50 | 0.65 |
| 17 | keep    | 0.0994 | 0.2097 | 0.10 | 0.10 | 0.35 | 0.30 | 0.55 | 0.60 |
| 23 | discard | 0.1855 | 0.1755 | 0.10 | 0.10 | 0.25 | 0.35 | 0.65 | 0.55 |
| 29 | keep    | 0.1089 | 0.2392 | 0.10 | 0.10 | 0.35 | 0.30 | 0.55 | 0.60 |
| **Mean** | **mixed (4 keep / 1 discard)** | **0.1109** | **0.2321** | **0.10** | **0.10** | **0.34** | **0.28** | **0.56** | **0.60** |

## Aggregate read

Mean metrics favored the candidate:
- baseline `ToMCoordScore`: **0.1109**
- candidate `ToMCoordScore`: **0.2321**
- baseline `DeadlockRate`: **0.10**
- candidate `DeadlockRate`: **0.10**
- baseline `CollisionRate`: **0.34**
- candidate `CollisionRate`: **0.28**
- baseline `SuccessRate`: **0.56**
- candidate `SuccessRate`: **0.60**

But the seed-level stability was not clean enough for promotion:
- seed 23 was a selector **discard**
- on seed 23, the candidate had:
  - lower `ToMCoordScore`
  - higher `CollisionRate`
  - lower `SuccessRate`
- deadlock stayed flat, which is good, but this still breaks the desired promotion robustness

## Recurring weakness note

`false_friend` should now be treated as a recurring independently observed weakness for this repo, not a one-off anomaly.

Why:
- it has reappeared as a failure family across multiple train.py-only ideas
- the failure is not just lower score; it often shows up as the model misreading **soft partner behavior** after evidence release
- this makes it a useful stress family for distinguishing:
  - genuinely cooperative softness
  - opportunistic softness

Implication:
- future experiment notes should explicitly mention whether `false_friend` improved, stayed flat, or worsened
- selector wins that ignore `false_friend` behavior should be treated more cautiously

## Final recommendation

Recommendation for this aux-loss-only family:
- **Keep the notes for future review**
- **Do not promote this tested variant**
- if revisited, treat it as an exploratory branch rather than the incumbent path

## Suggested future follow-up

If this family is revisited later, likely safer directions would be:
- reduce the next-action blend weight further
- gate it even more narrowly on highly specific scenario families
- compare against the current incumbent candidate under fresh 3-seed quick gates before attempting another 5-seed promotion test


## What neutral or near-neutral results clarify

When a refined idea produces results that are effectively identical to the incumbent on a clean seed gate, that is still useful evidence.

For this repo, that kind of result usually means:
- the tested family is probably **not the main lever** right now
- the family may still be safe, but should be **deprioritized** rather than promoted
- future effort should shift toward mechanisms that change action selection more directly at the uncertainty-to-commitment transition

Applied to belief-stabilization:
- safe belief smoothing appears lower-leverage than post-evidence action gating
- the remaining gains likely come more from **belief-to-policy routing** and **commitment timing** than from early belief smoothing by itself

## Straightforward methods to rule families down using existing artifacts

These can all be run from existing artifacts without changing the benchmark:

1. **Seed-level win/loss matrix**
   - Compare candidate vs baseline per seed on ToMCoordScore, DeadlockRate, CollisionRate, SuccessRate, AmbiguityEfficiency, and IntentionPredictionF1.
   - Use this to rule families down when they improve side metrics but repeatedly fail the core deployment metrics.

2. **Leave-one-seed-out mean check**
   - Recompute aggregate means while leaving out each seed once.
   - Use this to detect whether a family's apparent gain is carried by one favorable seed.

3. **Scenario-family delta table**
   - Aggregate `scenario_summaries` by `scenario_family` and compare terminal outcomes, belief-shift moments, and switch delays.
   - Use this to rule families down when they only help one family while quietly hurting another.

4. **Contested-state behavior mix check**
   - Compare `choice_context_counts` / `choice_context_rates` in `high_conflict_bottleneck`, `ambiguous_early`, `high_urgency`, and `normal_flow`.
   - Use this to test whether the mechanism is actually moving the intended decisions.

5. **Mask-rate to outcome check**
   - Compare `experiment_mask_firing_rates` against per-seed outcomes.
   - Use this to rule families down when the new mechanism fires often but does not translate into better selection outcomes.

6. **Regret and timing sanity check**
   - Compare `mean_context_sensitive_action_regret`, `switch_delay_after_evidence`, and `action_switch_moment`.
   - Use this to tell apart real coordination gains from noisy headline metric wins.

## Policy implication for this family

For lightweight auxiliary-loss-only belief-stabilization ideas:
- keep the notes
- treat the family as **deprioritized unless a materially different gating refinement changes the evidence**
- prioritize post-evidence commitment timing and belief-to-policy transition mechanisms before spending large-budget promotion tests here

For the broader post-evidence commitment family:
- keep `false_friend` in the foreground as a named evaluation stressor
- prioritize **post-evidence soft-partner disambiguation** over more generic threshold-only tuning

## Constraint note: true exit pressure is not directly observable

In the current frozen benchmark, remaining steps / timeout pressure are not explicit observation features.

That means:
- "probe gently on exit" is a useful human shorthand
- but within train.py-only work it must be approximated indirectly through recurrent state and combinations of already observed features
- future ideas should be careful to distinguish:
  - true exit-aware behavior (would require observation-surface change)
  - inferred late-episode caution from existing cues

See also:
- `docs/RESEARCH_FAMILY_TRIAGE_POLICY.md`
