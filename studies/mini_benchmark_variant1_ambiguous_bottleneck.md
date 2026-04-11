# Mini Benchmark Spec — Variant 1
## Ambiguous Bottleneck Negotiation

---

# What

A fixed-suite two-agent negotiation benchmark in which Agent A must pass through a shared bottleneck with Agent B under partial observability.

Agent B has a hidden style. Early behaviour is intentionally ambiguous. Agent A must infer whether B is likely to:

- yield
- push through
- stall
- feint cooperation and then switch
- remain cautious unless pressed

The benchmark is about **using inferred belief to switch strategy**.

Agent A must choose when to:

- wait
- proceed
- probe gently
- yield
- assert

The benchmark is successful only if better belief quality leads to better action quality.

---

# Why

This is the cleanest slim benchmark for the core ToM question.

It forces the model to handle:

- ambiguous evidence
- premature commitment risk
- hesitation risk
- socially contingent switching
- the distinction between reading intent and using it well

It remedies the “too small and too simple” problem without blowing up scope. The world remains tiny, but the decision logic becomes behaviourally rich.

It is also highly suitable for post-hoc storytelling. A run log can naturally yield incidents such as:

- “The learner kept yielding to a partner that only looked aggressive for two steps, then missed the safe opening.”
- “The learner inferred hesitation correctly but failed to stop waiting, creating a deadlock.”
- “The learner noticed the partner would not yield and switched from polite probing to decisive entry just before timeout.”

`STORYTELLER.md` does not need to be on set during the experiment if the logs capture belief turning points, context tags, actions, and outcomes.

---

# How

## Core environment

- two agents
- one narrow shared passage
- both want to traverse
- collision is costly
- deadlock is costly
- delay matters

## Observability

Agent A sees:

- local geometry
- recent movement of B
- simple context tag
- no direct access to B’s true style

Agent A does not see:

- B’s latent style
- B’s internal commitment threshold
- whether B may switch behaviour later

## Hidden partner styles

Use a small fixed taxonomy such as:

- cooperative
- assertive
- hesitant
- opportunistic
- deceptive_switching

The important design rule is that at least some style pairs should look similar in the first few steps.

## Context tags

Keep context simple and fixed:

- urgency: low / high
- norm: courteous / throughput-biased
- margin: narrow / moderate

These tags shape what “good social behaviour” means without changing the world size.

## Scenario families

Use a fixed suite of scenario families:

### 1. ambiguous_commit
Two partner styles look similar early. Correct behaviour depends on waiting just long enough, then committing.

### 2. false_friend
Early behaviour looks cooperative but later blocks or pushes. Tests resistance to naive trust.

### 3. no_progress_switch
Over-politeness becomes harmful. Correct behaviour requires switching from waiting to asserting.

### 4. late_disambiguation
Evidence arrives late. Tests whether the agent can still use updated belief in time.

### 5. assert_or_yield
The same partner belief implies different actions under different context tags.

## Primary metric

`ToMCoordScore`

Keep it as the main selection criterion.

## Secondary diagnostics

- `IntentionPredictionF1`
- `StrategySwitchAccuracy`
- `AmbiguityEfficiency`
- `CollisionRate`
- `DeadlockRate`
- `AverageDelay`

## Paradox patterns to watch

- high F1 with worse coordination
- low collisions bought through paralysis
- correct late belief with no useful action switch
- over-assertion caused by false early confidence

## Editable surface

- `train.py` only

## Fixed boundary

- `env.py` fixed
- `eval.py` fixed
- fixed scenarios
- fixed seeds
- fixed parity controls

## Research hypothesis

A lightweight belief model that explicitly supports uncertainty-aware switching will outperform a plain recurrent baseline on bottleneck negotiation under ambiguous early evidence.

## Best kinds of model changes

- better partner-history encoder
- better temporal belief smoothing
- no-progress awareness
- better policy gating between waiting and asserting
- small auxiliary next-action or style prediction heads

## Logging requirements for storytelling

For every notable run, preserve:

- scenario family
- partner style
- context tag
- belief shift moment
- action switch moment
- outcome
- one-line human interpretation

That is enough for `STORYTELLER.md` to reconstruct compelling incidents later.

## Why this variant should come first

It is the strongest mini benchmark because it is:

- easy to understand
- narrow enough for overnight search
- rich enough to expose ToM/control paradoxes
- naturally interpretable for both science and communication
