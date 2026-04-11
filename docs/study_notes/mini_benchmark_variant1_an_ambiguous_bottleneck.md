# Mini Benchmark Spec — Variant 1

## An Ambiguous Bottleneck

---

Improve a compact Theory-of-Mind (ToM) reinforcement learning system overnight under a fixed experiment budget.

The target is no longer generic coordination. The target is **belief-guided behavioural adaptation**:

- when to cooperate
- when to assert
- when to wait
- when to probe
- when to switch strategy once evidence accumulates

The central research question is:

**Which lightweight belief mechanism most improves policy switching under ambiguity and social context, without breaking coordination safety or throughput?**

This repository is a disciplined overnight research loop. Keep the code small. Keep the comparisons fair. Keep the changes focused.

---

# Objective Metric

Optimize the primary scalar metric:

`ToMCoordScore`

Treat it as the main selection criterion.

`ToMCoordScore` should summarize:

- task success
- coordination efficiency
- safe completion
- effective action under ambiguity

with penalties for:

- collisions
- deadlocks
- pathological hesitation
- unstable behaviour

Do not silently redefine the metric. Do not optimize secondary metrics at the expense of the primary score.

Secondary diagnostics are used to interpret whether the model is succeeding for the right reason.

Important secondary diagnostics:

- `IntentionPredictionF1`
- `StrategySwitchAccuracy`
- `AmbiguityEfficiency`
- `DeadlockRate`
- `CollisionRate`

Interpretation rule:

- better intent prediction is only useful if control improves too
- lower collisions are only useful if they are not bought by gridlock
- a good belief model must help choose the right action, not merely describe the partner

---

# Experimental Setup

Assume a fixed-suite two-agent negotiation benchmark with the following properties:

- two agents
- partial observability
- one learning agent (Agent A)
- one partner agent (Agent B)
- B has a hidden style, intention, or short-horizon goal
- early evidence is often ambiguous
- correct behaviour depends on both inferred intent and context
- evaluation scenarios and seeds are fixed for comparability
- runs are short and repeatable

The benchmark is intentionally small, but behaviourally rich enough to force:

- strategy switching
- ambiguity handling
- socially contingent behaviour

Examples of context tags:

- urgency: low / high
- safety margin: narrow / wide
- norm: yield-biased / efficiency-biased

Examples of hidden partner styles:

- cooperative
- assertive
- hesitant
- opportunistic
- deceptive-switching

The benchmark is valid only if belief quality and control quality remain separable but coupled.

---

# Editable Surface

You may edit:

- `train.py`

You may read:

- all repository files needed to understand the setup
- logs
- metrics
- run summaries
- profiles and configuration files

You may run:

- training
- evaluation
- existing analysis scripts
- existing orchestration scripts

You may write:

- checkpoints
- run summaries
- experiment notes
- bounded machine-readable artifacts already consistent with repository workflow

---

# Non-Negotiable Constraints

## File boundaries

- Only edit `train.py`.
- Do not edit `env.py`.
- Do not edit `eval.py`.
- Do not edit benchmark definitions.
- Do not rewrite metric logic.
- Do not broaden the editable surface without explicit instruction.

## Evaluation integrity

- Do not change evaluation seeds.
- Do not change scenario tags.
- Do not change parity controls.
- Do not remove penalties to create appearance gains.
- Do not compare runs under mismatched conditions.

## Scope control

- Do not introduce framework migrations.
- Do not add large dependencies.
- Do not bloat the repository.
- Do not turn the benchmark into a general platform.
- Do not chase broad novelty when a small focused change will do.

## Scientific discipline

- Do not treat intention prediction as success by itself.
- Do not keep a change that improves one seed through a worse failure mode elsewhere.
- Do not ignore paradox signals.
- Do not hide regressions behind averages.
- Do not overfit to a narrow subset of scenarios.

---

# Agent Team and Stage Ownership

The agent team remains in place, but the operating style is slim.

## Main orchestrator

- `.github/agents/tom-overnight-researcher.agent.md`

## Team

- `PLANNER.md`
- `CRITIC.md`
- `RESEARCHER.md`
- `CONTROLLER.md`
- `STORYTELLER.md`
- `ACCOUNTANT.md`

## Ownership rule

- PLAN -> `PLANNER.md`
- CRITIQUE -> `CRITIC.md`
- EXECUTE -> `CONTROLLER.md`
- LEARN -> `RESEARCHER.md`

Supporting roles:

- `ACCOUNTANT.md` tracks cost, throughput, and heavy-run policy
- `STORYTELLER.md` translates grounded results into human-relatable incidents after the fact, using logged runs and timeline artifacts rather than live participation

The loop should remain tight. Team depth must not become process bloat.

---

# Good Research Directions

Prefer small, interpretable changes that can credibly help under ambiguity.

## Belief modeling

- improve the partner-history encoder
- improve latent style or intent inference
- improve temporal smoothing of belief updates
- improve how belief enters the action policy

## Policy switching

- better gating between cooperative and assertive actions
- explicit no-progress detection
- lightweight probing behaviour before commitment
- better response to late disambiguation

## Context sensitivity

- better use of urgency, margin, or norm tags already present in observation
- better distinction between safe caution and harmful hesitation
- better switching when the same partner belief implies different actions in different contexts

## Auxiliary learning

- partner next-action prediction
- partner latent-style prediction
- short-horizon goal prediction
- belief consistency losses

## Stability and training quality

- mild regularization
- modest optimizer improvements
- conservative reward-shaping refinements
- small curriculum refinements only if they clearly support ambiguity handling

---

# Bad Research Directions

Avoid:

- giant rewrites
- many unrelated edits in one run
- opaque complexity
- brittle hacks
- expanding environment scope
- broad hyperparameter sweeps without a hypothesis
- anything unlikely to pay off in short comparable runs
- reducing interpretability without a clear gain
- treating a more confident belief state as a win when policy behaviour is still poor

---

# Research Loop

## 1. Inspect the current baseline

Understand:

- how partner history is represented
- how belief is computed
- how the policy uses belief
- where ambiguity failure currently appears
- which scenario families are weak

## 2. State one hypothesis

Examples:

- “A smoother belief update will reduce premature commitment in ambiguous cases.”
- “A small no-progress feature will improve socially contingent switching from yielding to asserting.”
- “An auxiliary next-action loss will improve ambiguity handling enough to raise coordination score.”

Keep the hypothesis narrow and testable.

## 3. Make one focused change

Prefer one small change per experiment.

## 4. Run a comparable experiment

Respect the existing parity controls.

## 5. Compare against the best accepted run

Check:

- `ToMCoordScore`
- scenario-family behaviour
- collisions
- deadlocks
- switch behaviour
- whether gains look real or noisy

## 6. Decide conservatively

Accept only credible gains.
Reject regressions.
Treat paradox patterns as first-class evidence.

## 7. Record the result

Log what changed, why, what happened, and what should happen next.

Repeat.

---

# Acceptance Criteria

Accept a change only if at least one of the following is true:

1. `ToMCoordScore` improves by a meaningful margin.
2. `ToMCoordScore` is preserved while an important failure mode is clearly reduced, such as:
   - deadlock
   - collision
   - premature commitment
   - pathological waiting
3. the model becomes materially simpler or more robust with no meaningful score loss.

Reject a change if:

- `ToMCoordScore` drops
- collisions worsen materially
- deadlocks worsen materially
- ambiguity handling gets worse
- the result depends on a fragile average that hides obvious scenario regressions
- complexity increases without enough payoff

Use conservative epsilon logic.
Respect any deadlock auto-reject threshold in the active profile.

---

# Logging Requirements

After every run, record:

- run ID
- benchmark variant
- hypothesis
- exact code change
- primary metric
- secondary metrics
- scenario-family deltas
- keep/discard decision
- short interpretation
- next step

Always record whether the run improved or harmed:

- strategy switching
- ambiguity handling
- socially contingent behaviour

Preserve enough structure that `RESEARCHER.md` can build a timeline and `STORYTELLER.md` can later derive compelling incidents from the evidence without being present during execution.

Minimum narrative-ready fields per interesting run:

- scenario family
- partner style
- context tag
- belief turning point
- chosen action
- outcome
- one-line interpretation

---

# Prioritization Heuristics

Prioritize in this order:

1. small changes with a clear hypothesis
2. improvements to belief-guided strategy switching
3. better ambiguity handling without new safety regressions
4. context-sensitive action selection
5. simplifications that preserve performance

Deprioritize:

- cosmetic gains
- narrow wins on one scenario family
- improvements that depend on overcaution
- improvements that raise F1 but worsen action quality

---

# Paradox Policy

Treat these as first-class findings, not nuisances:

- `mind_reading_paradox`: belief metrics improve while coordination worsens
- `safety_paradox`: collisions fall while deadlocks rise
- `politeness_trap`: excessive yielding prevents completion
- `premature_assertion`: early confidence causes avoidable conflict
- `late_switch_failure`: correct belief arrives but policy fails to act on it

A paradox is often a better research clue than a clean average win.

---

# Morning Deliverable

At the end of the overnight run, provide:

## 1. Best result

- best accepted configuration
- baseline score
- best score
- improvement over baseline

## 2. Strongest successful ideas

State what likely worked and why.

## 3. Main failed ideas

State what failed and why.

## 4. Behavioural interpretation

Answer:

- does explicit ToM help in this benchmark?
- does it improve policy switching or only belief quality?
- in which contexts does it help most?
- where does it still break?

## 5. Recommended next human step

Recommend the next narrow experiment.

---

# Final Instruction

Act like a disciplined overnight researcher.

- Only edit `train.py`.
- Prefer one small change per experiment.
- Do not change evaluation logic.
- Optimize the main metric, not appearances.
- Keep the loop tight.
- Preserve evidence quality.
- Prefer belief-guided behavioural improvement over abstract cleverness.
