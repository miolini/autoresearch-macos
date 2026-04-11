# Check2 / Check3 Replicability Report

This note uses only the current repo-resident evidence from the `check2` and
`check3` snapshots plus the existing provenance notes. It is meant to answer a
practical question: do these two reruns demonstrate a reproducible
configuration?

## Short Answer

Yes at the configuration level, no at the exact-number level.

`check2` and `check3` are strong evidence that the current duplicate-run setup is
now reproducible in the sense that:

- the same exact warm-start archive is being used
- the same continuation protocol is being used
- the same qualitative behavior pattern reappears across reruns
- the same relative strengths and weaknesses reappear across reruns

They are not evidence of strict deterministic replay, because the scores differ
meaningfully across the two reruns, especially on `seed11`.

## What Repeated Correctly

The most important provenance problem from the earlier branch has been fixed.

Both reruns are continuations from the exact local V2 `800` archive under:

- [`auxhead-lite-v2-local800`](/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite-v2-local800)

The dedicated duplicate runner mounts those checkpoints directly:

- [`modal_v2b_runner.py`](/Users/stephenbeale/Projects/ToM_AI_Research_Team/scripts/modal_v2b_runner.py)

And the completed run summaries record the corresponding exact warm-start source:

- [`check2 seed7`](/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/check2-140k-20260410/seed7/run_summary.json)
- [`check2 seed11`](/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/check2-140k-20260410/seed11/run_summary.json)
- [`check3 seed7`](/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/check3-140k-20260410/seed7/target-140000/run_summary.json)
- [`check3 seed11`](/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/check3-140k-20260410/seed11/target-140000/run_summary.json)

So the main earlier provenance issue, “was this branch really started from the
intended V2 `800` checkpoints?”, is now answered much more cleanly than before.

## Why This Counts As Reproducible Configuration

### 1. Concept

Both reruns still look like the same conceptual policy family:

- belief-guided coordination under ambiguity
- improved timing / switching relative to baseline
- the same unresolved failure pocket (`assert_or_yield`)
- the same strong family (`late_disambiguation`)

That means the training setup is not drifting into a different behavioral regime
from one rerun to the next.

### 2. Design

The duplicate protocol is stable across the two reruns:

- same seeds: `7` and `11`
- same target total episodes: `140000`
- same fixed benchmark / evaluation surface
- same archive lineage: exact V2 local `800`

This is why `check2` and `check3` are useful as a configuration replication,
even though they are not a new-seed generalization study.

### 3. Engineering

The current setup is materially better engineered for provenance than the older
branch:

- explicit mounted warm-start checkpoints
- per-seed run summaries
- per-seed choice analyses
- progress files and stdout logs
- named branch snapshots in the incumbent archive

That means the configuration can now be inspected after the fact without having
to reconstruct the lineage from memory.

### 4. Equations / Weighting

The runner passes the same inherited model and training settings from the saved
checkpoint arguments into the continuation run, including:

- hidden size
- learning rate
- gamma
- entropy coefficient
- auxiliary loss weight
- ToM experiment strength
- belief / context thresholds

This matters because the relevant “equation-level” repeatability here is not
just the top-line score formula, but the fact that the same loss-weighting and
decision-threshold regime is being continued rather than silently changed.

### 5. Training

The training story is also consistent:

- both runs continue the same `800 -> 140k` path
- both complete successfully for both seeds
- both produce distinct seed-specific outputs
- both beat baseline comfortably

That is enough to say the training configuration is reproducible as a working
pipeline.

## Why This Is Not Exact Reproducibility

The reruns are not numerically identical:

- `check2` branch mean `ToMCoordScore`: `0.4425`
- `check3` branch mean `ToMCoordScore`: `0.3712`

The biggest divergence is `seed11`:

- `check2 seed11` `ToMCoordScore`: `0.4196`
- `check3 seed11` `ToMCoordScore`: `0.3465`
- `DeadlockRate`: `0.10 -> 0.35`

So the right conclusion is:

- reproducible configuration: yes
- deterministic replay: no

This is still compatible with the duplicate-study goal, because the aim was to
confirm the exact warm-start configuration and see whether the same broad
behavioral regime returns, not to prove bitwise-repeatable optimization.

## What A Critical AI-Literate Audience Should Take Away

The decisive points are:

1. `check2` and `check3` use the intended exact V2 `800` warm-start archive, so
   the main provenance error in the earlier branch has been corrected.
2. Both reruns remain far above baseline, preserve the same qualitative family
   structure, and preserve the same broad seed ordering (`seed7 > seed11`).
3. The main unresolved problem remains `assert_or_yield`, which appears in both
   reruns rather than vanishing in one and reappearing in the other.
4. The variance between `check2` and `check3`, especially on `seed11`, means the
   line is promising but still variance-sensitive.
5. Therefore the configuration is replicable enough to trust lineage-level
   claims, but not stable enough yet to collapse all uncertainty into a single
   branch mean.

## Does This Satisfy The Provenance Issue?

Mostly yes, with one important limit.

It satisfies the provenance issue if the issue is:

- “Are we now actually running the intended exact V2 local `800` continuation
  setup?”

It does **not** fully satisfy the issue if the issue is:

- “Can we now treat one successful rerun as definitive evidence that the line is
  fully stable?”

The current evidence supports:

- corrected lineage
- corrected configuration
- repeated qualitative regime
- repeated baseline-beating behavior

The current evidence does not yet support:

- strict deterministic replay
- low-variance branch behavior
- broad new-cohort / new-seed generalization claims

## Between-Group / Participant Framing

If `check2` and `check3` are described as a “between-group participants study,”
that framing is only partly accurate.

What this pair of runs really is:

- two independent reruns of the same protocol
- on the same seed pair
- under the same intended configuration

So they are good for:

- configuration replication
- lineage verification
- behavior-pattern replication

They are not, by themselves, a full between-group generalization study, because
they do not introduce a new participant cohort or new seed cohort.
