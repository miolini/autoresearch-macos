# Current Working Method

## Scope

This document summarizes the current working benchmark method encoded in the repo overlay and policy YAMLs. The benchmark repo remains the source of truth for benchmark semantics, metric definitions, and scientific results. The active mode is train.py-only local quality work against frozen Variant 1. Reusable gate policy is the authoritative measurement workflow; branch-specific study or incumbent-note conventions are optional supporting evidence.

## Canonical Repo Root

- Confirmed benchmark repo root: `/Users/stephenbeale/Projects/ToM_AI_Research_Team`
- Do not use the similarly named path with spaces as the repo root.

## Safe Editable Zones

- `train.py` is the current bounded hypothesis-edit surface.

## Sensitive or Read-Only Zones

- `env.py`, `eval.py`, and `scripts/local_runner.py` are frozen benchmark-harness surfaces in the current mode.
- `scripts/select_candidate.py` participates in scientific decision policy and comparability.
- `logs/`, `modal/`, and incumbent/archive paths are evidence stores and should not be casually overwritten.

## Smoke Validation Commands

Run from `/Users/stephenbeale/Projects/ToM_AI_Research_Team`:

```bash
python scripts/local_runner.py --train-episodes 5 --seed 7 --output-root logs/local-smoke-validation
```

Expected packaged artifacts:

- `baseline_metrics/metrics.json`
- `candidate_metrics/metrics.json`
- `selection/selection.json`
- `selected_model/model.pt`

Smoke is a 1-seed operational check only. It is not sufficient for keep or promote decisions.

## Quick-Gate Seed Set

- Confirmed seeds: `7`, `11`, `17`
- Scientific gate standard: `800` episodes per seed
- Current rule: mean `DeadlockRate` not worse, mean `ToMCoordScore` higher, and no seed with `DeadlockRate` worse than baseline by more than `+0.10`

## Promotion-Gate Seed Set

- Confirmed seeds: `7`, `11`, `17`, `23`, `29`
- Scientific gate standard: `800` episodes per seed
- Current rule: mean `DeadlockRate` not worse, mean `ToMCoordScore` higher, mean `CollisionRate` lower or equal, mean `SuccessRate` higher or equal, and no catastrophic single-seed regression

## Artifact Checks

- Smoke bundle must keep baseline/candidate stdout logs, baseline/candidate `metrics.json`, `selection/selection.json`, and `selected_model/model.pt`.
- Per-seed quick-gate and promotion-gate runs must preserve per-seed logs and selection summaries.
- Reusable gate-policy summaries are the authoritative aggregate evidence.
- Canonical quick-gate aggregate summary path: `/Users/stephenbeale/Projects/ToM_AI_Research_Team/logs/<experiment_label>_quick_gate_summary.md`
- Canonical promotion-gate aggregate summary path: `/Users/stephenbeale/Projects/ToM_AI_Research_Team/logs/<experiment_label>_promotion_gate_summary.md`
- `RESULTS_TABLE.md`, `INCUMBENT*_NOTE.md`, and study-style markdown may still be used as optional branch-specific supporting evidence.
- Promotion-relevant runs should use fresh output roots rather than overwriting earlier evidence.

## Promotion Criteria

- Smoke must pass before wider seed-gated evaluation.
- Current method is a 1/3/5 ladder: 1-seed smoke, 3-seed quick gate, 5-seed promotion gate.
- Required artifacts must exist for the current stage.
- Current benchmark mode assumes train.py-only bounded changes.
- Quick gate advances only after all required seeds complete and the quick-gate rule passes.
- Allowed decision outputs are `reject`, `hold`, `promote_candidate`, and `needs_human_review`.

## Provenance Rules

- Every run record needs `run_id`, `timestamp_utc`, `operator`, `repo_root`, `patch_identifier`, `hypothesis_identifier`, `stage`, `command`, and `working_directory`.
- Per-seed outputs must record the seed used.
- Preserve failed run evidence.
- Use fresh output roots for promotion-relevant runs.
- When tracking a weakness or hypothesis, keep both a short label and a more detailed mechanism description.
- Separate artifact-observed facts from interpretation and mark uncertainty explicitly.

## Confirmed Facts

- The benchmark root, smoke command, quick-gate seeds, and promotion-gate seeds are already known.
- `scripts/local_runner.py` currently packages `baseline_metrics/`, `candidate_metrics/`, `selected_model/`, and `selection/`.
- `env.py`, `eval.py`, and `scripts/local_runner.py` are frozen surfaces during current train.py-only quality passes.
- Promotion evidence currently uses `RESULTS_TABLE.md` and `INCUMBENT*_NOTE.md` in incumbent snapshot directories.

## Working Assumptions

- Running seeds in listed order is current convention rather than benchmark semantics.
- `archive/` should be treated as protected whenever it exists.
- Study markdown and incumbent snapshot notes remain useful, but they do not define the canonical gate contract.

## Open Questions

- Should promotion-gate decisions require a clean benchmark repo commit hash?
