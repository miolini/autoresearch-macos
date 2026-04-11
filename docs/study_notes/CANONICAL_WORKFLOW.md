# Canonical Workflow

## Canonical Root

Use this path as the only real project root:

- `/Users/stephenbeale/Projects/ToM_AI_Research_Team`

Do not treat the similarly named path with spaces as the repo root. That path currently behaves like stray runtime state, not the source tree.

## Canonical Layout

- core experiment code:
  - `train.py`
  - `env.py`
  - `eval.py`
- execution scripts:
  - `scripts/`
- local experiment artifacts:
  - `logs/`
- long-run Modal artifacts:
  - `modal/`
- notebooks and notebook helper code:
  - `notebooks/`
- temporary scratch outputs:
  - `tmp/`
  - `tmp_smoke/`
- runtime/session state:
  - `.omx/`

## Canonical Meaning Of Key Folders

- `logs/`
  - local verification runs
  - local candidate comparisons
  - temporary scientific checks
- `modal/tom-130k-modal-results`
  - older long-run branch at 130k
- `modal/tom-140k-modal-results`
  - older 140k branch
- `modal/tom-140k-modal-results-v2`
  - current Variant 2 long-run branch
- `modal/tom-experiment-incumbent`
  - warm-start checkpoints and incumbent archive
- `modal/reports`
  - notebook-generated visual exports that are not tied to one single long-run folder
- `notebooks/`
  - all `.ipynb`
  - notebook helper modules such as `variant2_visuals.py`

## Related policy notes

- Research family triage policy: `docs/RESEARCH_FAMILY_TRIAGE_POLICY.md`
- Auxiliary-loss next-action review: `docs/AUXNEXT_ACTION_PROMOTION_REVIEW.md`

## Current Frontier

- active frontier: Variant 2 contextual right-of-way
- strongest long-run branch so far:
  - `modal/tom-140k-modal-results-v2`
- best single checkpoint so far:
  - `seed11 new140`
- main unresolved family:
  - `assert_or_yield`

## Execution Ladder

When choosing the next action, follow this order:

1. Establish the current canonical result.
2. If the result is surprising or operationally messy, run an exact duplicate first.
3. Compare duplicate vs canonical.
4. Only after that, branch to fresh seeds or new training changes.
5. Only after the branch decision is clear, generate polished reports and notebook cleanup.

Do not mix duplicate, branch, report polish, and notebook cleanup in one step unless explicitly requested.

## Modal Rules

- use detached Modal launches for long runs
- launch one detached run per seed rather than relying on one sequential parent entrypoint
- give each new branch or duplicate its own volume name
- prefer exact reruns over new branching when reproducibility is the main uncertainty
- do not treat `progress.json` or `run_status.json` as final scientific evidence
- final evidence comes from:
  - `run_summary.json`
  - final curve
  - final choice analysis

## Current Duplicate Policy

For the current project state, the next clean scientific move after a notable result is:

- `v2b`: exact duplicate of the current Variant 2 800 -> 140k run on seeds 7 and 11

Only after `v2b` is understood should fresh-seed expansion happen.

## Notebook Policy

- notebooks read report packs and artifacts
- notebooks do not define the canonical scientific state
- canonical scientific state lives in report packs and source-controlled docs
