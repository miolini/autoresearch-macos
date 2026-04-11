# Architecture

This document describes the current architecture of `ToM_AI_Research_Team` with a bias toward reproducibility and experiment validity.
It separates confirmed implementation facts from inferred structure so future edits do not accidentally turn assumptions into “truth.”

## Confirmed Core Architecture

### 1. Benchmark layer

`env.py` defines the benchmark surface.

Confirmed facts:

- Action space has 5 discrete actions.
- Partner space has 5 named partner types.
- The environment observation vector length is 10.
- `fixed_validation_scenarios()` defines the fixed scenario suite used for evaluation.
- `ToMCoordinationEnv` owns:
  - reset behavior
  - partner policy
  - observation construction
  - reward shaping
  - collision/deadlock/timeout termination

Implication:

- `env.py` is not just “environment code”; it is part of the benchmark specification.

### 2. Model layer

`train.py` defines two model families:

- `BaselinePolicy`
  A GRU-based recurrent policy with a single policy head.
- `ToMPolicy`
  A GRU-based recurrent policy with:
  - belief head over partner types
  - partner-action prediction head
  - policy head conditioned on hidden state plus belief distribution
  - hand-authored decision priors
  - optional bolt-on experiments

Confirmed experiment switches:

- `none`
- `belief_uncertainty_wait`
- `contextual_right_of_way_switch`

### 3. Training layer

`train.py::train()` is the main training loop.

Confirmed facts:

- seeds Python, NumPy, and Torch
- uses on-policy rollouts from `rollout_episode()`
- optimizes policy loss plus entropy regularization
- adds auxiliary losses:
  - belief classification loss
  - partner-action prediction loss for ToM variant
  - behavior-shaping loss from `heuristic_teacher_action()`
- can warm-start from checkpoints
- stores checkpoint metadata in the saved payload

### 4. Evaluation layer

Two evaluation products are emitted from training:

- `eval.py::evaluate_policy()`
  Produces aggregate metrics in `EvalMetrics`
- `train.py::analyze_choice_context_outcomes()`
  Produces rich analysis such as:
  - context-action counts/rates
  - context terminal outcomes
  - partner-style rates
  - experiment-mask firing counts/rates
  - scenario summaries
  - context-sensitive regret estimates

Confirmed top-line metrics include:

- `SuccessRate`
- `CoordinationEfficiency`
- `IntentionPredictionF1`
- `StrategySwitchAccuracy`
- `AmbiguityEfficiency`
- `AverageDelay`
- `CollisionRate`
- `DeadlockRate`
- `ToMCoordScore`

### 5. Selection layer

`scripts/select_candidate.py` is the promotion gate for local packaged runs.

Confirmed logic:

- compare baseline and candidate `ToMCoordScore`
- require improvement by at least `epsilon`
- auto-reject if candidate deadlock worsens beyond `deadlock_delta_threshold`
- copy the selected `model.pt` into `selected_model/`

This script is therefore part of the scientific decision policy, not just file movement.

### 6. Orchestration layer

There are three confirmed orchestration lanes:

- Local lane
  `scripts/local_runner.py`
- Azure lane
  `azure/pipeline.yml` + `scripts/aml_train_component.py`
- Modal lane
  `scripts/modal_v2b_runner.py`, `scripts/modal_auxhead_lite_runner.py`, shell launchers, and report scripts

### 7. Serving layer

`webapp/api/main.py` provides a small inference API.

Confirmed behavior:

- loads model path from `MODEL_PATH` or defaults to `logs/local-run/selected_model/model.pt`
- reconstructs the correct model variant from saved checkpoint args
- serves `/health` and `/predict`

`webapp/api/modal_app.py` wraps the same API for Modal ASGI serving.

## Main Execution Paths

### Local-first execution path

```text
Researcher
  -> scripts/local_runner.py
    -> train.py --variant baseline
      -> env.py
      -> eval.py
      -> checkpoint + curve + choice_analysis artifacts
    -> train.py --variant tom
      -> env.py
      -> eval.py
      -> checkpoint + curve + choice_analysis artifacts
    -> scripts/select_candidate.py
      -> selected_model/model.pt
      -> selection/selection.json
```

This is the clearest confirmed “primary path” in the current repo.

### Azure execution path

```text
Azure pipeline
  -> azure/components/train.yml
    -> scripts/aml_train_component.py
      -> train.py
  -> azure/components/select.yml
    -> scripts/select_candidate.py
  -> optional azure/register_model.yml
    -> scripts/register_model.py
```

### Modal continuation path

```text
Incumbent archive in modal/tom-experiment-incumbent/
  -> scripts/modal_v2b_runner.py or scripts/modal_auxhead_lite_runner.py
    -> remote train.py continuation
    -> progress/run_summary artifacts in Modal volume
  -> scripts/modal_*_report.py
    -> markdown/json/svg/pdf reports
```

### Serving path

```text
selected_model/model.pt
  -> webapp/api/main.py
    -> FastAPI /health
    -> FastAPI /predict
  -> optional webapp/api/modal_app.py
```

## Artifact Contract

The repo’s local packaged artifact contract is explicit in `README.md` and implemented in `scripts/local_runner.py`.

Confirmed packaged layout:

- `<output-root>/baseline_model/model.pt`
- `<output-root>/baseline_metrics/metrics.json`
- `<output-root>/baseline_metrics/learning_curve.csv`
- `<output-root>/baseline_metrics/choice_analysis.json`
- `<output-root>/candidate_model/model.pt`
- `<output-root>/candidate_metrics/metrics.json`
- `<output-root>/candidate_metrics/learning_curve.csv`
- `<output-root>/candidate_metrics/choice_analysis.json`
- `<output-root>/selected_model/model.pt`
- `<output-root>/selection/selection.json`

Why this matters:

- Azure wrappers depend on it.
- Inference depends on the saved checkpoint payload shape.
- Modal provenance and reporting depend on comparable artifacts.

## Confirmed Reproducibility Controls

- Fixed evaluation scenarios are centralized in `env.py`.
- `train.py` seeds Python, NumPy, and Torch.
- `configs/overnight_profile.json` stores default parity and controller settings.
- `scripts/azure_child_job_controller.py` records per-run metadata and uses shared thresholds.
- Checkpoints embed argument metadata, including progress-related fields.
- `docs/RESEARCH_FAMILY_TRIAGE_POLICY.md` defines an evidence ladder and family promotion rules.

## Inferred Structure

The statements in this section are informed by the repo layout and docs, but they are still interpretation.

- The project appears to have evolved from a smaller Variant 1 local benchmark into a broader local-first plus Modal long-run workflow, while retaining Azure compatibility.
- The repo treats committed run artifacts as part of the research record, not as disposable outputs.
- `.github` prompt and agent files appear to support semi-automated overnight research sessions around the benchmark.
- `modal/tom-experiment-incumbent/` functions as both a warm-start store and a provenance archive for promoted or candidate lineages.

## Probable Architecture Diagram In Words

At the center of the codebase is a compact benchmark loop:
`env.py` defines a partial-observability bottleneck negotiation task, `train.py` trains recurrent agents against that task, and `eval.py` measures them on a fixed suite of scenarios.

Around that center sits a packaging and decision layer:
`scripts/local_runner.py` turns two training runs into a comparable artifact pack, and `scripts/select_candidate.py` applies the promotion rule that chooses baseline or candidate.

Around that sits an experimentation shell:
`logs/` stores local evidence, `modal/` stores long-run continuations and report packs, `notebooks/` renders visual analysis, `docs/` stores policy and process notes, and `.github/` stores AI workflow prompts and role definitions.

At the edge sits deployment and serving:
Azure YAMLs wrap the same training/selection contract for cloud jobs, while `webapp/api/main.py` serves a selected checkpoint as a lightweight inference API.

## Fragile Or Unclear Parts

### Source-of-truth drift risk

- There are multiple archived `train.py` copies under `modal/tom-experiment-incumbent/`.
- At least two archived copies differ from the root `train.py`.
- One archived copy currently matches the root `train.py`.

Risk:

- Future contributors may edit one copy and assume they updated the system, or compare results produced by non-identical training code without realizing it.

### Benchmark-is-code risk

- The benchmark data lives inside `env.py`, not in a separately versioned dataset package.

Risk:

- Small environment edits can silently invalidate historical comparisons.

### Validation gap

- No automated test suite exists.

Risk:

- Reproducibility depends on manual smoke runs and artifact inspection, which is weaker than code-level regression protection.

### Provenance mixed with workspace state

- The repo contains active code, committed evidence, runtime caches, and research notes together.

Risk:

- It is easy to touch non-source material by accident or to mistake historical artifacts for active outputs.

### Canonical-path ambiguity

- `docs/CANONICAL_WORKFLOW.md` warns about path confusion.
- `modal/tom-experiment-incumbent/CURRENT_INCUMBENT.txt` points outside the canonical repo root.

Risk:

- Tooling or humans may read the wrong directory as the current scientific source of truth.

### Support-status ambiguity

- Azure assets are present.
- The `README.md` says Azure is no longer the primary path.

Risk:

- Maintainers may not know whether Azure should be kept fully production-ready or only minimally compatible.

## Safe-Change Zones

- `docs/`
  Clarify workflows, provenance rules, and validation expectations here first.
- report-generation code under `scripts/modal_*_report.py`
  Safer than modifying benchmark semantics.
- `notebooks/variant2_visuals.py`
  Visualization only.
- `.github/` prompt and role assets
  Safe if they accurately reflect the real workflow.

## Sensitive Zones

- `env.py`
  Benchmark revision zone.
- `eval.py`
  Metric revision zone.
- `train.py`
  Model/training/analysis revision zone.
- `scripts/select_candidate.py`
  Promotion-policy revision zone.
- `scripts/local_runner.py` and `scripts/aml_train_component.py`
  Artifact-contract revision zone.
- `modal/tom-experiment-incumbent/`, `logs/`, `modal/`, `archive/`
  Provenance and evidence zones.

## Validation Steps For Future Edits

### Minimum validation for docs-only edits

1. Confirm all referenced files still exist.
2. Confirm the text does not claim unsupported paths as canonical.

### Minimum validation for science-critical edits

Run:

```bash
python scripts/local_runner.py --train-episodes 5 --seed 7 --output-root logs/local-smoke-validation
```

Then verify:

1. both baseline and candidate runs complete
2. `metrics.json` exists for both
3. `selection.json` exists and contains a decision
4. `selected_model/model.pt` exists
5. `train.py` still emits checkpoint, curve, and choice-analysis lines expected by wrappers

### Extra validation if you changed benchmark or metric semantics

1. Document the change as a benchmark revision, not a routine refactor.
2. Explain how historical results should be interpreted afterward.
3. Re-check any docs or report logic that assumes the old metric or scenario definitions.

### Extra validation if you changed serving

1. Generate a fresh selected model.
2. Start the API and hit `/health`.
3. Send one valid `/predict` request with a 10-element observation.

### Extra validation if you changed Modal or Azure workflows

1. Prove the local-first path still works first.
2. Validate the wrapper against the current artifact contract.
3. Treat `progress.json` and `run_status.json` as operational status only, not final scientific evidence.
