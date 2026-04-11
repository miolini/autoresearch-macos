# Build Plan

Treat this as a separate orchestration and tooling project, not additional accretion inside the benchmark repository.

The portable value is in task framing, agent definitions, policy documents, experiment templates, evaluation conventions, backend adapters, and runner glue.

## What the new codebase is for

A small research orchestration toolkit that sits beside the benchmark repository and can drive:

- repo-specific task contracts such as OMX_TASK.md
- agent definitions
- run policies and promotion rules
- local and Modal execution
- portable report generation

This matches the existing project discipline: keep benchmark science narrow and frozen, while moving workflow logic into a reusable layer.

## Recommended repo split

Use two repositories.

### 1. Benchmark repo

Keep:

- train.py
- env.py
- eval.py
- configs
- logs, reports, and incumbents
- benchmark docs

This remains the science repository.

### 2. Orchestration repo

Use a repository such as:

- autoresearch-macos-tomx
- or more generally research-orchestration-kit

This holds:

- prompt and task contracts
- agent role definitions
- run plans
- seed ladders
- promotion policy
- backend adapters
- command wrappers
- report templates
- case studies

This split aligns with MAKE AUTORESEARCH_MACOS_TOMX.md.

## Minimal file layout

```text
autoresearch-macos-tomx/
├── README.md
├── pyproject.toml
├── docs/
│   ├── architecture.md
│   ├── concepts.md
│   ├── backend-adapters.md
│   └── case-studies/
│       └── METHOD_CASE_STUDY.md
├── task_policies/
│   ├── omx_task.md
│   └── tom_benchmark_policy.md
├── agents/
│   ├── patcher.yaml
│   ├── reviewer.yaml
│   ├── evaluator.yaml
│   └── promoter.yaml
├── policies/
│   ├── promotion_rules.yaml
│   ├── seed_sets.yaml
│   ├── artifact_contract.yaml
│   └── provenance_rules.yaml
├── templates/
│   ├── experiment_header.md
│   ├── result_summary.md
│   └── run_matrix.csv
├── backends/
│   ├── codex_adapter.py
│   ├── cli_adapter.py
│   └── modal_adapter.py
├── runners/
│   ├── local_run.py
│   ├── gated_run.py
│   └── compare_runs.py
├── examples/
│   └── tom_ai_research_team/
│       ├── repo_overlay.yaml
│       ├── seed_policy.yaml
│       └── command_examples.md
└── scripts/
    ├── bootstrap_repo.sh
    └── render_report.py
```

## Reusable vs repo-specific

### Reusable

- agent roles
- run-state machine
- promotion gate logic
- summary and report templates
- backend interfaces
- artifact expectations
- provenance guardrails

### Repo-specific

- canonical repo root
- exact command templates
- sensitive files and do-not-touch zones
- benchmark semantics warnings
- benchmark-specific seed policy
- incumbent paths
- report field names tied to one benchmark

This distinction matters because the benchmark has strong constraints: env.py and eval.py are fixed, the editable surface is narrow, and semantic drift is actively avoided during quality passes.

## Where METHOD_CASE_STUDY belongs

METHOD_CASE_STUDY.md is an excellent example artifact for the orchestration repo, not just a one-off note.

It captures:

- a bounded hypothesis
- a narrow patch family
- per-seed outputs
- aggregate means
- promotion reasoning
- explicit next action

Recommended location:

- docs/case-studies/
- or examples/tom_ai_research_team/

It demonstrates method, not just result. It also fits the benchmark discipline of multi-seed evidence and stable benchmark semantics.

## What to do first, in order

### Phase 1: planning only

Create:

- README.md
- docs/architecture.md
- docs/concepts.md
- task_policies/omx_task.md
- policies/promotion_rules.yaml
- examples/tom_ai_research_team/repo_overlay.yaml

### Phase 2: encode current working method

Capture:

- canonical repo root
- safe and sensitive zones
- smoke command
- 3-seed quick gate
- 5-seed promotion gate
- artifact checks
- provenance rules

### Phase 3: add adapters

Only after the policy layer is stable:

- Codex adapter
- local CLI adapter
- optional Modal adapter

This sequencing follows the architecture-before-coding principle.

## What not to do

Do not:

- move benchmark semantics into the orchestration repo
- duplicate train.py logic there
- let the toolkit decide scientific thresholds implicitly
- mix archived evidence with editable templates
- make the orchestration repo the source of truth for benchmark results

The benchmark repository should remain the scientific source of truth. The orchestration repository should remain the reusable workflow layer around it.

## Naming recommendation

Best practical name:

- autoresearch-macos-tomx

Best descriptive subtitle:

- Portable research orchestration for LLM-guided benchmark iteration

## Short decision

- Worth doing: yes
- Separate repo: yes
- No web UI needed: yes, terminal-first is sufficient
- Use METHOD_CASE_STUDY as first example artifact: yes
- Start with planning docs, not code: yes

Next best deliverable: a concrete README plus repo tree for autoresearch-macos-tomx.
