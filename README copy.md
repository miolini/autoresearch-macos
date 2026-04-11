# autoresearch-macos-tomx

Portable research orchestration for LLM-guided benchmark iteration.

## 1. What this repo is

`autoresearch-macos-tomx` is a terminal-first toolkit for running disciplined, reusable research workflows around narrow benchmark repos.

It is designed for cases where the scientific core already lives in a separate project, but the surrounding workflow has started to sprawl across prompts, agent configs, shell habits, run policies, promotion rules, and ad hoc notes. The aim is to pull that orchestration layer into a portable, inspectable codebase.

This repo is therefore **not** the benchmark itself. It is the layer that helps structure how an LLM or operator works with that benchmark:
- task contracts
- agent roles
- run policies
- promotion gates
- artifact expectations
- backend adapters
- report generation
- provenance-aware workflow rules

The intended interface is simple:
- use it from VS Code Terminal or a normal shell
- keep workflows explicit
- keep outputs reviewable
- keep benchmark semantics under the control of the benchmark repo, not the orchestration layer

---

## 2. Why it exists

In active research, useful method often gets trapped in scattered places:
- one task brief in markdown
- repo-specific instructions in chat history
- local agent definitions in hidden folders
- seed policies in notes
- promotion rules in memory
- report formatting done differently every time
- runner logic split between local scripts and cloud glue

That works for short bursts of experimentation, but it scales poorly if the goal is:
- reproducibility
- portability across repositories
- compatibility with different LLM backends
- long unattended runs
- cleaner separation between scientific code and orchestration code

This repo exists to extract the reusable part of that workflow into a dedicated layer.

The main idea is:

> keep benchmark science narrow and stable, while making the research process around it more structured, portable, and inspectable.

---

## 3. Relationship to `ToM_AI_Research_Team`

This toolkit is being shaped first around the `ToM_AI_Research_Team` workflow, but it is intended to be reusable beyond that single repo.

For that benchmark repo:
- the science-critical core remains in files such as `train.py`, `env.py`, `eval.py`, and selection logic
- benchmark semantics remain defined in the benchmark repo
- fixed scenarios, metrics, and thresholds should not be silently changed by the orchestration layer
- logs, archives, incumbent snapshots, and report packs remain provenance-sensitive artifacts owned by the benchmark repo

`autoresearch-macos-tomx` should therefore be understood as a **companion repo**, not a replacement.

### Benchmark repo responsibilities
- environment semantics
- evaluation semantics
- training code
- canonical artifacts
- experiment evidence
- scientific source of truth

### Orchestration repo responsibilities
- task framing
- agent definitions
- run discipline
- promotion rules
- template generation
- backend integration
- report and summary scaffolding
- reusable workflow logic

The boundary matters. This repo should help an operator or LLM work on a benchmark repo more effectively, without becoming the hidden source of scientific decisions.

---

## 4. Core components

The toolkit is expected to revolve around a small set of components.

### Task policies
Markdown or YAML contracts describing the exact shape of work to be done.

Examples:
- what files are in scope
- what kinds of changes are allowed
- what counts as a bounded patch
- what validation is required
- what counts as a comparability break

### Agent definitions
Structured role definitions for different kinds of work.

Examples:
- patcher
- reviewer
- evaluator
- promoter
- provenance checker

These roles should be portable across LLM backends where possible.

### Research policies
Explicit documents encoding the workflow rules that are otherwise easy to lose.

Examples:
- quick-gate seed sets
- promotion-gate seed sets
- keep/discard/promote criteria
- failure-family interpretation rules
- provenance guardrails

### Experiment templates
Reusable structures for naming, running, and summarizing experiments.

Examples:
- run naming conventions
- output directory contracts
- per-seed result tables
- aggregate summary templates
- decision logs

### Backend adapters
Thin integration layers for different execution contexts.

Examples:
- Codex-style CLI workflows
- plain local shell execution
- Modal or other batch/remote runners
- future LLM backends

### Reporting utilities
Helpers for turning raw run artifacts into readable summaries.

Examples:
- markdown summaries
- result tables
- comparison views
- promotion notes
- case-study outputs

---

## 5. Typical workflow

A normal workflow in this repo should look like this:

1. **Load a repo-specific overlay**
   - identify the target benchmark repo
   - read its sensitive zones, safe zones, validation rules, and canonical commands

2. **Load the task policy**
   - interpret the current objective
   - confirm whether the task is planning, patching, reviewing, or promoting

3. **Choose an agent role**
   - patcher for a bounded code change
   - reviewer for critical inspection
   - evaluator for run comparison
   - promoter for incumbent recommendations

4. **Generate or apply a bounded plan**
   - one narrow hypothesis
   - one constrained change surface
   - explicit validation steps
   - explicit output roots

5. **Run validation**
   - smoke run if required
   - seed-gated runs if required
   - collect metrics and artifacts

6. **Summarize results**
   - per-seed outcomes
   - aggregate metrics
   - keep/discard/promote recommendation
   - clear separation between confirmed evidence and inference

7. **Preserve provenance**
   - do not overwrite canonical evidence casually
   - snapshot important results
   - record what was changed, compared, and promoted

The goal is not maximum automation. The goal is disciplined automation that still leaves a legible audit trail.

---

## 6. Example case study

The initial motivating case is the `ToM_AI_Research_Team` benchmark workflow.

A representative example is a bounded `train.py` patch cycle:
- identify a narrow failure family
- propose one `train.py`-only change
- run a smoke check
- run multi-seed evaluation
- compare against baseline
- decide keep/discard/promote
- snapshot the resulting incumbent candidate

A concrete case study in this style can include:
- the motivating failure pattern
- the exact patch family
- per-seed outputs
- aggregate metric changes
- interpretation of gains and tradeoffs
- promotion recommendation

This kind of case study is important because it demonstrates the method, not just the result. It shows how the orchestration layer supports:
- bounded intervention
- multi-seed discipline
- explicit interpretation
- provenance-aware promotion

---

## 7. Planned repo layout

A likely initial layout is:

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
│   └── benchmark_policy.md
├── agents/
│   ├── patcher.yaml
│   ├── reviewer.yaml
│   ├── evaluator.yaml
│   ├── promoter.yaml
│   └── provenance_checker.yaml
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

This layout is meant to keep reusable workflow logic separate from repo-specific overlays.

⸻

8. Design principles

Terminal-first

The default workflow should work cleanly in VS Code Terminal or a plain shell. A web UI is not required.

Local-first

Local execution should be the default path. Remote or cloud execution should be treated as an extension, not the primary workflow.

Explicit boundaries

The orchestration layer should never silently redefine benchmark semantics, metric definitions, or scientific thresholds that belong to the benchmark repo.

Bounded changes

The system should encourage one narrow hypothesis and one bounded patch at a time.

Provenance awareness

Logs, archives, incumbent snapshots, and report artifacts should be treated as evidence, not just disposable output.

Portability

The most reusable value should live in markdown, YAML, and small Python adapters rather than in a single backend-specific format.

Inspectability

Every important workflow step should be understandable from files and commands, not hidden in opaque automation.

Confirmed facts vs inference

Outputs should distinguish:
	•	what the artifacts actually show
	•	what is being inferred from them
	•	what remains uncertain

Research discipline over convenience

The toolkit should make it easier to be careful, not easier to be sloppy faster.

⸻

9. Near-term roadmap

Phase 1: architecture and policy capture

Create the initial documentation and policy skeleton:
	•	README.md
	•	docs/architecture.md
	•	docs/concepts.md
	•	task_policies/omx_task.md
	•	policies/promotion_rules.yaml
	•	examples/tom_ai_research_team/repo_overlay.yaml

Phase 2: encode current working method

Capture the current benchmark workflow in explicit files:
	•	canonical repo root
	•	safe vs sensitive zones
	•	smoke validation commands
	•	quick-gate seed set
	•	promotion-gate seed set
	•	artifact checks
	•	promotion criteria
	•	provenance rules

Phase 3: add minimal execution helpers

Add small terminal-friendly helpers for:
	•	running a standard local command sequence
	•	comparing per-seed outputs
	•	rendering a markdown result summary

Phase 4: backend adapters

Add optional adapters for:
	•	Codex-style workflows
	•	plain local CLI use
	•	Modal or equivalent long-run infrastructure

Phase 5: extend beyond first benchmark repo

Once the first overlay is stable:
	•	test reuse on a second benchmark repo
	•	identify what is truly portable
	•	reduce project-specific assumptions
	•	keep the interface small

⸻

What this repo is not

This repo is not:
	•	the benchmark itself
	•	the canonical source of model results
	•	a hidden place to change evaluation semantics
	•	a replacement for careful scientific review
	•	a generic “autonomous scientist” platform

It is a practical orchestration layer for structured human-plus-LLM research workflows.

⸻

Status

This is an emerging toolkit, not a finished product.

The immediate goal is to turn a working research method into a portable, inspectable codebase without losing the discipline that made the original workflow useful.

A next sensible step is a matching `docs/architecture.md` that defines the boundary between reusable orchestration logic and repo-specific overlays.