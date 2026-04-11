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

```
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
```

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

#About # autoresearch-macos

## Local path note

Canonical local working-copy path for the `tomx` branch:

- `/Users/stephenbeale/Projects/autoresearch-macos-tomx`

Compatibility symlink retained for older references:

- `/Users/stephenbeale/Projects/autoresearch-macos` -> `/Users/stephenbeale/Projects/autoresearch-macos-tomx`

Use the `-tomx` path in new notes, scripts, and links when referring to the local checkout.

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## Open source project worth to look at

Open source collabaration platform for agentic swarms in organizations and communityies.

[SentientWave Automata](https://github.com/sentientwave/automata)

## How it works

The repo is deliberately kept small and only really has a three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

## Quick start

**Requirements:** Apple Silicon Mac (M1/M2/M3/M4 with Metal/MPS support) or a single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).


```
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

**Platforms support**. This fork officially supports **macOS (Apple Silicon / MPS)** and CPU environments, while preserving the original NVIDIA GPU support. It removes the hardcoded dependency on FlashAttention-3, falling back to PyTorch's native Scaled Dot Product Attention (SDPA) with manual sliding window causal masking when needed. It also features MPS-specific optimizations (disabling unsupported `torch.compile` paths, lowering memory batch sizes for Metal bounds, and precisely casting optimizer states) allowing you to run autonomous research agents directly on your Mac!

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```text
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

There is now also a repo-local Codex skill at `.codex/skills/autoresearch-lab/SKILL.md` that packages the experiment loop with helper scripts for setup checks, branch creation, bounded runs, log parsing, and `results.tsv` updates.

For a thin local launcher/dashboard, run:

```
./autoresearch-dashboard
```

That opens a local web UI with:

- a chat-style prompt composer and local transcript for the current autoresearch repo
- a second ToMX profile that reuses the external ToM workspace plus its repo-local agent TOMLs
- a separate `Send To Codex` path so prompt delivery is distinct from direct run buttons
- lightweight buttons for readiness checks and bounded run launching

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

If you're going to be using autoresearch on Apple Macbooks in particular, I'd recommend one of the forks below. On top of this, if you'd like half-decent results at such a small scale, I'd recommend this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) which is cleaner than what exists out there otherwise. It should be a drop in replacement because I have encoded it in exactly the same format. Any of your favorite coding agents should be able to do the swap :)

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx)

## Licenses

MIT
