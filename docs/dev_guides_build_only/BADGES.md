## Badges (stub)

<!-- Replace these placeholders with real links once CI and release metadata exist. -->

[![CI](https://img.shields.io/badge/CI-pending-lightgrey)](#)
[![License](https://img.shields.io/badge/license-TBD-lightgrey)](#)
[![Status](https://img.shields.io/badge/status-emerging-blue)](#)

## Quick Start

1. Clone the repository.
2. Open it in VS Code or use a plain terminal.
3. Read the policy and architecture docs first.
4. Set the target benchmark overlay and task policy.
5. Run a bounded workflow (plan -> patch -> validate -> summarize).
6. Preserve artifacts and record provenance decisions.

```bash
git clone <repo-url>
cd autoresearch-macos-tomx
# replace with your project commands once scripts are finalized
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

Status

This is an emerging toolkit, not a finished product.

The immediate goal is to turn a working research method into a portable, inspectable codebase without losing the discipline that made the original workflow useful.

## Contributing (stub)

Contributions are welcome, especially around:

- policy clarity
- reproducible workflow helpers
- backend adapter reliability
- reporting and artifact summarization

Before opening a PR:

1. Keep changes bounded to one clear objective.
2. Preserve benchmark-orchestration boundaries.
3. Include validation notes and artifact impact.
4. Document confirmed evidence separately from inference.

Suggested future files:

- `CONTRIBUTING.md` for contributor workflow and standards
- `CODE_OF_CONDUCT.md` for collaboration expectations