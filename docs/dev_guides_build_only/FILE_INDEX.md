research-orchestrator-kit/
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