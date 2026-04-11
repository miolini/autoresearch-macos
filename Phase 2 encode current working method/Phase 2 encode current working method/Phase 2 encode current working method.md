**Phase 2: encode current working method**  
  
Capture:  
	•	canonical repo root  
	•	safe/sensitive zones  
	•	smoke command  
	•	3-seed quick gate  
	•	5-seed promotion gate  
	•	artifact checks  
	•	provenance rules  
  
Leave these in place:/Users/stephenbeale/Projects/ToM_AI_Research_Team/eval.py  
/Users/stephenbeale/Projects/ToM_AI_Research_Team/env.py  
/Users/stephenbeale/Projects/ToM_AI_Research_Team/train.py  
  
You are working in the orchestration repo:  
/Users/stephenbeale/Projects/autoresearch-macos-tomx  
  
Not the benchmark repo though you will need to refer to it:  
/Users/stephenbeale/Projects/ToM_AI_Research_TeamAnd these:/Users/stephenbeale/Projects/ToM experiment incumbent  
/Users/stephenbeale/Projects/ToM experiment incumbent v3-omx  
/Users/stephenbeale/Projects/ToM experiment incumbent v4-postevidence-reengage  
/Users/stephenbeale/Projects/ToM experiment incumbent v5-delayedtrust-split-candidate  
  
This task is **Phase 2 only: Encode the current working benchmark method into explicit files with stable schemas**. Do not add runners, adapters, helper scripts, or UI. Do not redefine benchmark semantics, metric definitions, or scientific thresholds. The benchmark repo remains the scientific source of truth; the orchestration repo captures workflow structure, policy, and provenance.      
  
  
/Users/stephenbeale/Projects/ToM_AI_Research_Team/eval.py  
/Users/stephenbeale/Projects/ToM_AI_Research_Team/env.py  
/Users/stephenbeale/Projects/ToM_AI_Research_Team/train.py  
  
  
**Objective**  
  
Create or update these files accordingly. Refer to the schemas below for policy and direction:  
	•	examples/tom_ai_research_team/repo_overlay.yaml  
	•	policies/seed_sets.yaml  
	•	policies/artifact_contract.yaml  
	•	policies/promotion_rules.yaml  
	•	policies/provenance_rules.yaml  
	•	docs/current_method.md  
  
The method being captured must explicitly cover:  
	•	canonical repo root  
	•	safe vs sensitive zones  
	•	smoke validation commands  
	•	quick-gate seed set  
	•	promotion-gate seed set  
	•	artifact checks  
	•	promotion criteria  
	•	provenance rules.    
  
Keep repo-specific facts in the example overlay and policy values, consistent with the planned repo structure.    
  
⸻  
  
**Hard requirements**  
  
**1) Do not invent missing facts**  
  
If a value is not already known from existing docs or current working practice, set it explicitly as one of:  
	•	UNKNOWN  
	•	TODO  
	•	empty list []  
	•	empty mapping {}  
	•	commented placeholder with a short note  
  
Do not fabricate commands, paths, thresholds, seeds, or pass/fail criteria.  
  
**2) Separate fact from inference**  
  
Where relevant, encode whether something is:  
	•	confirmed operational fact  
	•	current convention / working assumption  
	•	unresolved item needing human confirmation  
  
This distinction matters and must be visible in either YAML fields or markdown notes.    
  
**3) Treat artifacts as evidence**  
  
Logs, reports, archives, incumbent snapshots, and comparisons are evidence-bearing artifacts, not disposable output. The artifact schema must reflect that.    
  
**4) Keep benchmark semantics frozen**  
  
Do not move or redefine benchmark semantics from the benchmark repo into this repo. In particular, do not create new meanings for env.py, eval.py, metric names, or scientific thresholds unless they are already explicit in the existing method.    
  
⸻  
  
**Exact file schemas to implement**  
  
**File: examples/tom_ai_research_team/repo_overlay.yaml**  
  
Use exactly this top-level structure:  
  
```
schema_version: "1.0"

overlay_id: "tom_ai_research_team"
overlay_name: "Tom AI Research Team benchmark overlay"
status: "draft"   # draft | active | deprecated

repo:
  canonical_root:
    value: "UNKNOWN"
    status: "unknown"   # confirmed | assumed | unknown
    notes: ""

  benchmark_identity:
    name: "UNKNOWN"
    status: "unknown"
    notes: ""

  source_of_truth_statement: >
    The benchmark repository is the source of truth for benchmark semantics,
    metric definitions, and scientific results. The orchestration repository
    must not silently redefine them.

zones:
  safe_editable:
    - path: "UNKNOWN"
      rationale: ""
      status: "unknown"

  sensitive_read_only:
    - path: "UNKNOWN"
      rationale: ""
      status: "unknown"

  prohibited_without_explicit_human_approval:
    - path: "UNKNOWN"
      rationale: ""
      status: "unknown"

benchmark_invariants:
  files_or_areas_with_frozen_semantics:
    - path: "env.py"
      rationale: "Frozen benchmark semantics unless explicitly approved."
      status: "assumed"
    - path: "eval.py"
      rationale: "Frozen evaluation semantics unless explicitly approved."
      status: "assumed"

  forbidden_changes:
    - "Do not silently redefine metric semantics."
    - "Do not silently change benchmark thresholds."
    - "Do not treat orchestration policy as scientific truth."

run_context:
  incumbent_locations:
    - path: "UNKNOWN"
      purpose: "incumbent snapshots"
      status: "unknown"

  report_locations:
    - path: "UNKNOWN"
      purpose: "reports"
      status: "unknown"

  log_locations:
    - path: "UNKNOWN"
      purpose: "logs"
      status: "unknown"

  archive_locations:
    - path: "UNKNOWN"
      purpose: "archived evidence"
      status: "unknown"

validation:
  smoke_commands:
    - id: "smoke_main"
      command: "TODO"
      shell: "bash"
      working_directory: "UNKNOWN"
      expected_artifacts: []
      status: "unknown"

notes:
  confirmed_facts: []
  working_assumptions: []

```
  open_questions: []  
  
**Requirements for this file**  
	•	Put repo-specific paths here.  
	•	Put zone boundaries here.  
	•	Put smoke command definitions here.  
	•	Keep values narrow and concrete.  
	•	Do not add extra top-level keys.  
  
⸻  
  
**File: policies/seed_sets.yaml**  
  
Use exactly this structure:  
  
```
schema_version: "1.0"

seed_policies:
  quick_gate:
    description: "Fast early gate for bounded validation before wider promotion testing."
    seed_count:
      value: 3
      status: "assumed"   # confirmed | assumed | unknown
    seeds:
      - "TODO"
      - "TODO"
      - "TODO"
    ordering_rule: "Run in listed order."
    pass_to_next_stage_if: "TODO"
    notes: []

  promotion_gate:
    description: "Wider gate used before promotion or incumbent challenge decisions."
    seed_count:
      value: 5
      status: "assumed"
    seeds:
      - "TODO"
      - "TODO"
      - "TODO"
      - "TODO"
      - "TODO"
    ordering_rule: "Run in listed order."
    pass_to_next_stage_if: "TODO"
    notes: []

escalation:
  from_stage: "quick_gate"
  to_stage: "promotion_gate"
  trigger_condition: "TODO"
  forbid_escalation_if:
    - "Smoke validation failed."
    - "Required artifacts missing."
    - "Patch scope is unclear."
  notes: []

notes:
  confirmed_facts: []
  working_assumptions: []

```
  open_questions: []  
  
**Requirements for this file**  
	•	The schema must explicitly encode the 3-seed quick gate and 5-seed promotion gate as the current working structure.    
	•	If actual seed IDs are known, fill them in. Otherwise leave TODO.  
	•	Do not use vague prose instead of structured fields.  
  
⸻  
  
**File: policies/artifact_contract.yaml**  
  
Use exactly this structure:  
  
```
schema_version: "1.0"

artifact_contract:
  required_after_smoke:
    - artifact_id: "smoke_stdout"
      kind: "log"   # log | report | snapshot | summary | comparison | manifest | other
      path_glob: "TODO"
      required: true
      evidence_class: "primary"   # primary | supporting
      validation_rule: "Must exist and be non-empty."

  required_after_quick_gate:
    - artifact_id: "quick_gate_per_seed_logs"
      kind: "log"
      path_glob: "TODO"
      required: true
      evidence_class: "primary"
      validation_rule: "One per required seed."

    - artifact_id: "quick_gate_summary"
      kind: "summary"
      path_glob: "TODO"
      required: true
      evidence_class: "primary"
      validation_rule: "Must summarize seed-level outcomes."

  required_after_promotion_gate:
    - artifact_id: "promotion_gate_per_seed_logs"
      kind: "log"
      path_glob: "TODO"
      required: true
      evidence_class: "primary"
      validation_rule: "One per required seed."

    - artifact_id: "promotion_gate_aggregate_summary"
      kind: "summary"
      path_glob: "TODO"
      required: true
      evidence_class: "primary"
      validation_rule: "Must summarize aggregate and per-seed outcomes."

    - artifact_id: "incumbent_comparison"
      kind: "comparison"
      path_glob: "TODO"
      required: false
      evidence_class: "supporting"
      validation_rule: "Required when promotion decision depends on incumbent comparison."

artifact_bundle_rules:
  minimally_valid_smoke_bundle:
    required_artifact_ids:
      - "smoke_stdout"

  minimally_valid_quick_gate_bundle:
    required_artifact_ids:
      - "quick_gate_per_seed_logs"
      - "quick_gate_summary"

  minimally_valid_promotion_bundle:
    required_artifact_ids:
      - "promotion_gate_per_seed_logs"
      - "promotion_gate_aggregate_summary"

retention:
  preserve_primary_evidence: true
  preserve_supporting_evidence: true
  allow_disposable_outputs: false
  notes:
    - "Evidence-bearing artifacts must not be treated as disposable."

notes:
  confirmed_facts: []
  working_assumptions: []

```
  open_questions: []  
  
**Requirements for this file**  
	•	Encode the minimal valid artifact bundle at each stage.  
	•	Primary evidence must be explicit.  
	•	Add more artifact entries if the method requires them, but do not change the top-level structure.  
  
⸻  
  
**File: policies/promotion_rules.yaml**  
  
Use exactly this structure:  
  
```
schema_version: "1.0"

promotion_policy:
  preconditions:
    smoke_validation_required: true
    required_precondition_checks:
      - check_id: "smoke_pass"
        rule: "Smoke validation command completed successfully."
        status: "active"
      - check_id: "artifact_bundle_present"
        rule: "Required stage artifacts exist."
        status: "active"

  quick_gate:
    enabled: true
    stage_name: "quick_gate"
    success_criteria:
      - criterion_id: "quick_gate_defined_seed_set_completed"
        rule: "All required quick-gate seeds completed."
      - criterion_id: "quick_gate_artifacts_present"
        rule: "All required quick-gate artifacts are present."
      - criterion_id: "quick_gate_promotion_condition"
        rule: "TODO"
    disqualifying_failures:
      - "Any required quick-gate seed missing."
      - "Any required quick-gate artifact missing."
      - "Semantic drift detected."
      - "Patch scope exceeds bounded change expectations."

  promotion_gate:
    enabled: true
    stage_name: "promotion_gate"
    success_criteria:
      - criterion_id: "promotion_gate_defined_seed_set_completed"
        rule: "All required promotion-gate seeds completed."
      - criterion_id: "promotion_gate_artifacts_present"
        rule: "All required promotion-gate artifacts are present."
      - criterion_id: "promotion_gate_decision_condition"
        rule: "TODO"
    disqualifying_failures:
      - "Any required promotion-gate seed missing."
      - "Any required promotion-gate artifact missing."
      - "Semantic drift detected."
      - "Provenance incomplete."

decision_outputs:
  allowed_decisions:
    - "reject"
    - "hold"
    - "promote_candidate"
    - "needs_human_review"

  decision_requirements:
    reject:
      minimum_evidence: "Stage evidence sufficient to show failure or non-promotion."
    hold:
      minimum_evidence: "Stage evidence present but insufficient for promotion."
    promote_candidate:
      minimum_evidence: "Promotion-gate evidence complete and policy conditions satisfied."
    needs_human_review:
      minimum_evidence: "Conflicting, ambiguous, or incomplete evidence requiring review."

notes:
  confirmed_facts: []
  working_assumptions: []

```
  open_questions: []  
  
**Requirements for this file**  
	•	Promotion criteria must be explicit and machine-readable.  
	•	Disqualifying failure modes must be explicit.  
	•	Keep scientific thresholds out unless already confirmed by the method.  
  
⸻  
  
**File: policies/provenance_rules.yaml**  
  
Use exactly this structure:  
  
```
schema_version: "1.0"

provenance_policy:
  required_run_metadata:
    - field: "run_id"
      required: true
      description: "Unique identifier for the run."
    - field: "timestamp_utc"
      required: true
      description: "UTC timestamp for run start or record creation."
    - field: "operator"
      required: true
      description: "Human or agent responsible for the run."
    - field: "repo_root"
      required: true
      description: "Canonical benchmark repo root used for the run."
    - field: "git_commit"
      required: false
      description: "Benchmark repo commit or equivalent version identifier."
    - field: "patch_identifier"
      required: true
      description: "Identifier linking run to the specific patch or patch family."
    - field: "hypothesis_identifier"
      required: true
      description: "Identifier linking run to the bounded hypothesis under test."
    - field: "stage"
      required: true
      description: "smoke | quick_gate | promotion_gate | other"
    - field: "seed"
      required: false
      description: "Seed identifier for per-seed runs."
    - field: "command"
      required: true
      description: "Exact executed command."
    - field: "working_directory"
      required: true
      description: "Directory from which the command was run."

  linkage_rules:
    - rule_id: "patch_to_hypothesis"
      rule: "Every patch identifier must map to one bounded hypothesis."
    - rule_id: "run_to_patch"
      rule: "Every run record must identify the patch under test."
    - rule_id: "run_to_artifacts"
      rule: "Every required artifact should be attributable to a run_id."
    - rule_id: "seed_to_run"
      rule: "Per-seed outputs must record the seed used."

  evidence_interpretation:
    require_confirmed_vs_inferred_split: true
    require_uncertainty_notes: true
    rules:
      - "Separate artifact-observed facts from interpretation."
      - "Mark unresolved issues explicitly."
      - "Do not present inference as confirmed result."

  retention_rules:
    preserve_run_records: true
    preserve_primary_artifacts: true
    preserve_stage_summaries: true
    preserve_failed_run_evidence: true
    notes: []

notes:
  confirmed_facts: []
  working_assumptions: []

```
  open_questions: []  
  
**Requirements for this file**  
	•	Provenance must cover run linkage, command provenance, seed provenance, and evidence interpretation.  
	•	Failed run evidence must not be silently discarded.  
  
⸻  
  
**File: docs/current_method.md**  
  
Use exactly this section structure:  
  
```
# Current Working Method

## Scope

## Canonical Repo Root

## Safe Editable Zones

## Sensitive or Read-Only Zones

## Smoke Validation Commands

## Quick-Gate Seed Set

## Promotion-Gate Seed Set

## Artifact Checks

## Promotion Criteria

## Provenance Rules

## Confirmed Facts

## Working Assumptions

## Open Questions

```
  
**Requirements for this file**  
	•	Keep it concise.  
	•	It must summarize the YAML files, not replace them.  
	•	Where values are unknown, say so directly.  
	•	Do not bury operational rules only in markdown; the YAML is authoritative.  
  
⸻  
  
**Implementation instructions**  
	1.	Read the existing planning docs and align with them:  
	•	separate orchestration repo  
	•	policy-first structure  
	•	repo-specific overlays  
	•	terminal-first, local-first, inspectable workflow.      
	2.	Create the files above if missing.  
	3.	If a file already exists, update it to match the exact schema above.  
	4.	Fill in confirmed values where they are already known from current working practice.  
	5.	Leave unknown values as explicit placeholders. Do not invent them.  
	6.	Keep naming stable and boring. Prefer exactness over cleverness.  
	7.	Do not create any additional files except these six unless strictly necessary for schema consistency.  
  
⸻  
  
**Output requirements after editing**  
  
After making the edits, return:  
  
**A. Files changed**  
  
A short list of the files created or updated.  
  
**B. Confirmed values filled in**  
  
A short list of any values that were filled with confirmed information rather than placeholders.  
  
**C. Remaining placeholders needing human confirmation**  
  
List all remaining UNKNOWN / TODO items that matter for execution.  
  
**D. Assumptions made**  
  
List only actual assumptions.  
  
**E. Execution-readiness judgment**  
  
State one of:  
	•	not execution-ready  
	•	partially execution-ready  
	•	execution-ready  
  
Then give a 3-6 line explanation focused on whether a future Codex/local runner could execute the workflow without hidden tribal knowledge.  
  
⸻  
  
**Quality bar**  
  
The result is good only if:  
	•	a future agent can locate the canonical repo root  
	•	a future agent can tell what may be edited and what must not be touched  
	•	a future agent can run smoke validation from an explicit command field  
	•	a future agent can tell which seeds belong to quick gate vs promotion gate  
	•	a future agent can tell which artifacts are required at each stage  
	•	a future agent can tell what blocks promotion  
	•	a future agent can reconstruct provenance for a run  
	•	unknowns are explicit rather than hidden in prose  
  
Do not produce a loose planning note. Produce concrete files with the exact schemas above.  
