---
name: autoresearch-lab
description: Run or resume autonomous train.py experimentation in this repository, including setup checks, autoresearch branch creation, fixed-budget training runs, run.log parsing, and results.tsv updates.
---

# Autoresearch Lab

Use this skill when the user wants to start, resume, or operationalize the autonomous research loop in this repository.

## Use When

- The user wants to kick off or resume `train.py` experiments in this repo
- The user wants help managing `results.tsv`, `run.log`, or experiment branches
- The user wants the repo's original `program.md` workflow turned into a reusable Codex surface

## Do Not Use When

- The task is unrelated to the training experiment loop
- The user wants broad refactors outside the experiment surface
- The work requires changing dependencies or modifying `prepare.py`

## Repo Contract

- Treat `train.py` as the primary experiment surface
- Do not modify `prepare.py`
- Do not add new dependencies
- Preserve the fixed-time-budget benchmark semantics
- Keep git history usable for keep/discard decisions

## Workflow

1. Read `README.md`, `train.py`, and `prepare.py` for repo context.
2. Run the helper to verify prerequisites:

```bash
python .codex/skills/autoresearch-lab/scripts/autoresearch_ops.py check-setup --json
```

3. If the user is starting a fresh run, create a dedicated branch:

```bash
python .codex/skills/autoresearch-lab/scripts/autoresearch_ops.py create-run-branch --tag apr10
```

4. Ensure the results file exists:

```bash
python .codex/skills/autoresearch-lab/scripts/autoresearch_ops.py ensure-results
```

5. For each experiment iteration:
- Edit `train.py`
- Commit intentionally
- Launch a bounded run:

```bash
python .codex/skills/autoresearch-lab/scripts/autoresearch_ops.py run --log run.log --timeout 600
```

- Parse the summary:

```bash
python .codex/skills/autoresearch-lab/scripts/autoresearch_ops.py parse-log run.log --json
```

- Append the result row:

```bash
python .codex/skills/autoresearch-lab/scripts/autoresearch_ops.py append-result \
  --commit abc1234 \
  --status keep \
  --description "baseline" \
  --log run.log
```

6. Keep or revert the experiment based on the metric and simplicity tradeoff.

## References

- Read `references/protocol.md` for the detailed experiment policy, logging schema, and crash/timeout handling guidance.
- `program.md` is still the legacy prompt surface, but prefer this skill plus the helper script for deterministic steps.

