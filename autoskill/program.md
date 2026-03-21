# autoskill

Autonomous skill evolution using the autoresearch pattern.

## Concept

Instead of evolving neural network training code (`train.py`), we evolve Claude Code skills (`skill.md`). Instead of measuring `val_bpb`, we measure **benchmark pass rate**.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar20`). Branch `autoskill/<tag>` must not exist.
2. **Create the branch**: `git checkout -b autoskill/<tag>` from current HEAD.
3. **Read the files**:
   - `skill.md` — the skill you're evolving (YOU EDIT THIS)
   - `benchmarks/*.md` — test cases with expected behavior (DO NOT EDIT)
   - `evaluate.py` — runs benchmarks and computes score (DO NOT EDIT)
4. **Initialize results.tsv**: Create with header row only.
5. **Confirm and go**.

## The Skill Under Evolution

`skill.md` is a Claude Code skill that you iteratively improve. The current example is a **code explainer skill** — it takes code and explains what it does.

## Benchmarks

Each file in `benchmarks/` is a test case with:
- **Input**: Code to explain
- **Expected behaviors**: What a good explanation must include
- **Scoring rubric**: Criteria for 0/0.5/1 scores

The evaluation runs the skill against each benchmark and scores it.

## Evaluation

```bash
uv run evaluate.py > eval.log 2>&1
```

Output format:
```
---
pass_rate:     0.750
total_score:   6.0
max_score:     8.0
benchmarks:    8
avg_time_sec:  2.3
```

Extract the metric:
```bash
grep "^pass_rate:" eval.log
```

## Logging Results

Log to `results.tsv` (tab-separated):

```
commit	pass_rate	status	description
```

- commit: git hash (7 chars)
- pass_rate: 0.000 to 1.000
- status: `keep`, `discard`, or `crash`
- description: what you tried

Example:
```
commit	pass_rate	status	description
a1b2c3d	0.625	keep	baseline
b2c3d4e	0.750	keep	add step-by-step breakdown instruction
c3d4e5f	0.625	discard	require line-by-line comments (too verbose)
d4e5f6g	0.875	keep	add edge case handling section
```

## The Experiment Loop

LOOP FOREVER:

1. Read current `skill.md` and recent results
2. Propose a modification (add instruction, rephrase, restructure)
3. Edit `skill.md`
4. git commit
5. Run: `uv run evaluate.py > eval.log 2>&1`
6. Extract score: `grep "^pass_rate:" eval.log`
7. If empty, run crashed — `tail -50 eval.log` to debug
8. Log to results.tsv
9. If pass_rate improved: keep commit
10. If equal or worse: `git reset --hard HEAD~1` and try different idea

## Modification Ideas

- Add explicit instructions for common failure modes
- Restructure the workflow steps
- Add examples of good vs bad output
- Make instructions more specific or more general
- Add constraints (length, format, depth)
- Remove unnecessary instructions (simplicity wins)
- Add edge case handling
- Improve the rubric/criteria sections

## Simplicity Criterion

All else equal, simpler is better:
- +0.05 pass_rate with 10 new lines? Maybe keep
- +0.01 pass_rate with 20 new lines? Probably discard
- Equal pass_rate with fewer lines? Definitely keep

## NEVER STOP

Once the loop begins, do NOT pause to ask if you should continue. Run until manually interrupted. If stuck, try more radical changes or revisit discarded ideas with tweaks.
