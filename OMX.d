# OMX Research Workflow (Train.py-Only)

## Owner
tom

## Current Priority Queue

1. Deadlock-focused micro-iteration.
2. Lightweight auxiliary-loss-only next-action comparison.
3. One history-length comparison.
4. One belief-stabilization comparison.

## New Incumbent Candidate Snapshot

Created:

- /Users/stephenbeale/Projects/ToM experiment incumbent v4-postevidence-reengage/train.py
- /Users/stephenbeale/Projects/ToM experiment incumbent v4-postevidence-reengage/INCUMBENT_CANDIDATE_NOTE.md
- /Users/stephenbeale/Projects/ToM experiment incumbent v4-postevidence-reengage/RESULTS_TABLE.md

### Repo-Local Subagents

Also added repo-local Codex agents so they are editable in-repo and available from this workspace:

- /Users/stephenbeale/Projects/ToM_AI_Research_Team/.codex/agents/tom-train-tuner.toml
- /Users/stephenbeale/Projects/ToM_AI_Research_Team/.codex/agents/tom-results-judge.toml
- /Users/stephenbeale/Projects/ToM_AI_Research_Team/.codex/agents/tom-incumbent-curator.toml
- /Users/stephenbeale/Projects/ToM_AI_Research_Team/.codex/agents/README.md

---

## Command-Based Workflow for Fine-Tuning with oh-my-codex

oh-my-codex extends Codex with prompts, skills, hooks, agent teams, and runtime tooling. The published quickstart is:

- clone repo
- install dependencies
- build
- link CLI
- run setup

For this project, use OMX as a strict train.py-only local research harness around frozen Variant 1, not as a general coding shell.

## A) Install and Initialize OMX (one time)

```bash
git clone https://github.com/Yeachan-Heo/oh-my-codex.git
cd oh-my-codex
npm install
npm run build
npm link
omx setup
```

## B) Prepare Your Experiment Repo

```bash
cd "/Users/stephenbeale/Projects/ToM AI Research Team"
source .venv/bin/activate
python scripts/local_runner.py --train-episodes 5 --seed 7 --output-root logs/pre_omx_smoke
```

Goal:

- Confirm the frozen benchmark still runs before Codex changes anything.

## C) Define a Hard Operating Rule for Codex

Create an OMX_TASK.md file in repo root:

```text
You are in train.py-only local quality mode for frozen Variant 1.

Rules:
- edit train.py only
- do not edit env.py
- do not edit eval.py
- do not edit scripts/local_runner.py
- do not change benchmark semantics
- use smoke run only for breakage checks
- use 800-episode runs as the scientific gate
- optimize ToMCoordScore without worsening deadlock
- preserve belief-guided switching under ambiguity

Primary comparison target:
- current incumbent snapshot

Preferred directions:
- belief-to-policy gating
- deadlock-aware micro-improvements
- lightweight auxiliary losses
- observation-history tuning

Forbidden directions:
- benchmark redesign
- multi-file refactors
- Azure work
- dashboard work
```

## D) Run OMX with One Narrow Task at a Time

Start with a single bounded task:

```bash
codex "Read OMX_TASK.md and the current train.py. Make one train.py-only change aimed at reducing residual deadlock without weakening ambiguity handling. Then run:
python scripts/local_runner.py --train-episodes 5 --seed 7 --output-root logs/omx_smoke_1
and if that passes, run:
python scripts/local_runner.py --train-episodes 800 --seed 7 --output-root logs/omx_full_1
Summarize the patch, metrics, and keep/discard recommendation."
```

If OMX exposes Codex through its own wrappers in your install, use the same task content there. The critical part is process discipline, not command alias.

## E) Force Comparison Against the Incumbent

After each full run:

```bash
cat logs/omx_full_1/selection/selection.json
```

Then request a structured comparison:

```bash
codex "Compare logs/omx_full_1/selection/selection.json against the incumbent metrics. State:
1. did ToMCoordScore improve?
2. did deadlock worsen?
3. did collisions worsen?
4. did switching improve or hold?
5. keep/discard?"
```

## F) Promote Only on Explicit Success

If the run wins:

```bash
mkdir -p "/Users/stephenbeale/Projects/ToM experiment incumbent v3-omx"
cp train.py "/Users/stephenbeale/Projects/ToM experiment incumbent v3-omx/train.py"
cp logs/omx_full_1/selection/selection.json "/Users/stephenbeale/Projects/ToM experiment incumbent v3-omx/selection.json"
```

Then create promotion note:

```bash
cat > "/Users/stephenbeale/Projects/ToM experiment incumbent v3-omx/INCUMBENT_NOTE.md" <<'EOF'
Promoted from OMX train.py-only pass.
Benchmark: Variant 1 frozen ambiguous bottleneck
Reason:
- deadlock not worsened (0.1 vs baseline 0.1)
- higher ToMCoordScore
- lower collision
- higher success
- much better ambiguity/intention metrics
Patch: One logical change to the post-evidence decision prior in ToMPolicy._apply_decision_prior:
- tightened late_yield in non-narrow contexts so it only prefers yielding when the partner is actually pressing
- added a soft_reengage path to bias PROCEED/PROBE over WAIT/YIELD when evidence is out, margin is not narrow, the partner stays soft, and belief is no longer strongly assertive
EOF
```