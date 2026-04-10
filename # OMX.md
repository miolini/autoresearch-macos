# OMX

  1.	deadlock-focused micro-iteration
  2.	lightweight auxiliary-loss-only next-action comparison
  3.	one history-length comparison
  4.	one belief-stabilization comparison

## New incumbent candidate snapshot

  Created:

  - /Users/stephenbeale/Projects/ToM experiment incumbent v4-
    postevidence-reengage/train.py
  - /Users/stephenbeale/Projects/ToM experiment incumbent v4-
    postevidence-reengage/INCUMBENT_CANDIDATE_NOTE.md
  - /Users/stephenbeale/Projects/ToM experiment incumbent v4-
    postevidence-reengage/RESULTS_TABLE.md

  ### Repo-local subagents

  Also added repo-local Codex agents so they’re editable in-repo and
  available from this workspace:

  - /Users/stephenbeale/Projects/ToM_AI_Research_Team/.codex/agents/
    tom-train-tuner.toml
  - /Users/stephenbeale/Projects/ToM_AI_Research_Team/.codex/agents/
    tom-results-judge.toml
  - /Users/stephenbeale/Projects/ToM_AI_Research_Team/.codex/agents/
    tom-incumbent-curator.toml
  - /Users/stephenbeale/Projects/ToM_AI_Research_Team/.codex/agents/
    README.md

2. Command-based workflow for fine-tuning this experiment with oh-my-codex

oh-my-codex is positioned as a Codex enhancer that adds prompts, skills, hooks, agent teams, and runtime tooling around Codex; its published quickstart is git clone, npm install, npm run build, npm link, then omx setup.  

For your project, I would not use it as a generic coding shell. I would use it as a strict train.py-only local research harness around the frozen Variant 1 benchmark.

Below is the workflow I would use.

⸻

Command-based oh-my-codex workflow

A. Install and initialize OMX once

git clone https://github.com/Yeachan-Heo/oh-my-codex.git
cd oh-my-codex
npm install
npm run build
npm link
omx setup

That matches the project’s published setup flow.  

B. Prepare your experiment repo

cd "/Users/stephenbeale/Projects/ToM AI Research Team"
source .venv/bin/activate
python scripts/local_runner.py --train-episodes 5 --seed 7 --output-root logs/pre_omx_smoke

Goal:
	•	confirm the frozen benchmark still runs before Codex changes anything

C. Create a hard operating rule for Codex

Make a file like OMX_TASK.md in the repo root:

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

D. Run OMX with one narrow task at a time

Start with a single, bounded task such as:

codex "Read OMX_TASK.md and the current train.py. Make one train.py-only change aimed at reducing residual deadlock without weakening ambiguity handling. Then run:
python scripts/local_runner.py --train-episodes 5 --seed 7 --output-root logs/omx_smoke_1
and if that passes, run:
python scripts/local_runner.py --train-episodes 800 --seed 7 --output-root logs/omx_full_1
Summarize the patch, metrics, and keep/discard recommendation."

If OMX exposes Codex through its own wrappers in your install, use the same task content there; the important part is the discipline, not the shell name.

E. Force comparison against the incumbent

After every run, require a structured note:

cat logs/omx_full_1/selection/selection.json

Then ask Codex for a comparison:

codex "Compare logs/omx_full_1/selection/selection.json against the incumbent metrics. State:
1. did ToMCoordScore improve?
2. did deadlock worsen?
3. did collisions worsen?
4. did switching improve or hold?
5. keep/discard?"

F. Promote only on explicit success

If the run wins:

mkdir -p "/Users/stephenbeale/Projects/ToM experiment incumbent v3-omx"
cp train.py "/Users/stephenbeale/Projects/ToM experiment incumbent v3-omx/train.py"
cp logs/omx_full_1/selection/selection.json "/Users/stephenbeale/Projects/ToM experiment incumbent v3-omx/selection.json"

Then write a note:

cat > "/Users/stephenbeale/Projects/ToM experiment incumbent v3-omx/INCUMBENT_NOTE.md" <<'EOF'
Promoted from OMX train.py-only pass.
Benchmark: Variant 1 frozen ambiguous bottleneck
Reason: - deadlock not worsened (0.1 vs baseline 0.1)
        - higher ToMCoordScore
        - lower collision
        - higher success
        - much better ambiguity/intention metrics
Patch: One logical change to the post-evidence decision prior in ToMPolicy._apply_decision_prior:
        - tightened late_yield in non-narrow contexts so it only prefers yielding when the partner is actually pressing
        - added a soft_reengage path to bias PROCEED/PROBE over WAIT/YIELD when evidence is out, margin is not narrow, the partner stays soft, and belief is no longer strongly assertive
EOF
