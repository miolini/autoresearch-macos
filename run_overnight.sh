#!/usr/bin/env bash
# Launch Claude Code in fully autonomous "never ask permission" mode against
# the autoresearch project. Auto-restarts on any unexpected exit.
#
# Prerequisites:
#   - `claude` CLI installed (npm install -g @anthropic-ai/claude-code)
#   - logged in (claude login) OR ANTHROPIC_API_KEY exported
#   - setup_runpod.sh already ran successfully (so prepare.py + uv sync are done)
#
# Behavior:
#   - Spawns `claude` with --dangerously-skip-permissions --permission-mode acceptEdits
#   - Pipes a startup prompt that points the agent at program.md
#   - If the process exits (crash, OOM, stale connection), waits 30 s and respawns
#   - Logs everything to overnight.log
#
# Stop with: Ctrl-C from the controlling terminal, or `pkill -f run_overnight.sh`.
#
# Notes:
#   - The agent state lives in git on the autoresearch/<tag> branch and in
#     results.tsv. A restart just rejoins the loop where it left off.
#   - You can also run this without the auto-restart loop:
#         claude --dangerously-skip-permissions --permission-mode acceptEdits \
#                "Read program.md and run the autonomous KV-cache compression loop."
set -euo pipefail

cd "$(dirname "$0")"

LOG=overnight.log
PROMPT='You are the engineer driving the KV-cache compression project for the
ICML AdaptFM workshop. Read program.md and paper.md, then loop the workflow
in program.md (critique → plan → implement → run → log → commit →
update paper/figures → repeat). Branch: autoresearch/apr29-kvcompress (or
the latest *-kvcompress). Top priorities: (1) broaden compression-family
coverage beyond quantization (eviction, low-rank, head sharing, hybrids);
(2) substrate-scale sweep; (3) airtight byte-accounting in Appendix B.
Never ask for permission. Never stop voluntarily. Edit files, commit, and
push when there is meaningful progress.'

if ! command -v claude >/dev/null 2>&1; then
    echo "claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
    echo "Or run claude from your laptop pointed at this directory via SSH."
    exit 1
fi

echo "[overnight] starting autonomous Claude loop in $(pwd)"
echo "[overnight] logs -> $LOG (tail -f $LOG to follow)"
echo "[overnight] stop with Ctrl-C or 'pkill -f run_overnight.sh'"

iter=0
while true; do
    iter=$((iter+1))
    {
        echo ""
        echo "=========================================================="
        echo "[overnight] iteration $iter @ $(date -u +%FT%TZ)"
        echo "=========================================================="
    } | tee -a "$LOG"

    # --dangerously-skip-permissions   = no permission prompts
    # --permission-mode acceptEdits    = auto-accept file edits
    # --model claude-opus-4-7          = use the latest Opus
    # We pipe the prompt on stdin; claude reads the first message and starts.
    set +e
    echo "$PROMPT" | claude \
        --dangerously-skip-permissions \
        --permission-mode acceptEdits \
        --print 2>&1 | tee -a "$LOG"
    rc=$?
    set -e

    echo "[overnight] iteration $iter exited rc=$rc — restarting in 30 s" | tee -a "$LOG"
    sleep 30
done
