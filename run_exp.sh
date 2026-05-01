#!/usr/bin/env bash
# Run one compressor experiment, parse the output, append a row to results.tsv.
# Usage: ./run_exp.sh <COMPRESSOR_NAME> <substrate_tag> [extra_env_vars...]
# Example: ./run_exp.sh sliding_W128 medium DEPTH=6 ASPECT_RATIO=64
#
# Reads:    train.py stdout
# Writes:   results.tsv  (one new row, status auto-set by score)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE=copy

COMPRESSOR="${1:?usage: run_exp.sh COMPRESSOR substrate_tag [env...]}"
SUBSTRATE="${2:-medium}"
shift 2 || true

# Pass any further args through as env (e.g. DEPTH=10 ASPECT_RATIO=80)
LOG="run.log"
COMPRESSOR="$COMPRESSOR" "$@" uv run train.py > "$LOG" 2>&1 || {
  echo "CRASH: $COMPRESSOR ($SUBSTRATE)"
  HEAD_SHA=$(git rev-parse --short HEAD)
  printf '%s\t%.6f\t%.4f\t%.6f\t%.6f\t%s\t%s\n' \
    "$HEAD_SHA" 0.0 0.0 0.0 0.0 crash "$COMPRESSOR [$SUBSTRATE]" >> results.tsv
  return 1 2>/dev/null || exit 1
}

# Parse the summary block.
ratio=$(grep -E "^compression_ratio:" "$LOG" | awk '{print $2}')
delta=$(grep -E "^val_bpb_delta:"    "$LOG" | awk '{print $2}')
cbpb=$(grep -E "^compressed_bpb:"    "$LOG" | awk '{print $2}')
bpb=$(grep -E "^val_bpb:"            "$LOG" | awk '{print $2}')
bpt=$(grep -E "^compressed_bytes_per_tok:" "$LOG" | awk '{print $2}')
score=$(grep -E "^compression_score:" "$LOG" | awk '{print $2}')
score20=$(grep -E "^score_alpha20:"  "$LOG" | awk '{print $2}')
score50=$(grep -E "^score_alpha50:"  "$LOG" | awk '{print $2}')
name=$(grep -E "^compressor_name:"   "$LOG" | awk '{print $2}')

# Determine status based on running-max α20 score WITHIN current substrate.
# α20 is the median deployment regime: high enough to penalize quality loss
# (so eviction methods at Δ>0.5 can't masquerade as winners on ratio alone)
# but low enough to reward useful compression.
# We also require Δbpb < 0.10 — even a high-α20 score is meaningless if the
# compressor destroys quality, so a strict gate prevents pathological keeps.
running_max20=$(awk -F'\t' -v stag="[$SUBSTRATE " '
  NR>1 && $6=="keep" && index($7, stag) {
    a20 = ""
    n = split($7, parts, "α20=")
    if (n > 1) {
      split(parts[2], rest, " ")
      a20 = rest[1]
    }
    if (a20 != "" && (a20+0 > m || m == "")) m = a20+0
  } END { print (m=="")?-1e9:m
}' results.tsv)
status="discard"
awk -v s="$score20" -v m="$running_max20" -v d="$delta" \
  'BEGIN { exit !(s+0 > m+0 + 1e-9 && d+0 < 0.10) }' && status="keep"

HEAD_SHA=$(git rev-parse --short HEAD)
desc="$name [$SUBSTRATE bpt=$bpt α20=$score20 α50=$score50]"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$HEAD_SHA" "$score" "$ratio" "$delta" "$cbpb" "$status" "$desc" >> results.tsv
echo "$status: $name | ratio=$ratio Δ=$delta score=$score (α20=$score20 α50=$score50)"
