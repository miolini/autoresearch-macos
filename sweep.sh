#!/usr/bin/env bash
# Run a curated sweep of compressors on a substrate, unattended.
# No Claude needed; pure shell. Each compressor is one row in results.tsv,
# committed individually. Logs to sweep_<substrate>.log.
#
# Usage:
#   ./sweep.sh <substrate-tag> [PROFILE] [extra-env...]
#
# Substrate tags map to (DEPTH, ASPECT_RATIO, HEAD_DIM, DEVICE_BATCH_SIZE):
#   small   : DEPTH=3  ASPECT_RATIO=40 HEAD_DIM=96  DEVICE_BATCH_SIZE=16
#   medium  : DEPTH=6  ASPECT_RATIO=64 HEAD_DIM=96  DEVICE_BATCH_SIZE=16  (default)
#   large   : DEPTH=10 ASPECT_RATIO=80 HEAD_DIM=96  DEVICE_BATCH_SIZE=4
#   hd64    : DEPTH=6  ASPECT_RATIO=64 HEAD_DIM=64  DEVICE_BATCH_SIZE=16
#   hd128   : DEPTH=6  ASPECT_RATIO=64 HEAD_DIM=128 DEVICE_BATCH_SIZE=16
#
# Profiles:
#   core      : the 7-compressor minimal set (int8, int4, int2, int4_g16,
#               int4_asym, hybrid_R64_int4, mixed_K8_V4) — ~10 min
#   quant     : full quantization zoo (~12 compressors)
#   evict     : eviction zoo (sliding/sink/topk)
#   lowrank   : SVD + randproj
#   hybrid    : hybrid recents + stacks
#   all       : everything (~30 compressors, ~30 min)  (default)
#
# Examples:
#   ./sweep.sh small
#   ./sweep.sh large all
#   ./sweep.sh hd64 quant
#
# After completion, run `uv run figures.py` to refresh figures.

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE=copy

SUB="${1:?usage: ./sweep.sh <substrate-tag> [profile]}"
PROFILE="${2:-all}"
shift; [ $# -gt 0 ] && shift || true

case "$SUB" in
  small)  DEPTH=3  ASPECT_RATIO=40 HEAD_DIM=96  DEVICE_BATCH_SIZE=16 ;;
  medium) DEPTH=6  ASPECT_RATIO=64 HEAD_DIM=96  DEVICE_BATCH_SIZE=16 ;;
  large)  DEPTH=10 ASPECT_RATIO=80 HEAD_DIM=96  DEVICE_BATCH_SIZE=4  ;;
  hd64)   DEPTH=6  ASPECT_RATIO=64 HEAD_DIM=64  DEVICE_BATCH_SIZE=16 ;;
  hd128)  DEPTH=6  ASPECT_RATIO=64 HEAD_DIM=128 DEVICE_BATCH_SIZE=16 ;;
  *) echo "unknown substrate tag: $SUB"; exit 1 ;;
esac
export DEPTH ASPECT_RATIO HEAD_DIM DEVICE_BATCH_SIZE

# Compressor lists per profile.
QUANT="int8 int4 int2 int4_g16 int4_g8 int4_g32 int4_asym mixed_K8_V4 mixed_K4_V8 mixed_K8_V2 mixed_K4_V2"
EVICT="sliding_W64 sliding_W128 sliding_W256 sliding_W512 sink4_W64 sink4_W128 sink4_W256 topk_knorm_25pct topk_knorm_50pct topk_knorm_75pct"
LOWRANK="svd_r8 svd_r16 svd_r32 randproj_r32 headprune_1 headprune_2"
HYBRID="hybrid_R64_int2 hybrid_R64_int4 hybrid_R128_int2 stack:sliding_W256+int4 stack:sink4_W256+int4 stack:headprune_1+int4"
CORE="int8 int4 int2 int4_g16 int4_asym hybrid_R64_int4 mixed_K8_V4"

case "$PROFILE" in
  core)    LIST="$CORE" ;;
  quant)   LIST="$QUANT" ;;
  evict)   LIST="$EVICT" ;;
  lowrank) LIST="$LOWRANK" ;;
  hybrid)  LIST="$HYBRID" ;;
  all)     LIST="$QUANT $EVICT $LOWRANK $HYBRID" ;;
  *) echo "unknown profile: $PROFILE"; exit 1 ;;
esac

LOG="sweep_${SUB}.log"
{
  echo "==================================================================="
  echo "[sweep] substrate=$SUB profile=$PROFILE"
  echo "[sweep] DEPTH=$DEPTH ASPECT_RATIO=$ASPECT_RATIO HEAD_DIM=$HEAD_DIM DEVICE_BATCH_SIZE=$DEVICE_BATCH_SIZE"
  echo "[sweep] $(echo $LIST | wc -w) compressors: $LIST"
  echo "[sweep] start: $(date -u +%FT%TZ)"
  echo "==================================================================="
} | tee -a "$LOG"

# Add a substrate marker if there is no recent one for this tag.
if ! tail -20 results.tsv | grep -q "===== substrate.*$SUB ====="; then
  echo "PENDING	0.000000	0.0000	0.000000	0.000000	marker	===== substrate change: $SUB (DEPTH=$DEPTH, ASPECT_RATIO=$ASPECT_RATIO, HEAD_DIM=$HEAD_DIM) =====" >> results.tsv
fi

n=0
ok=0
fail=0
for c in $LIST; do
  n=$((n+1))
  echo "[sweep] [$n/$(echo $LIST | wc -w)] $c on $SUB at $(date -u +%FT%TZ)" | tee -a "$LOG"
  if ./run_exp.sh "$c" "$SUB" >> "$LOG" 2>&1; then
    ok=$((ok+1))
  else
    fail=$((fail+1))
    echo "[sweep] FAIL: $c (continuing)" | tee -a "$LOG"
  fi
done

{
  echo "[sweep] done: $(date -u +%FT%TZ)"
  echo "[sweep] $ok succeeded, $fail failed"
  echo "[sweep] now run: uv run figures.py"
} | tee -a "$LOG"
