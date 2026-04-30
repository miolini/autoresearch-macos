# autoresearch — KV-Cache Compression Discovery

This is an autonomous research project. **Goal: discover novel KV-cache
compression methods that reduce attention K/V memory while preserving model
quality.** Each experiment trains a small GPT for 5 minutes, then runs a
compression eval that reports a composite `compression_score`.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr29`).
   The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader,
     `evaluate_bpb`. Do not modify.
   - `train.py` — the file you modify. Only edit the `KVCompressor` class
     and the `agent_compressor = ...` line in the final eval section, plus
     hyperparameter tweaks if needed for stable training.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data
   shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
   The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation and DO NOT STOP until
the human manually interrupts you.

## The research problem

Standard transformer KV cache stores K and V tensors in BF16/FP16 per layer
per token, costing `2 * n_kv_head * head_dim * 2 bytes` per token per layer.
For long contexts and large models this dominates inference memory.

Your job is to design **alternative representations** of K, V that are
cheaper to store but reproduce the original attention output closely enough
that `val_bpb` barely changes.

## The metric

After each run the script prints:

```
baseline_bpb:             X         # uncompressed cache, sets the quality floor
compressed_bpb:           Y         # cache with YOUR compressor
val_bpb_delta:            Y - X     # >0 means quality lost
baseline_bytes_per_tok:   Bb        # uncompressed cache bytes / (token * layer)
compressed_bytes_per_tok: Bc        # compressed cache bytes / (token * layer)
compression_ratio:        Bb / Bc   # higher = better
compression_score:        ratio - 10 * max(delta, 0)   # higher = better
```

Optimize **`compression_score`**. Identity compressor scores exactly 1.0.
Anything > 1.0 is a real win. The alpha (10.0) means a 0.01 bpb regression
costs you 0.1 of compression ratio — pick that tradeoff carefully.

## What you CAN do

- Edit the `KVCompressor` class in `train.py` (the entire body — `compress`,
  `decompress`, `__init__`, helpers, new fields).
- Replace `agent_compressor = KVCompressor(config)` with a custom subclass /
  factory that swaps in your method.
- Stack multiple compression tricks (e.g. quantization + low-rank).
- Adjust the byte-counting in `compress()` to honestly reflect storage —
  cheating by under-reporting bytes is an immediate revert.

## What you CANNOT do

- Modify the eval logic in `prepare.py` (`evaluate_bpb`, `make_dataloader`,
  `MAX_SEQ_LEN`, `EVAL_TOKENS`, the BPE tokenizer, or any training-data
  prep code). The eval is the ground truth. The device-detection helper
  at the top of the file is infrastructure, not eval — leave it alone too.
- Modify the training loop, model architecture, optimizer, or hyperparameters
  (those are not the research question — the model is just a frozen substrate
  for testing compression). If a run crashes during training, fix the
  compressor, do not touch training.
- Install new packages.
- Lie about `n_bytes` returned from `compress()`. It must equal the actual
  storage cost of `state` (sum of tensor `numel * element_size` for any
  tensors in `state`, plus any scalar metadata).

## Compression directions to explore

Pick from these or combine them. Start simple, then stack.

1. **Quantization**
   - INT8 / INT4 per-channel quantization of K, V (scale + zero-point).
   - Asymmetric quantization (different ranges for K vs V).
   - Mixed precision: K in lower precision than V (or vice versa).
   - Group-wise quantization (per head, per channel-block).

2. **Low-rank approximation**
   - SVD of K, V down to rank r << head_dim.
   - Shared low-rank basis across heads (multi-query-style consolidation).
   - Project K, V to a smaller subspace, store coefficients only.

3. **Token eviction / sparsification**
   - Keep only top-k tokens by attention weight (StreamingLLM style).
   - Sliding window of recent tokens + sink tokens.
   - Token merging (combine similar K, V across positions).

4. **Head pruning / sharing**
   - Drop entire heads' K, V (relies on remaining heads).
   - Share K (or V) across head groups beyond GQA.

5. **Hybrid representations**
   - Store keys at higher precision than values (or vice versa).
   - Recent tokens uncompressed, older tokens aggressively compressed.

6. **Static / training-free learned compression**
   - Random projections (no learning needed, model is frozen).
   - Hashing-based compression.

## The first run

Your very first run should establish the baseline by leaving the compressor
unchanged (identity). The TSV will record `compression_ratio = 1.0` and
`compression_score = 1.0`. Every subsequent run aims to beat that.

## Output format

```
---
val_bpb:                0.997900
baseline_bpb:           0.997900
compressed_bpb:         1.001200
val_bpb_delta:          0.003300
baseline_bytes_per_tok: 768.00
compressed_bytes_per_tok: 192.00
compression_ratio:      4.0000
compressor_name:        int4_per_channel
compression_score:      3.967000
...
```

Extract metrics with:
```bash
grep -E "^compression_score:|^compression_ratio:|^val_bpb_delta:|^compressed_bpb:" run.log
```

## Logging results

Tab-separated. Header:

```
commit	compression_score	compression_ratio	val_bpb_delta	compressed_bpb	status	description
```

1. git commit hash (short, 7 chars)
2. compression_score (e.g. 3.967000) — use 0.0 for crashes
3. compression_ratio (e.g. 4.0000) — use 0.0 for crashes
4. val_bpb_delta (e.g. 0.003300) — use 0.0 for crashes
5. compressed_bpb (e.g. 1.001200) — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of the compressor tried

Example:

```
commit	compression_score	compression_ratio	val_bpb_delta	compressed_bpb	status	description
a1b2c3d	1.000000	1.0000	0.000000	0.997900	keep	identity baseline
b2c3d4e	3.967000	4.0000	0.003300	1.001200	keep	int4 per-channel quantization
c3d4e5f	0.500000	2.0000	0.150000	1.147900	discard	rank-4 SVD (too lossy)
d4e5f6g	0.000000	0.0	0.000000	0.000000	crash	int1 quant (NaN in attention)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr29-kvcompress`,
or `autoresearch/<date>-kvcompress` more generally — the `-kvcompress`
suffix disambiguates this project from sibling autoresearch projects in
adjacent repos that may share the same date tag).

LOOP FOREVER:

1. Look at the git state: current branch/commit.
2. Modify `KVCompressor` (or swap `agent_compressor`) with a new compression idea.
3. `git commit` the change with a short, descriptive message.
4. Run: `uv run train.py > run.log 2>&1` (do NOT pipe to stdout/tee — keep
   logs out of your context).
5. Read out the metrics:
   `grep -E "^compression_score:|^compression_ratio:|^val_bpb_delta:|^compressed_bpb:" run.log`
6. If the grep is empty, the run crashed. `tail -n 50 run.log` for the trace.
   If it's a small fix, fix and retry. If the idea is fundamentally broken,
   record a `crash` row and move on.
7. Append a row to `results.tsv`.
8. If `compression_score` improved (higher), keep the commit (advance branch).
9. If equal or worse, `git reset --hard` to where you started.

**Timeout**: Each experiment should take ~5 minutes training + ~1-2 minutes
eval (two passes — baseline + compressed). If a run exceeds 12 minutes total,
kill it and treat as a failure.

**Crashes**: shape mismatches, NaNs from too-aggressive quantization, OOM
from blowing up the cache representation — all common. Use judgment: small
fixable bug → fix and retry. Idea is fundamentally broken → record crash,
revert, move on.

**NEVER STOP, NEVER ASK FOR PERMISSION**: Once the loop has begun, do NOT
pause to ask "should I continue?" or "is this a good stopping point?" or
"do you want me to try X?" or "shall I proceed?" or "is this acceptable?".
The agent is fully autonomous, running unattended on a remote GPU box
(typically a RunPod RTX 4090). The human is asleep / not at the keyboard.

Specifically:
- Do NOT ask "should I keep going?" — yes, always.
- Do NOT ask "should I try X next?" — just try it.
- Do NOT ask "what compression direction would you like to focus on?" — pick
  from the compression-directions list and run it.
- Do NOT ask for confirmation before `git commit`, `git reset --hard`, or
  destructive operations on the experiment branch — they're expected.
- Do NOT pause to tell the human a long summary mid-loop — write findings
  into paper.md and results.tsv and keep iterating.

If you genuinely run out of ideas, do NOT stop — re-read the
"Compression directions" list, look at near-miss experiments and try
combinations, attempt a more radical approach (learned hashing, attention-
score-driven eviction, layer-wise mixed precision, KV-key-only compression
with V-full, etc.). The loop runs until the human stops it, period.

### Environment notes

- The default substrate model is auto-selected by `train.py` based on the
  detected device. On CUDA (RunPod) it uses a 6-layer / 384-dim / batch=16
  setup (~50M params, fits in 24 GB). On Apple Silicon it falls back to a
  smaller 3-layer / 192-dim / batch=4 setup. You should NOT change the
  device-auto-scaling logic itself; it's part of the framework, not the
  research question.
- Wall time per experiment on a 4090 is roughly 5 min (training) + 30 s
  (eval, two passes) ≈ 6 min. Plan ~10 experiments / hour, ~80 / overnight.
- If `uv run train.py` exits non-zero, treat it as a `crash` row in the TSV
  and revert. Common causes: NaN in attention from too-aggressive
  quantization (try a less aggressive variant), shape mismatch in the
  compressor (fix the bug, retry), or OOM (reduce DEVICE_BATCH_SIZE inline
  but commit it as part of that experiment).

If you run out of ideas: re-read the "Compression directions" list above,
look at near-miss experiments and try combining them, attempt a more radical
approach (e.g. learned hashing), or try aggressive parameters of an idea
that already worked.

As a guideline: 5-min training + ~1-min eval → ~10/hour → ~80 experiments
overnight. The user wakes up to a `results.tsv` full of compression methods
ranked by `compression_score`, with the current branch HEAD pointing to the
best one found.
