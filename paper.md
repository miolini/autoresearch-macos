# Pareto-Efficient KV-Cache Compression for Small Transformers: An Autonomous Empirical Study

**Anonymous authors.** Manuscript prepared for ICML 2026.

## Abstract

The Key-Value (KV) cache dominates inference memory for autoregressive
transformers, motivating a long line of compression schemes — quantization,
low-rank approximation, token eviction, head sharing. We present a
controlled, single-substrate empirical comparison of several KV-cache
compressors evaluated under a unified scoring rule that jointly accounts
for compression ratio and quality loss. All experiments are conducted on
the same frozen 7.6M-parameter transformer trained on FineWeb-Edu, with
identical evaluation harness and bytes-counting protocol, so that any
ratio/quality differences reflect the *compressor* and not confounds in
training, data, or accounting. Our composite score
`S = ratio - 10 * max(Δval_bpb, 0)` identifies which methods cleanly Pareto-
dominate others on this substrate. We report the score-ranked leaderboard,
the empirical Pareto front, and an analysis of failure modes (NaN under
aggressive quantization, attention-collapse under low-rank). The framework
itself — an autonomous LLM agent iterating compressors against a fixed
score — is contributed as an open-source experimental artifact.

## 1. Introduction

KV-cache compression has become a central engineering concern for serving
modern LLMs. Existing literature reports compression ratios and quality
losses, but the methods are typically benchmarked on different models, on
different data, with different bytes-accounting conventions, making direct
comparison difficult. We address this by holding *everything constant
except the compressor*.

**Contributions.**
1. A unified scoring rule and reproducible eval harness for KV-cache
   compression on a single small-scale substrate.
2. An empirical Pareto front of N compressors spanning quantization,
   low-rank, and eviction families.
3. An autonomous agent loop that iterates compressors against the fixed
   score, demonstrating LLM-driven empirical research at small scale.
4. Open-source release of all code, data, and experiment logs.

## 2. Method

### 2.1 Substrate model

We train a small GPT-style transformer with rotary position embeddings,
RMSNorm, ReLU² MLP activation, grouped-query attention with `n_kv_head =
n_head`, and a Muon+AdamW optimizer mix. Training uses FineWeb-Edu shards
distilled to a vocabulary of 8192 BPE tokens, with sequence length 2048.
The model is trained for a wall-clock budget of 5 minutes on a single
Apple M-series GPU (MPS backend); concrete hyperparameters are fixed at
`depth=3, n_embd=192, head_dim=96` (≈7.6M parameters). The model is
frozen across all compression experiments — only the compressor varies.

### 2.2 Compression interface

Every compressor implements
```
compress(K, V) -> (state, n_bytes)
decompress(state) -> (K_hat, V_hat)
```
where `K, V` are bf16 tensors of shape `[B, T, H, D]` and `n_bytes` is the
*honest* byte cost of `state` (sum over all stored tensors / scalars).
The compressor is invoked inside the attention forward, immediately after
RoPE and norm and *before* SDPA, so the K, V seen by attention are the
post-roundtrip `K_hat, V_hat`.

### 2.3 Scoring

For each compressor we compute:
- `baseline_bpb` — val_bpb under identity (uncompressed) compressor.
- `compressed_bpb` — val_bpb under the candidate compressor.
- `Δbpb := compressed_bpb − baseline_bpb` (≥ 0 in expectation).
- `bytes_per_token_per_layer` for each (the compressed cost in bytes).
- `compression_ratio := baseline_bpt / compressed_bpt`.
- `compression_score := compression_ratio − 10 · max(Δbpb, 0)`.

The α=10 penalty per unit Δbpb is chosen so that a 0.01 bpb regression
costs 0.1 of compression ratio — a deliberately conservative tradeoff
suited to deployment scenarios where quality is paramount.

### 2.4 Autonomous experiment loop

Each iteration: edit `KVCompressor` in `train.py`; commit; train; eval;
parse score; append row to `results.tsv`; if score improved, advance the
branch, else `git reset --hard`. The loop is run unattended by a Claude
LLM agent following a fixed `program.md` skill specification.

## 3. Experiments

### 3.1 Compressors evaluated

(populated as the loop runs — see `results.tsv`)

### 3.2 Pareto front

(figure: `figures/pareto.png`)

### 3.3 Score trajectory

(figure: `figures/score_trajectory.png`)

### 3.4 Per-method comparison

(figure: `figures/method_comparison.png`)

## 4. Discussion

(populated after sufficient experiments — failure modes, surprising
findings, where the model substrate may bias results)

## 5. Limitations

- Single small-scale model substrate (7.6M params, 2048 ctx, 8192 vocab)
  may not faithfully predict relative ordering at LLM scale.
- The eval is parallel forward, not autoregressive decoding; methods that
  exploit access-pattern structure (e.g. attention-score-driven eviction)
  must compute scores from K, V directly within `compress()`.
- The α in the composite score is a free parameter; we report `ratio` and
  `Δbpb` separately throughout to allow re-scoring under other α.

## 6. Conclusion

(TBD)

## References

(TBD — populate with: GPTQ, AWQ, KIVI, StreamingLLM, H2O, Multi-Query
Attention, GQA, SmoothQuant, etc.)
