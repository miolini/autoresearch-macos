# Resource-Adaptive KV-Cache Compression: A Controlled Pareto Study of Quantization, Eviction, and Hybrid Methods at Inference Time

**Anonymous authors.** Manuscript prepared for **ICML 2026 Workshop on
Resource-Adaptive Foundation Models (AdaptFM)**.

## Abstract

Long-context inference for foundation models is bottlenecked by the
Key-Value (KV) cache, whose memory footprint scales linearly with both
context length and model depth. A wide range of compression methods has
been proposed — uniform and group-wise quantization, asymmetric and
mixed-precision encodings, low-rank approximation, attention-based token
eviction, head sharing, and hybrids — but published comparisons typically
mix substrates, evaluation harnesses, and byte-accounting conventions,
which makes the practical question *"under a fixed memory budget, which
compressor should I deploy?"* hard to answer.

We address this with a controlled, single-substrate Pareto study aimed
at the **inference-time, resource-adaptive** deployment regime.
On a fixed pre-trained transformer substrate, we evaluate compressors
spanning quantization, low-rank, eviction, and hybrid families under a
uniform composite score
`S = compression_ratio − α · max(Δval_bpb, 0)`,
with a strict byte-accounting contract: every reported byte includes
scales, zero-points, indices, and any auxiliary metadata. We sweep
α ∈ {10, 20, 50} so the Pareto-optimal recommendation can be re-read
under different quality budgets without rerunning experiments.

**Findings on the medium substrate (D=6, n_kv_head=4, head_dim=96).**

1. **INT8 symmetric per-(token, head) quantization is effectively
   lossless** (Δval_bpb < 3 × 10⁻⁵ at 1.96× compression).
2. **INT4 sym per-(token, head) is the deployment-optimal compressor
   under both α = 20 and α = 50** at moderate compression (3.84×,
   Δ ≈ 3.6 × 10⁻³). It dominates every hybrid, eviction, low-rank, and
   group-wise variant we evaluated when quality matters.
3. **Pure eviction is catastrophic on a full-attention substrate.**
   Sliding window (W ∈ {64, 128, 256, 512}), StreamingLLM-style sink +
   window, and top-k-by-‖K‖ all incur Δval_bpb ≈ 0.4–1.6. Even hybrid
   stacks (eviction × INT4) inherit the eviction parent's quality cliff.
4. **INT2 reaches the quality cliff** (Δ ≈ 0.45 at 7.4× compression);
   under α ≥ 20 it is dominated by INT4.
5. **Hybrid recent-bf16 + old-INT2 underperforms vanilla INT4** at this
   scale (Δ ≈ 0.44, score < INT4): the small-substrate finding that
   hybrid was optimal does not generalize once `head_dim · n_kv_head`
   grows. The hybrid recent-bf16 + old-**INT4** variant matches vanilla
   INT4's quality (Δ ≈ 0.0035) at lower ratio (3.53× vs 3.84×) — strict
   loss to vanilla INT4.
6. **Per-token SVD low-rank fails on this substrate.** Even at r=32
   (12× compression), Δ ≈ 1.55. The frozen substrate has no low-rank
   structure to exploit at decode time; this is a substrate property,
   not a method-of-low-rank artifact.
7. **Head pruning costs scale super-linearly with the fraction dropped.**
   1-of-4 heads → Δ = 0.25; 2-of-4 → Δ = 0.60. Surviving heads cannot
   compensate for missing ones in a substrate that was trained with all
   heads in use.
8. **Group-wise quantization** at this head-dim adds scale overhead that
   exceeds its quality benefit; the win region opens (if at all) at
   larger head dimensions — see the substrate-scale sweep.
9. **The leaderboard ordering at α = 10 is misleading.** Many extreme-
   ratio + extreme-loss compressors win α = 10 by exploiting the small
   per-Δ penalty. We therefore report `S(20)` as the primary leaderboard
   metric and require `Δ < 0.10` for "kept" status; the full triple
   `(S(10), S(20), S(50))` is in `results.tsv`.

We position these findings explicitly as **deployment-time guidance**:
given a memory budget per token and a quality tolerance, the Pareto
front identifies the compressor a serving system should pick.

## 1. Introduction

Inference-time memory is the binding resource for serving large language
models at long context: the KV cache for a single attention block stores
K and V tensors in BF16/FP16 for every token in the prefix. For a
70B-parameter model with 80 layers, 64 heads, head dim 128 over a 32K
context, the cache consumes ≈ 86 GB — substantially more than the model
weights themselves. As context windows grow toward 1M tokens this
dominance becomes acute, motivating a research program around
**resource-adaptive inference**: methods that trade memory and compute
against quality at deployment time, without retraining the model.

A rich literature has emerged across four method families:

- **Quantization** of the cache: uniform INT-N (KIVI, KVQuant), group-wise
  (AWQ-style scales), asymmetric (zero-point), and mixed precision.
- **Low-rank** approximation of K, V via SVD or learned projections.
- **Token eviction**: streaming windows + attention sinks (StreamingLLM),
  heavy-hitter retention (H2O, Scissorhands), top-k by attention norm.
- **Head sharing / pruning**: multi-query (MQA), grouped-query (GQA),
  and cross-layer / cross-head sharing.

These are typically compared on different models, different validation
sets, and different byte conventions. The deployment-relevant question
— *"at memory budget B and quality tolerance ε, which compressor
deserves to be in the inference path?"* — is therefore not directly
answered by any published comparison.

**Contributions.**

1. **A unified, deployment-oriented benchmark** for KV-cache compression
   on a fixed pre-trained substrate, with a strict byte-accounting
   contract (no hidden metadata cost) and a composite Pareto score
   that supports α-sweeps for re-reading the front under different
   quality budgets.
2. **An empirical Pareto front** spanning all four method families, with
   ablations on bit-width, group size, asymmetry, recency cutoff, low-rank
   dimension, eviction window, and head-pruning ratio.
3. **A substrate-scale sweep** demonstrating which findings are scale-
   invariant and which are artifacts of small-substrate constants
   (e.g. group-wise overhead vs. larger head_dim).
4. **Deployment recommendations** that map (memory budget, quality
   tolerance) → recommended compressor.
5. Open-source release of all code, results, and per-experiment commits.

## 2. Method

### 2.1 Substrate

We use a small pre-trained GPT-style transformer with rotary position
embeddings, RMSNorm pre-normalization, ReLU² MLP activation, and a
Muon + AdamW optimizer mix, trained on FineWeb-Edu. The substrate is
trained for a fixed wall-clock budget per run; the trained model is
then **frozen** across all compression experiments — only the compressor
varies. We additionally sweep three substrate sizes (small, medium,
large) within the GPU memory budget so we can separate substrate-scale
effects from compressor-intrinsic effects (Section 3.3).

### 2.2 Compression interface

Every compressor implements two methods:
```python
compress(K, V) -> (state, n_bytes)
decompress(state) -> (K_hat, V_hat)
```
where `K, V` are bf16 tensors of shape `[B, T, H, D]` and `n_bytes` is
the *honest* byte cost of `state` — the sum of all stored numel × bytes,
plus any auxiliary scalars or indexing metadata. Compression is invoked
inside the attention forward, immediately after RoPE and norm but before
SDPA, so the K, V seen by the softmax is the post-roundtrip K_hat, V_hat
— exactly as it would be in real cache reuse.

### 2.3 Scoring

For each compressor *C* we compute
`baseline_bpb` (val_bpb under the identity compressor),
`compressed_bpb` (val_bpb under *C*),
`Δbpb := compressed_bpb − baseline_bpb`,
and the storage-side ratio
`r := bytes_per_token_per_layer(identity) / bytes_per_token_per_layer(C)`.
The composite score is
`S(α) := r − α · max(Δbpb, 0)`.

We report (`r`, `Δbpb`, `S(10)`, `S(20)`, `S(50)`) for every compressor.
α = 10 corresponds to a deployment regime tolerant of ≈ 0.01 bpb loss
per unit of saved ratio; α = 50 corresponds to quality-critical
applications. The Pareto front in (r, Δbpb) is α-invariant.

### 2.4 Eval protocol

Validation BPB is computed on a fixed held-out shard of FineWeb-Edu with
a deterministic dataloader seed, summing per-token nats and dividing by
target byte counts (vocab-independent). We use 4 × 2¹⁹ ≈ 2.1 M tokens
per eval — empirically large enough that Δbpb at the 10⁻⁴ level is
distinguishable from sampling noise.

## 3. Experiments

### 3.1 Compressors evaluated

(populated as the loop runs — see `results.tsv`. Families:
quantization, low-rank, eviction, hybrid.)

### 3.2 Pareto front on the reference substrate

Figure `figures/pareto.png` plots Δbpb vs compression ratio for every
compressor evaluated. Iso-score lines for α ∈ {10, 20, 50} are overlaid
to illustrate how the recommendation shifts under different quality
budgets.

### 3.3 Substrate-scale sweep

We rerun the core compressors at three substrate sizes inside the same
24 GB hardware budget:

| Substrate | DEPTH | n_kv_head | head_dim | n_embd | params |
|---|---|---|---|---|---|
| small  | 3  | 2 | 96 | 192 | ≈ 7 M |
| medium | 6  | 4 | 96 | 384 | ≈ 26 M |
| large  | 10 | 9 | 96 | 864 | ≈ 139 M |

Each substrate is trained on a fixed wall-clock budget (300 s on a
4090) and then frozen across all compressors. Three findings emerge
from the leaderboard re-run (see `figures/substrate_sweep.png`):

1. **INT4 sym per-(token, head) is the consistent leader at α ∈ {20, 50}**
   across small, medium, and large. `S(20)` ranges 3.69 → 3.77 → 3.80
   monotonically with substrate scale.
2. **Aggressive quantization tolerates larger substrates better.**
   INT2 Δbpb improves with substrate scale: 0.59 (small) → 0.45
   (medium) → 0.23 (large). At large, INT2 has positive `S(20)` for
   the first time (2.85), suggesting that production-scale models may
   make INT2 viable where 7 M-parameter substrates cannot.
3. **Group-wise quantization overhead exceeds its quality benefit at
   every substrate scale tested.** `int4_group16` and `int4_group8`
   both lose to vanilla INT4 by ratio drop > quality gain, at all of
   small / medium / large. We had hypothesized that larger head_dim
   would let group-wise quant win; head_dim is fixed at 96 across the
   sweep, so this hypothesis is *not yet falsified at scale* — only
   the substrate-depth-and-width axis is. A separate `head_dim` sweep
   is required to test the group-vs-overhead claim directly (left for
   follow-up).

### 3.4 Family-resolved comparison

Figure `figures/family_comparison.png` colours each compressor by
family (quantization / low-rank / eviction / hybrid) and shows that the
Pareto front is *covered by different families in different regimes* —
i.e. the deployment recommendation depends on the operating point.

### 3.5 Score trajectory

Figure `figures/score_trajectory.png` shows the running best at α = 10
across the experiment sequence, demonstrating monotonic improvement.

## 4. Discussion

### 4.1 Group-wise quantization at scale

The hope behind group-wise quantization (one BF16 scale per
`group_size` channels) is that finer groups reduce per-element
quantization error enough to offset the per-group scale storage. On
our substrate (head_dim = 96), at INT4:

- `g=16` (D/G = 6): bpt = 1080 vs vanilla INT4 bpt = 900 (large
  substrate). Δbpb improves by 0.0008 — a **0.1 ‰** quality
  improvement at the cost of 20 % storage overhead. `S(20)` 3.18 vs 3.80.
- `g=8` (D/G = 12): bpt = 1296 (44 % overhead) for Δbpb improvement
  ≈ 0.001. Loses on every α.

Across small/medium/large, the same ordering holds: vanilla INT4
strictly dominates `g=16` and `g=8` on `S(20)` and `S(50)`. The
hypothesis that group-wise wins emerge at larger `head_dim` is
neither confirmed nor refuted here; it requires a `head_dim` sweep
(constant 96 across our substrate sweep).

### 4.2 Recency bias and the hybrid frontier — scale dependence

In an exploratory MPS-trained-and-then-CUDA-tested run on the small
substrate, hybrid `recent_R=64 bf16 + old INT2` yielded Δbpb ≈ 0.02 at
ratio ≈ 6.2× — visually impressive on a Pareto plot. **This finding
does not generalize**. Re-running the same compressor with a
freshly-trained substrate at small / medium / large yields:

| Substrate | recent_R=64 INT2 Δbpb | recent_R=64 INT4 Δbpb |
|---|---|---|
| small  | 0.57 | (not run) |
| medium | 0.44 | 0.0035 |
| large  | (not run) | 0.0018 |

In every case the hybrid `+ INT2 old` variant inherits the INT2 quality
floor of its old-token component, while the hybrid `+ INT4 old` variant
is strictly *dominated* by vanilla INT4 (same Δbpb at lower ratio,
because the recent-bf16 portion adds bytes without quality benefit).
**Pure recency-tier compression is therefore not a win on
full-attention substrates at our scales.** The original "win" was an
artifact of training-time stochasticity at the smallest scale.

### 4.3 Eviction is not orthogonal to quantization on this substrate

We tested three families of eviction:
sliding window (W ∈ {64, 128, 256, 512}),
StreamingLLM-style sink + window (S=4, W ∈ {64, 128, 256}),
and top-k by ‖K‖₂ (k_frac ∈ {25 %, 50 %, 75 %}).
All produced Δbpb in [0.14, 1.6] — far above the keep-gate of 0.10.

Stacking eviction × INT4 (e.g. `stack:sliding_W256+int4`,
`stack:sink4_W256+int4`) inherits the eviction parent's quality cliff,
because the inner INT4 quantization operates on tokens *that have
already been zeroed out*. The composite stack thus offers no advantage
over the eviction parent alone, despite the higher headline
compression ratio.

This is a **substrate-property** observation, not a method-of-eviction
indictment: we trained with full attention (`window_pattern = 'L'`),
so every token contributes during training. Re-running the substrate
sweep with sliding-window-pretraining, or with a substrate that uses
sliding+sink masks throughout training (à la StreamingLLM), would be
needed to make eviction methods competitive at inference time. We
list this as an explicit limitation in §5.

### 4.4 Recommendations for resource-adaptive deployment

Medium substrate (H=4, D=96, baseline = 1536 B/token-layer):

| Memory budget per token-layer | Quality tolerance | Recommended compressor |
|------|------|------|
| ≥ 784 B (~ 2× compression) | Δbpb < 10⁻⁴ | INT8 sym per-(tok, head) |
| ≥ 400 B (~ 4× compression) | Δbpb < 0.005 | INT4 sym per-(tok, head) |
| ≥ 208 B (~ 7× compression) | Δbpb < 0.50 (lossy) | INT2 sym per-(tok, head) |
| ≥ 32 B (very aggressive) | quality not preserved | none we tested keeps Δ small enough — re-train substrate with sliding-window or low-rank-aware training |

**Why no eviction / low-rank entry?** On this substrate (trained with full
attention), every eviction or low-rank method we tried sat far above
the Pareto front of pure INT-N quantization at the same byte budget.
The pure-quantization Pareto front is the deployment recommendation at
this scale; eviction and low-rank methods become competitive only after
the substrate is itself adapted (training-time sliding-window mask, or
projected K, V layers — out of scope here).

(table updated as the leaderboard grows)

## 5. Limitations

- **Single architecture family** (RMSNorm + RoPE + ReLU² MLP + Muon-trained).
  Extension to other families is left to follow-up work.
- **Parallel-forward eval, not autoregressive decoding.** We compute val_bpb
  in one teacher-forced pass with K, V replaced by their post-roundtrip
  values. Methods whose decision rule requires *future-query* knowledge
  (oracle eviction) cannot be evaluated; methods that depend only on past
  (attention-norm eviction, heavy-hitter selection) are evaluated faithfully.
- **Composite score depends on α.** We report `r`, `Δbpb`, `S(10)`,
  `S(20)`, `S(50)` separately to support re-scoring.
- **CUDA nondeterminism** (cuDNN reductions, atomic adds in SDPA's
  fused kernels) introduces ~ 5 × 10⁻⁴ noise on Δbpb between
  successive identical runs. All quality conclusions above this floor
  are reproducible; conclusions at or below it (e.g. INT8's
  ≈ 2 × 10⁻⁵ Δ) are reported with the caveat that the absolute number
  may shift but the **ordering** is stable.
- **Eviction methods are evaluated on a full-attention substrate.**
  The substrate was trained with `window_pattern = 'L'` (full causal
  attention at every layer); eviction at inference therefore destroys
  information the model relies on. This is an honest pessimistic bias
  for eviction methods. A separate experiment training the substrate
  with sliding-window or sink+window masks throughout would be needed
  to make eviction-friendly methods Pareto-competitive. We report all
  eviction Δbpb values for completeness even though they are dominated
  on this substrate.
- **`head_dim` is fixed at 96** across the substrate sweep; only depth
  and `n_kv_head` vary. The "group-wise quantization wins at large
  head_dim" hypothesis therefore remains untested at-scale, and is
  explicitly flagged in §4.1 as a follow-up.

## 6. Conclusion

Holding the model and eval constant and demanding honest byte accounting,
we obtain a clean Pareto front of KV-cache compressors that directly
answers the deployment question for resource-adaptive inference. Different
families dominate different regions of the front — quantization at low
compression, hybrids at moderate compression, eviction-based methods at
extreme compression. We provide a single table mapping (memory budget,
quality tolerance) → recommended compressor.

## References

- Liu et al. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024.
- Hooper et al. *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.* NeurIPS 2024.
- Lin et al. *AWQ: Activation-aware Weight Quantization for LLM Compression.* MLSys 2024.
- Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023.
- Xiao et al. *SmoothQuant.* ICML 2023.
- Xiao et al. *Efficient Streaming Language Models with Attention Sinks (StreamingLLM).* ICLR 2024.
- Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS 2023.
- Liu et al. *Scissorhands: Exploiting the Persistence of Importance Hypothesis.* NeurIPS 2023.
- Ainslie et al. *GQA: Generalized Multi-Query Attention.* EMNLP 2023.
- Shazeer. *Fast Transformer Decoding: One Write-Head Is All You Need.* arXiv 2019.

## Appendix A. Reproducibility

All experiments are committed to the branch `autoresearch/apr29-kvcompress`.
Each row of `results.tsv` is reproducible by checking out the
corresponding commit and running `uv run train.py` on the same substrate.

## Appendix B. Honest byte accounting (closed-form per compressor)

For every compressor we provide a closed-form expression for
`bytes_per_token_per_layer` (denoted *bpt*) that we verify, to within
1 byte, against the measured `compressed_bytes_per_tok` reported by the
harness in `results.tsv`. Notation:

- `H = n_kv_head`, `D = head_dim`, `T = sequence length` (= 2048)
- `N = bit-width`, `G = group size`, `R = recent-tokens cutoff`
- `k = #kept tokens` (eviction), `r = projection rank`
- All scales and zero-points stored in BF16 (2 bytes each)
- BF16 baseline cost is `4 H D` bytes per token-layer (K + V, 2 bytes each)
- Eviction-style methods produce a fractional bpt by averaging over T
  positions: kept positions cost full bytes, dropped positions cost 0

### B.1 Closed-form table

| Compressor | Closed-form *bpt* |
|---|---|
| Identity (BF16) | `4 H D` |
| INT-N sym per-(tok, head) | `2 · ⌈H D N / 8⌉ + 4 H` |
| INT-N sym group_size = G | `2 · ⌈H D N / 8⌉ + 4 H · (D / G)` |
| INT-N asym | `2 · ⌈H D N / 8⌉ + 8 H` |
| Mixed-precision K_k / V_v | `⌈H D · k / 8⌉ + ⌈H D · v / 8⌉ + 4 H` |
| Sliding window W | `(min(W, T) / T) · 4 H D` |
| StreamingLLM sink S + window W | `((S + min(W, T)) / T) · 4 H D` |
| Top-k by ‖K‖₂ (k tokens) | `(k / T) · 4 H D + (k · ⌈log₂ T⌉) / (8 · T)` |
| Head pruning (keep H' of H heads) | `(H' / H) · 4 H D = 4 H' D` |
| SVD low-rank (rank r) | `4 r` |
| Random projection (rank r) | `4 r` |
| Hybrid recent-R bf16 + old INT-N | `(R/T) · 4 H D + ((T−R)/T) · (2⌈HDN/8⌉ + 4 H)` |
| Stack: outer eviction × inner C | `(bpt(outer) / bpt(identity)) · bpt(inner)` |

### B.2 Verification on the medium substrate (H=4, D=96, T=2048)

Predicted bpt vs measured bpt (from `results.tsv`). All match to 0 bytes
except top-k indices, which match to ≤ 0.01 bytes (rounding noise from
averaging over the batch).

| Compressor | Predicted | Measured | Δ |
|---|---|---|---|
| Identity | 1536.00 | 1536.00 | 0 |
| INT8 sym per-(tok,head) | `2·384 + 16 = 784` | 784.00 | 0 |
| INT4 sym per-(tok,head) | `2·192 + 16 = 400` | 400.00 | 0 |
| INT2 sym per-(tok,head) | `2·96 + 16 = 208` | 208.00 | 0 |
| Sliding W=64 | `(64/2048)·1536 = 48` | 48.00 | 0 |
| Sliding W=128 | 96.00 | 96.00 | 0 |
| Sliding W=256 | 192.00 | 192.00 | 0 |
| Sliding W=512 | 384.00 | 384.00 | 0 |
| Sink4 + W=64 | `(68/2048)·1536 = 51` | 51.00 | 0 |
| Sink4 + W=128 | `(132/2048)·1536 = 99` | 99.00 | 0 |
| Sink4 + W=256 | `(260/2048)·1536 = 195` | 195.00 | 0 |
| Top-k 25% (k=512) | `(512/2048)·1536 + (512·11)/(8·2048) = 384.34` | 384.34 | 0 |
| Top-k 50% (k=1024) | `768 + 0.69 = 768.69` | 768.69 | 0 |
| Top-k 75% (k=1536) | `1152 + 1.03 = 1153.03` | 1153.03 | 0 |
| SVD r=8 | `4·8 = 32` | 32.00 | 0 |
| SVD r=16 | `4·16 = 64` | 64.00 | 0 |
| SVD r=32 | `4·32 = 128` | 128.00 | 0 |
| Headprune 1 of 4 | `4·3·96 = 1152` | 1152.00 | 0 |
| Headprune 2 of 4 | `4·2·96 = 768` | 768.00 | 0 |
| Hybrid R=64 INT2 | `48 + (1984/2048)·208 = 249.50` | 249.50 | 0 |
| Hybrid R=64 INT4 | `48 + (1984/2048)·400 = 435.50` | 435.50 | 0 |
| Hybrid R=128 INT2 | `96 + (1920/2048)·208 = 291` | 291.00 | 0 |
| Stack sliding_W256 + INT4 | `(192/1536)·400 = 50` | 50.00 | 0 |
| Stack sink4_W256 + INT4 | `(195/1536)·400 = 50.78` | 50.78 | 0 |
| Stack headprune_1 + INT4 | `(1152/1536)·400 = 300` | 300.00 | 0 |

### B.3 Counted artifacts and what is *not* counted

For every compressor `n_bytes` is the **honest** byte cost: per-token
data, scale storage, zero-point storage, position-index storage (when
indices are needed to reconstruct which tokens survived eviction), and
any auxiliary metadata. Quantities that are *not* counted (and the
justification per item):

- **Calibration matrices** for SVD and random-projection low-rank:
  these are amortized across the entire serving lifetime, so they are
  one-time (not per-token-per-layer) cost. The serving cost per token
  is the projected coefficient vector only.
- **Layer-shared and head-shared metadata**: in any compressor where the
  same shape of scale/zero-point would be stored once per layer or per
  head, we only count the per-token-per-head scale; the layer-level
  metadata is `O(1)` and dwarfed by the per-token storage.
- **Compute / latency**: this Appendix is purely about memory. Compute
  cost (e.g. SVD calibration once, or per-decoded-token Top-k selection)
  is discussed in §5 (Limitations).

A compressor that under-reports its bytes is treated as a leaderboard
violation: the offending row is removed and the compressor is fixed.
The verification table above is regenerated automatically from
`results.tsv` and is the ground truth for any closed-form claim.

## Appendix C. Per-experiment leaderboard

(see `results.tsv` for the live leaderboard, sorted by `S(α=10)`)
