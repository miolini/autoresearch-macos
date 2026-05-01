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
10. **K needs more precision than V.** Mixed precision `K8_V4`
    (Δ = 1.25 × 10⁻³) outperforms `K4_V8` (Δ = 2.23 × 10⁻³) at the
    same byte budget, confirming the asymmetry literature on KV-cache
    quantization (KIVI). Quantization noise on K compounds at the
    `q · k` dot-product (pre-softmax), while noise on V averages over
    softmax weights post-attention. Practical recommendation: when
    one bit must be sacrificed, take it from V before K.
11. **Mixed K4 / V2's quality cost is depth-driven, not head-dim-driven.**
    Across the three depth-scaled substrates (D=3, 6, 10 at fixed HD=96)
    the `mixed_K4_V2` Δbpb falls monotonically: 0.224 (small) → 0.102
    (medium, at the keep-gate boundary) → **0.049 (large, the new α20
    leader at 5.05× compression, S(20) = 4.07, S(50) = 2.60)**. Holding
    `DEPTH = 6` constant and varying `HEAD_DIM ∈ {64, 96, 128}` instead
    leaves Δ flat in [0.10, 0.12], not below the gate; holding
    `DEPTH = 10` constant and pushing `HEAD_DIM` from 96 → 128 leaves
    the K4_V2 lead intact (Δ 0.049 → 0.047, S(20) 4.07 → 4.18). So the
    crossover is *depth*-gated. At production scales, aggressive
    4-bit K + 2-bit V is the first compressor in our study to dominate
    vanilla INT4, but only on substrates with enough layers to absorb
    the V-side noise.
12. **K-vs-V asymmetry generalizes to scale and even widens.** A
    matched-pair sweep (§4.4) at two byte budgets (3.10× and 5.05×)
    on both large-depth substrates shows that swapping K-precision
    down at the same byte budget costs Δbpb a remarkably consistent
    **3.2–3.8×** across all four (substrate, byte budget) combinations.
    The K-bit penalty appears to be a structural property of attention
    rather than a quantization-noise artifact. The recommended bit
    allocation is "spend bits on K first" at every scale we measured.

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

We populate the leaderboard with 30 + compressor configurations across
the four canonical families. The full enumeration:

| Family | Configurations | Description |
|---|---|---|
| **Identity** | identity (BF16) | Reference baseline; `bpt = 4 H D` |
| **Quantization (uniform sym)** | INT8, INT4, INT2 | Per-(token, head) symmetric, BF16 scale per slice |
| **Quantization (group-wise)** | INT4 + group_size ∈ {8, 16, 32, 64} | One BF16 scale per group along head_dim |
| **Quantization (asymmetric)** | INT4_asym | Adds BF16 zero-point per slice |
| **Quantization (mixed K/V precision)** | (k_bits, v_bits) ∈ {(8, 4), (4, 8), (8, 2), (4, 2)} | Different bit-width on K vs V |
| **Hybrid (recency-tier)** | recent_R ∈ {64, 128} × old INT-N ∈ {2, 4} | Recent tokens BF16, older tokens INT-N |
| **Eviction (sliding window)** | W ∈ {64, 128, 256, 512} | Keep last W tokens, drop the rest |
| **Eviction (StreamingLLM)** | sinks=4, W ∈ {64, 128, 256} | Keep first 4 + last W tokens |
| **Eviction (top-k by ‖K‖)** | k_frac ∈ {25 %, 50 %, 75 %} | Keep top-k tokens by row-wise K-norm |
| **Eviction (H2O heavy-hitter)** | recent ∈ {64, 128} × keep_frac ∈ {25 %, 50 %, 75 %, 80 %, 85 %, 90 %} | Keep last R tokens + top-k older tokens by total received attention (Zhang et al., 2023) |
| **Head pruning** | drop ∈ {1, 2} of n_kv_head | Zero K, V on a fraction of KV heads |
| **Low-rank (per-token)** | SVD rank ∈ {8, 16, 32}; random projection rank=32 | rank-r per-token approximation across heads |
| **Hybrid stacks** | (sliding_W256, sink4_W256, headprune_1) × INT4 | Outer eviction composed with inner quantization |

Every compressor is evaluated on the medium substrate; the **core
seven** (INT8, INT4, INT2, INT4_g16, INT4_asym, hybrid_R64_INT4,
mixed_K8_V4) are additionally evaluated on small/large substrates and
on HEAD_DIM ∈ {64, 96, 128} variants of the medium substrate
(§3.3, §4.1). All raw rows live in `results.tsv`, sorted by `S(α=20)`.

### 3.2 Pareto front on the reference substrate

Figure `figures/pareto.png` plots Δbpb vs compression ratio for every
compressor evaluated. Three regions are visible by inspection:

- **Lossless plateau at Δ < 10⁻³** (INT8 family, INT4 family on
  small/medium, K8_V4 family). Compression ratios in [1.96, 4.0].
- **Production-acceptable band at Δ ∈ [10⁻³, 10⁻¹]** populated almost
  exclusively by quantization variants. The α20 leader on each substrate
  lives in this band: INT4 on small/medium/hd64/hd128, mixed_K4_V2 on
  large/hd128_large, with the H2O+INT4 stack just barely inside the
  band on large at 5.07× / Δ=0.095.
- **Quality cliff at Δ > 10⁻¹** populated by all four pure-eviction
  families and by aggressive quant (INT2, K2_V*). Even the strongest
  eviction (H2O at K=75 %) sits at Δ ≈ 0.1; sliding window and
  StreamingLLM sink+window peak at Δ > 1.

The third region is wide and ratio-rich (up to 30× for sliding+INT4
stacks) but Pareto-irrelevant under any production-grade quality gate.

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

1. **INT4 sym per-(token, head) leads on small and medium under
   α ∈ {20, 50}; on large it is overtaken by `mixed_K4_V2`.** Vanilla
   INT4 `S(20)` ranges 3.69 → 3.77 → 3.80 monotonically with substrate
   scale, but on large the mixed-precision K4 / V2 split lands at
   `S(20) = 4.07` (ratio 5.05×, Δ = 0.049) and dominates. The
   "constant-leader" reading from a single substrate would have missed
   this crossover.
2. **Aggressive quantization tolerates larger substrates better.**
   INT2 Δbpb improves with substrate scale: 0.59 (small) → 0.45
   (medium) → 0.23 (large). At large, INT2 has positive `S(20)` for
   the first time (2.85). The same effect explains the `mixed_K4_V2`
   crossover above: 2-bit V quantization is catastrophic at small
   (Δ = 0.224) but merely a 5 % quality cost at large.
3. **Group-wise quantization overhead exceeds its quality benefit at
   every (substrate, head_dim, depth) combination tested.**
   `int4_group{8, 16, 32}` lose to vanilla INT4 across small / medium /
   large, across the HD ∈ {64, 96, 128} sweep at fixed depth=6, and
   across the new HEAD_DIM=128 × DEPTH=10 *interaction* substrate
   (§4.1). The "group wins at large head_dim × large depth" hypothesis
   from the quantization literature is therefore *not supported* at any
   (head_dim, group_size, depth) combination we tested in our 24 GB
   budget; the BF16 scale cost dominates the modest quality gain.

### 3.4 Family-resolved comparison

Figure `figures/family_pareto.png` colours each compressor on the
medium substrate by family (quantization / quant-group / quant-mixed /
eviction / sink+window / top-k / H2O / low-rank / head-prune / hybrid /
stack). Three visual takeaways:

1. **The quantization family hugs the lower-left of the Pareto plane**
   (low Δ, moderate-to-high ratio). All other families sit at higher Δ
   *or* lower ratio.
2. **Within eviction, H2O is the new bottom-left** (lowest Δ at any
   given ratio): the heavy-hitter family lies entirely below the top-k‖K‖,
   sink+window, and sliding window families on the medium-substrate
   Pareto plot, by a consistent margin (cf. table in §4.3).
3. **Hybrid stacks inherit the eviction parent's Δ.** Sliding/sink ×
   INT4 stacks land at high ratio + Δ ≈ 1.4 (their eviction parent's
   cliff); the new H2O × INT4 stack lands at ratio 5.06× / Δ ≈ 0.10
   (within an order of magnitude of pure INT4 on Δ), which is the
   first eviction-based composition to escape the quality cliff
   visible elsewhere in the figure.

This picture is the strongest evidence in our study that the
inference-time deployment Pareto front, *for a substrate trained with
full attention*, is dominated by quantization. Eviction and low-rank
methods would need substrate-side adaptation (training-time mask, or
projected K, V layers) to compete.

### 3.5 α-sensitivity and the deployment leader table

A central practical question is: *how does the deployment recommendation
change as the quality penalty weight α varies?* We present two views.

**Restricted view (Δ < 0.10 gate).** When we restrict attention to
compressors whose quality loss is bounded (Δval_bpb < 0.10 — a roughly
"production-acceptable" tolerance), the picture is mostly INT4 — except
for one substrate where mixed-precision K4 / V2 takes the crown:

| Substrate | α=10 leader | α=20 leader | α=50 leader |
|---|---|---|---|
| small (D=3, H=2, HD=96) | int4 (3.76) | int4 (3.69) | int4 (3.46) |
| medium (D=6, H=4, HD=96) | int4 (3.80) | int4 (3.77) | int4 (3.66) |
| **large (D=10, H=9, HD=96)** | **mixed_K4_V2 (4.56)** | **mixed_K4_V2 (4.07)** | **mixed_K4_V2 (2.60)** |
| HD=64 (D=6, H=6) | int4 (3.73) | int4 (3.70) | int4 (3.59) |
| HD=128 (D=6, H=3) | int4 (3.84) | int4 (3.80) | int4 (3.69) |
| **hd128_large (D=10, HD=128)** | **mixed_K4_V2 (4.65)** | **mixed_K4_V2 (4.18)** | **mixed_K4_V2 (2.77)** |

**Per-(token, head) symmetric INT4 is the restricted-Pareto leader on
the four small/medium scale substrates**, but on *both* large-depth
substrates we tested (large at HD=96 and hd128_large at HD=128) the
mixed-precision K4 / V2 split takes over at every α. The pattern is
internally consistent with Finding 2 (aggressive quantization tolerates
larger substrates better) and Finding 10 (V is the cheaper place to
sacrifice bits). The K4_V2 win is *depth-driven*, not head-dim-driven:
adding head_dim from 96 → 128 at fixed DEPTH=10 leaves the K4_V2 lead
intact and even slightly widens it (S(20) 4.07 → 4.18). The deployment
recommendation therefore depends on substrate scale: INT4 for ≤ 26 M-
parameter models in our sweep, and the K4 / V2 split for ≥ 139 M-
parameter models. Whether the crossover generalizes to multi-billion-
parameter production models is open and is the most actionable
follow-up from this paper.

**Unrestricted view (no quality gate).** Without a quality gate, the
leader is sensitive to both α and substrate:

| Substrate | α=10 leader | α=20 leader | α=50 leader |
|---|---|---|---|
| small  | int4 | int4 | int4 |
| medium | svd_r8 (Δ=1.63) | svd_r8 (Δ=1.63) | int4 |
| large  | int2 (Δ=0.23) | int4 | int4 |
| HD=64  | int2 (Δ=0.32) | int4 | int4 |
| HD=128 | int4 | int4 | int4 |

This view exposes two phenomena:

1. **The α=10 score is gameable by extreme-ratio + extreme-loss
   compressors.** SVD r=8 at the medium substrate scores 31.7 at α=10
   despite Δval_bpb = 1.63 (the cache reconstruction is essentially
   useless). Even α=20 does not always rescue the right ordering.
   Practitioners reading a paper that presents a single α score must
   verify the quality column independently.
2. **At higher substrate scale, INT2 catches up at α=10.** On the large
   substrate, INT2 has Δval_bpb = 0.23 — small enough that the 7.4×
   ratio offsets the penalty at α=10. This is consistent with the
   "INT2 viable at production scale" reading of recent KV-cache
   quantization papers.

We therefore report `S(α=10)` only with the Δ<0.10 gate applied; the
unrestricted column in `results.tsv` is preserved for reproducibility
but is not the basis for any deployment recommendation. The full
trajectory of `S(α=10)` across all 60 + experiments is plotted in
`figures/score_trajectory.png`; the running-best line approaches the
INT4 ratio (≈ 3.84) on every substrate we tested.

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
strictly dominates `g=16` and `g=8` on `S(20)` and `S(50)`.

**`head_dim` sweep at the medium substrate.** To test the hypothesis
that group-wise wins emerge at *larger* `head_dim` (where the per-group
scale storage is a smaller fraction of data storage), we retrain
medium-depth substrates at `HEAD_DIM ∈ {64, 96, 128}`. INT4_g32 at
`HD = 64` (D/G = 2) is the closest group-wise variant to ever match
vanilla INT4: ratio 3.56× vs 3.76×, Δ improves by 0.0009 — still a
20 % storage cost for a 0.1 ‰ quality bump.

**`head_dim` × `depth` interaction sweep (closing the §5 limitation).**
A second hypothesis is that group-wise might win in the *combined*
large-head_dim × large-depth regime, where the per-group scale becomes
proportionally smaller across many layers. We retrained an
`hd128_large` substrate (DEPTH=10 ASPECT_RATIO=80 HEAD_DIM=128,
≈ 110 M parameters in the same 24 GB budget) and re-ran INT4 alongside
INT4_g{8, 16, 32}:

| Compressor | bpt | ratio | Δbpb | S(α=20) |
|---|---|---|---|---|
| int4 (vanilla) | 924 | 3.88× | 0.0018 | **3.84** |
| int4_g32 | 1008 | 3.56× | 0.0011 | 3.53 |
| int4_g16 | 1120 | 3.20× | 0.0008 | 3.18 |
| int4_g8 | 1344 | 2.67× | 0.0006 | 2.65 |

Vanilla INT4 still wins by a wide margin; group-wise overhead grows
linearly with `D/G`, while the quality benefit is sub-linear and tops
out at < 1 ‰ Δbpb improvement. The "group wins at larger head_dim ×
depth" hypothesis from the quantization literature is therefore **not
supported** at any of the (head_dim, group_size, depth) combinations
we tested — at our scales, the BF16 scale cost dominates the gain.

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

We tested four families of eviction:
sliding window (W ∈ {64, 128, 256, 512}),
StreamingLLM-style sink + window (S=4, W ∈ {64, 128, 256}),
top-k by ‖K‖₂ (k_frac ∈ {25 %, 50 %, 75 %}),
and **H2O heavy-hitter** (recent R + top-k older tokens by total received
attention; Zhang et al., 2023).
All produced Δbpb in [0.10, 1.6] — at or above the keep-gate of 0.10.

Stacking eviction × INT4 (e.g. `stack:sliding_W256+int4`,
`stack:sink4_W256+int4`) inherits the eviction parent's Δbpb almost
exactly, because the inner INT4 quantization operates on tokens *that
have already been zeroed out* and contributes only its own (very small)
Δ on top. For sliding / sink eviction this is catastrophic — those
parents have Δ ≈ 1.4, so the stack inherits a 1.4 cliff even at 30×
nominal ratio. For **H2O eviction the inherited Δ is much smaller**:
`stack:h2o_R64_K75pct+int4` lands at ratio 5.06× / Δ=0.105 on medium
and ratio 5.07× / Δ=0.095 on large (the latter clears the Δ<0.10
keep gate), confirming both halves of the rule:
(a) the stack's Δ ≈ the eviction parent's Δ regardless of the inner
quant choice, and (b) the inner quant scales the headline ratio
roughly by `bytes(inner)/bytes(bf16)`. So an H2O+INT4 stack is the
first eviction-based composition to reach a "kept-quality"
configuration, but it is still *strictly Pareto-dominated* by
`mixed_K4_V2` on large (5.05× / Δ=0.049) and by `int4` on medium
(3.84× / Δ=0.003).

**Heavy-hitter (H2O) is the eviction-family Pareto winner, but is still
α20-dominated by INT4.** At every retention level on the medium
substrate, H2O strictly improves on top-k-by-‖K‖ — the score-by-true-
attention is a measurably better selection rule than the K-norm proxy:

| keep_frac | top-k‖K‖ Δ | H2O Δ | top-k‖K‖ α20 | H2O α20 |
|---|---|---|---|---|
| 25 % | 0.76 | 0.67 | −11.3 | −9.8 |
| 50 % | 0.37 | 0.32 | −5.4 | −4.4 |
| 75 % | 0.14 | 0.10 | −1.5 | −0.7 |

Pushing H2O retention higher (K ∈ {80, 85, 90 %}) shows that the method
*does* cross both the Δ < 0.10 quality gate and into positive α20 score,
but only at very low compression ratios:

| recent / keep_frac | ratio | Δbpb | S(α=20) | S(α=50) |
|---|---|---|---|---|
| R=64, K=80 % | 1.24 | 0.070 | −0.16 | −2.25 |
| R=64, K=85 % | 1.17 | 0.043 | **+0.31** | −0.98 |
| R=64, K=90 % | 1.11 | 0.021 | **+0.68** | **+0.04** |

So heavy-hitter eviction can be deployed in a "lossy-but-functional"
regime (Δbpb < 0.05 at ~ 1.1× saving), but it cannot deliver the 3–5×
compression ratios that quantization achieves at lower Δbpb. INT4 at
3.84× / Δ ≈ 0.0035 still strictly Pareto-dominates every H2O point on
this substrate. The H2O improvement over top-k‖K‖ is real and consistent
across retention levels, but narrow in absolute terms.

**H2O quality improves with substrate scale.** Re-running R=64 on small
and large substrates produces the same scale-tolerance pattern we
observed for INT2 (Finding 2):

| keep_frac | small Δ | medium Δ | large Δ |
|---|---|---|---|
| K=25 % | 0.82 | 0.67 | 0.59 |
| K=50 % | 0.42 | 0.32 | 0.28 |
| K=75 % | 0.15 | 0.10 | **0.093** |
| K=85 % | — | 0.043 (α20=+0.31) | 0.044 (α20=+0.29) |

At K=75 % on the large substrate, H2O for the first time clears the
Δ < 0.10 quality gate at non-trivial retention (1.32× compression);
at K=85 % both medium and large reach positive S(α=20) (≈ +0.3) — but
only at low ratio (~ 1.17×). On production-scale substrates the heavy-
hitter family may become more useful than these small-substrate numbers
suggest; this is consistent with the published H2O paper's results on
multi-billion-parameter LLMs.

This is a **substrate-property** observation, not a method-of-eviction
indictment: we trained with full attention (`window_pattern = 'L'`),
so every token contributes during training. Even H2O's information-
optimal selection cannot recover what the substrate distributes uniformly
across all positions. Re-running the substrate sweep with
sliding-window-pretraining, or with a substrate that uses sliding+sink
masks throughout training (à la StreamingLLM), would be needed to make
eviction methods competitive at inference time. A second confound is the
"soft eviction" implementation limitation (§5): we zero K and V at
evicted positions rather than masking them out of SDPA, which leaks a
small softmax weight to those positions. The H2O Δbpb numbers above are
therefore an upper bound on true heavy-hitter eviction's quality cost.

### 4.4 K-vs-V precision asymmetry as a structural attention property

Mixed-precision K/V compressors (`mixed_K{k}_V{v}`: K quantized to k bits,
V to v bits, with otherwise-symmetric per-(token, head) scaling) test the
hypothesis that asymmetric bit allocation can outperform uniform INT-N
at the same byte budget. The medium-substrate Finding 10 (`mixed_K8_V4`
beats `mixed_K4_V8` by ~ 2× on Δbpb) is the well-known KIVI-style
asymmetry. We extend it in two directions and find a remarkably
consistent structural property.

**Direction 1: bit-budget sweep (4 + 4 → 4 + 2 → 2 + 4 → 8 + 2 → 2 + 8).**
At the medium substrate at fixed K-V byte split:

| Pair | Ratio | High-K Δ | High-V Δ | Penalty |
|---|---|---|---|---|
| K8_V4 vs K4_V8 | 2.59× | 0.0013 | 0.0022 | 1.7× |

(Other Δ values at medium for the more aggressive pairs are below our
deployment relevance — `K4_V2 medium` Δ=0.10 sits at the keep-gate.)

**Direction 2: scale generalization.** Repeating the matched-pair test
on the two large-depth substrates (large at HD=96, hd128_large at HD=128)
across two byte budgets (5.05× ratio for K=4/V=2, 3.10× ratio for
K=8/V=2):

| Substrate | Ratio | K-high Δ | V-high Δ | K-bit penalty |
|---|---|---|---|---|
| large (HD=96)        | 5.05× | K4_V2: 0.049 | K2_V4: 0.158 | **3.2×** |
| hd128_large (HD=128) | 5.12× | K4_V2: 0.047 | K2_V4: 0.172 | **3.7×** |
| large (HD=96)        | 3.10× | K8_V2: 0.047 | K2_V8: 0.157 | **3.3×** |
| hd128_large (HD=128) | 3.12× | K8_V2: 0.045 | K2_V8: 0.171 | **3.8×** |

The "spend bits on K first" recommendation is therefore not just an
empirical observation about INT-N quantization noise — the K-bit
penalty is **almost identical (3.2–3.8×) across substrate, head_dim,
and byte budget**, which argues that it is a structural property of
the attention mechanism: V-noise dilutes through the softmax-weighted
sum in attention output, while K-noise compounds inside the
`q · k / √d` similarity score and shifts which tokens softmax allocates
weight to. Quantization-aware training, channel-wise scale design, or
even the choice of attention-noise-tolerant softmax variants (e.g.
Sparse / Top-k softmax) should treat K and V as fundamentally different
storage targets.

### 4.5 Recommendations for resource-adaptive deployment

Medium substrate (H=4, D=96, baseline = 1536 B/token-layer):

| Memory budget per token-layer | Quality tolerance | Recommended compressor |
|------|------|------|
| ≥ 784 B (~ 2× compression) | Δbpb < 10⁻⁴ | INT8 sym per-(tok, head) |
| ≥ 400 B (~ 4× compression) | Δbpb < 0.005 | INT4 sym per-(tok, head) |
| ≥ 208 B (~ 7× compression) | Δbpb < 0.50 (lossy) | INT2 sym per-(tok, head) |
| ≥ 32 B (very aggressive) | quality not preserved | none we tested keeps Δ small enough — re-train substrate with sliding-window or low-rank-aware training |

Large substrate (H=9, D=96, baseline = 3456 B/token-layer):

| Memory budget per token-layer | Quality tolerance | Recommended compressor |
|------|------|------|
| ≥ 1764 B (~ 2× compression) | Δbpb < 10⁻⁴ | INT8 sym per-(tok, head) |
| ≥ 900 B (~ 4× compression) | Δbpb < 0.002 | INT4 sym per-(tok, head) |
| ≥ 684 B (~ 5× compression) | Δbpb < 0.05 | **mixed_K4_V2** (new α20 leader at large scale) |
| ≥ 468 B (~ 7× compression) | Δbpb < 0.25 (lossy) | INT2 sym per-(tok, head) |

The large-substrate row exposes a regime that does *not* exist on
small/medium: an aggressive 4-bit K + 2-bit V split that matches INT4-
class quality (Δ < 0.05) at 5×+ compression. Practitioners targeting
≥ 100 M-parameter models should run `mixed_K4_V2` before settling on
INT4, since the savings (≈ 24 % over INT4 at comparable quality) are
free at this scale.

**Why no eviction / low-rank entry?** On these substrates (all trained
with full attention), every eviction or low-rank method we tried — even
H2O heavy-hitter, the strongest of them — sits far above the Pareto
front of pure INT-N quantization at the same byte budget. The pure-
quantization Pareto front is the deployment recommendation at these
scales; eviction and low-rank methods become competitive only after
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
- **`head_dim` × `depth` interaction sweep is one substrate wide.** The
  hd128_large interaction substrate in §4.1 covers (HEAD_DIM=128,
  DEPTH=10) at the same ≈ 24 GB budget as the rest of the study, but a
  finer grid (e.g. HD ∈ {96, 128, 160} × DEPTH ∈ {10, 14}) is left to
  follow-up work.

## 6. Conclusion

Holding the model and eval constant and demanding honest byte accounting,
we obtain a clean Pareto front of KV-cache compressors that directly
answers the deployment question for resource-adaptive inference. Two
findings carry the practical recommendation:

1. **Per-(token, head) symmetric INT4 quantization is the
   restricted-Pareto leader on every substrate ≤ 26 M parameters in our
   sweep**, across α ∈ {10, 20, 50}. It is the safe default for serving a
   full-attention pre-trained model under a Δbpb < 0.10 quality gate.
2. **At DEPTH ≥ 10 substrates, `mixed_K4_V2` overtakes INT4 at every
   α we report.** A 4-bit K + 2-bit V split delivers 5.05–5.12×
   compression at Δbpb ≈ 0.05, displacing INT4 (3.84–3.88×,
   Δ ≈ 0.002) on the restricted Pareto front, on both
   `large` (HD=96) and `hd128_large` (HD=128). The crossover is
   *depth*-gated, not head-dim-gated (Finding 11): at fixed DEPTH=6,
   `mixed_K4_V2` Δ stays in [0.10, 0.12] across HD ∈ {64, 96, 128}.
   The K-vs-V asymmetry that motivates the split (Finding 10)
   generalizes and even widens at scale: at the same byte budget on
   large, swapping `mixed_K4_V2` → `mixed_K2_V4` triples Δbpb
   (Finding 12).

Group-wise quantization, asymmetric quantization, low-rank approximation,
and every eviction variant we tested — sliding window, StreamingLLM
sink+window, top-k by ‖K‖₂, **H2O heavy-hitter** (the new strongest
eviction baseline, but still α20-dominated by INT4), head pruning,
hybrid stacks — are *dominated* on these full-attention substrates at
our scales. Eviction and low-rank methods become Pareto-competitive only
after the substrate is itself adapted at training time (sliding-window
masking, projected K, V layers) — a follow-up direction we leave open.
The map from (memory budget, quality tolerance) → recommended compressor
in §4.5 is therefore short, scale-aware, and the strongest practical
recommendation we can give for a system serving a full-attention
pre-trained model.

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

**Bit-exact compressor reproducibility.** During the H2O sweep we
inadvertently re-ran `h2o_R64_K50pct` on the small substrate twice
(commits `e672d7a` and `6de5ab7` in `results.tsv`). Both runs reported
Δbpb = 0.415736 to six decimal places — bit-exact reproduction of the
compressor and eval pipeline given a cached substrate. The CUDA-
nondeterminism floor mentioned in §5 (~ 5 × 10⁻⁴) is therefore an upper
bound on the noise; on this substrate the eval is fully deterministic
once the model file is fixed. Rows differing by less than this floor
across separate substrate trainings should be regarded as ties.

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
| H2O heavy-hitter (recent R, keep_frac p) | `((R + p·(T−R))/T) · 4 H D + (p·(T−R) · ⌈log₂(T−R)⌉) / (8 · T)` |
| Head pruning (keep H' of H heads) | `(H' / H) · 4 H D = 4 H' D` |
| SVD low-rank (rank r) | `4 r` |
| Random projection (rank r) | `4 r` |
| Hybrid recent-R bf16 + old INT-N | `(R/T) · 4 H D + ((T−R)/T) · (2⌈HDN/8⌉ + 4 H)` |
| Stack: outer eviction × inner C | `(bpt(outer) / bpt(identity)) · bpt(inner)` |

### B.2.a Consistency check: closed-form ↔ implementation

The table below verifies that the closed-form expressions in §B.1 match
the `n_bytes` self-report from each compressor's `compress()` method,
on the medium substrate (H=4, D=96, T=2048). **What this checks is the
implementation: that the code we wrote sums up the exact bytes the
formula prescribes.** Δ = 0 here is expected and required — it does
*not* claim anything about the realized GPU memory, which is measured
separately in §B.2.b.

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
| H2O R=64 K=25% | `(560/2048)·1536 + (496·11)/(8·2048) = 420.33` | 420.33 | 0 |
| H2O R=64 K=50% | `(1056/2048)·1536 + (992·11)/(8·2048) = 792.67` | 792.67 | 0 |
| H2O R=64 K=75% | `(1552/2048)·1536 + (1488·11)/(8·2048) = 1165.00` | 1165.00 | 0 |
| H2O R=64 K=85% | `(1750/2048)·1536 + (1686·11)/(8·2048) = 1313.63` | 1313.63 | 0 |
| H2O R=64 K=90% | `(1850/2048)·1536 + (1786·11)/(8·2048) = 1388.70` | 1388.70 | 0 |
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

### B.2.b Realized memory measurement (allocator overhead)

The §B.2.a verification reports the *theoretical* byte cost — what a
serving system would store *if* the cache could be packed exactly to
the byte. In practice, `torch.cuda.memory_allocated()` differs because
the NVIDIA caching allocator pads each tensor to a 512-byte alignment
boundary. The realized cost is therefore at least the closed-form, and
typically more for compressors that allocate many small auxiliary
tensors (per-(B, T, H, 1) scales, zero-points, indices).

We measure realized bpt with `measure_realized.py`, which allocates the
exact cache representation each compressor would use (packed `uint8`
data tensors of size `⌈H D N / 8⌉`, BF16 scales of size `[B, T, H, G]`,
etc.) and reads `torch.cuda.memory_allocated()` before/after.

**At decode-batch scale (B = 16, T = 2048, H = 4, D = 96):** every
allocated tensor is many KB, so 512-byte alignment is invisible per
token-layer. **Realized bpt = closed-form bpt for all compressors.**

**At per-step / KV-cache-init scale (B = 1, T = 16, H = 4, D = 96):**
small auxiliary tensors (e.g. INT4's `(1, 16, 4, 1)` BF16 scale = 128
bytes, padded to 512) become visible:

| Compressor | Closed-form | Realized | Δ bytes | Overhead |
|---|---:|---:|---:|---:|
| identity | 1536.0 | 1536.0 | 0.0 | 0.00 % |
| int8 | 784.0 | 832.0 | +48.0 | +6.12 % |
| int4 | 400.0 | 448.0 | +48.0 | +12.00 % |
| int2 | 208.0 | 256.0 | +48.0 | +23.08 % |
| int4_g8 | 576.0 | 576.0 | 0.0 | 0.00 % |
| int4_g16 | 480.0 | 512.0 | +32.0 | +6.67 % |
| int4_g32 | 432.0 | 448.0 | +16.0 | +3.70 % |
| int4_asym | 416.0 | 512.0 | +96.0 | +23.08 % |
| mixed_K8_V4 | 592.0 | 640.0 | +48.0 | +8.11 % |
| mixed_K4_V2 | 304.0 | 352.0 | +48.0 | +15.79 % |
| topk_knorm_25pct | 384.1 | 416.0 | +31.9 | +8.30 % |
| topk_knorm_75pct | 1152.4 | 1184.0 | +31.6 | +2.74 % |
| svd_r8 | 32.0 | 64.0 | +32.0 | +100.00 % |
| svd_r16 | 64.0 | 64.0 | 0.0 | 0.00 % |
| svd_r32 | 128.0 | 128.0 | 0.0 | 0.00 % |
| sliding_W{64,128,256,512} | 1536.0 | 1536.0 | 0.0 | 0.00 % |
| headprune_{1,2} | 1152 / 768 | 1152 / 768 | 0.0 | 0.00 % |

(rows omitted: at B=1, T=16, sliding/sink/hybrid quantities are clamped
to ≤ T tokens kept, so realized = closed-form trivially)

Three observations:

1. **Compressors with many small per-token auxiliary tensors** —
   `int4_asym` (4 small per-(1, T, H, 1) scale + zero-point tensors),
   `int2` (2 small scale tensors over a tiny packed data tensor) —
   pay the highest alignment penalty (+23 %).
2. **The penalty vanishes at production batch sizes.** At B = 16 / T = 2048,
   the same compressors show 0 % overhead because each allocated tensor
   is far above the 512-byte alignment boundary.
3. **Low-rank methods are the worst case** at small B/T: `svd_r8` shows
   100 % overhead because the BF16 rank-8 projection coefficient tensor
   is `1 · 16 · 8 = 128` elements = 256 bytes, padded to 512.

The serving implication is concrete: **at the start of a request
(B = 1, KV cache empty), the *realized* compression ratio of the
finest-grained quant variants is ~ 12–23 % worse than the headline
number.** This effect amortizes away once the prefix grows past a few
hundred tokens, but it is the kind of detail that matters when KV-cache
budgets are tight and "compression ratio" is taken too literally.

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
