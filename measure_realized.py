"""Measure realized GPU memory for each compressor's storage representation.

For each compressor in the medium-substrate leaderboard, allocate the
*real* cache representation (packed integer tensors, bf16 scales,
zero-points, etc. — what a serving system would actually keep in VRAM)
and report `realized_bytes_per_token_per_layer` from
`torch.cuda.memory_allocated()`.

Compares against the closed-form expressions in Appendix B.2.a; the
delta is allocator alignment overhead (NVIDIA caching allocator pads
each tensor to 512 bytes), which the closed-form does not model.

Usage:  uv run measure_realized.py [BATCH] [SEQ] [HEADS] [HEAD_DIM]
        defaults: B=16 T=2048 H=4 D=96  (medium substrate)
"""
import sys
import torch


def realized_storage_shapes(name, B, T, H, D):
    """Return the list of (shape, dtype) pairs a serving system would
    allocate for this compressor on one layer of K, V at [B, T, H, D]."""
    bf16 = torch.bfloat16
    u8 = torch.uint8
    out = []

    def add(shape, dtype):
        out.append((tuple(shape), dtype))

    if name == "identity":
        add((B, T, H, D), bf16)
        add((B, T, H, D), bf16)
    elif name.startswith("int") and "_g" in name:
        n_bits, _, gtail = name[3:].partition("_g")
        n_bits, G = int(n_bits), int(gtail)
        n_groups = D // G
        packed = (H * D * n_bits + 7) // 8
        add((B, T, packed), u8)        # K data
        add((B, T, packed), u8)        # V data
        add((B, T, H, n_groups), bf16) # K scales
        add((B, T, H, n_groups), bf16) # V scales
    elif name.startswith("int") and name.endswith("_asym"):
        n_bits = int(name[3:-5])
        packed = (H * D * n_bits + 7) // 8
        add((B, T, packed), u8)
        add((B, T, packed), u8)
        add((B, T, H, 1), bf16)        # scales
        add((B, T, H, 1), bf16)
        add((B, T, H, 1), bf16)        # zero-points
        add((B, T, H, 1), bf16)
    elif name.startswith("int") and name[3:].isdigit():
        n_bits = int(name[3:])
        packed = (H * D * n_bits + 7) // 8
        add((B, T, packed), u8)
        add((B, T, packed), u8)
        add((B, T, H, 1), bf16)
        add((B, T, H, 1), bf16)
    elif name.startswith("mixed_K"):
        rest = name[len("mixed_K"):]
        k_bits, _, v_bits = rest.partition("_V")
        k_bits, v_bits = int(k_bits), int(v_bits)
        k_packed = (H * D * k_bits + 7) // 8
        v_packed = (H * D * v_bits + 7) // 8
        add((B, T, k_packed), u8)
        add((B, T, v_packed), u8)
        add((B, T, H, 1), bf16)
        add((B, T, H, 1), bf16)
    elif name.startswith("hybrid_R"):
        rest = name[len("hybrid_R"):]
        recent_str, _, intpart = rest.partition("_int")
        R, n_bits = int(recent_str), int(intpart)
        T_old = max(0, T - R)
        if T_old:
            packed = (H * D * n_bits + 7) // 8
            add((B, T_old, packed), u8)
            add((B, T_old, packed), u8)
            add((B, T_old, H, 1), bf16)
            add((B, T_old, H, 1), bf16)
        if R:
            add((B, R, H, D), bf16)
            add((B, R, H, D), bf16)
    elif name.startswith("sliding_W"):
        W = int(name[len("sliding_W"):])
        kept = min(W, T)
        add((B, kept, H, D), bf16)
        add((B, kept, H, D), bf16)
    elif name.startswith("sink"):
        rest = name[len("sink"):]
        s_str, _, w_part = rest.partition("_W")
        S, W = int(s_str), int(w_part)
        kept = S + min(W, T)
        add((B, kept, H, D), bf16)
        add((B, kept, H, D), bf16)
    elif name.startswith("topk_knorm_"):
        pct = int(name[len("topk_knorm_"):].rstrip("pct"))
        k = max(1, pct * T // 100)
        add((B, k, H, D), bf16)
        add((B, k, H, D), bf16)
        idx_bytes = (B * k * (T - 1).bit_length() + 7) // 8
        add((idx_bytes,), u8)
    elif name.startswith("svd_r") or name.startswith("randproj_r"):
        prefix = "svd_r" if name.startswith("svd_r") else "randproj_r"
        r = int(name[len(prefix):])
        add((B, T, r), bf16)
        add((B, T, r), bf16)
    elif name.startswith("headprune_"):
        n_drop = int(name[len("headprune_"):])
        H_keep = H - n_drop
        add((B, T, H_keep, D), bf16)
        add((B, T, H_keep, D), bf16)
    else:
        return None
    return out


def closed_form_bpt(name, B, T, H, D):
    """Return predicted bpt from Appendix B.2.a (no allocator overhead)."""
    bytes_per = lambda dtype: 2 if dtype == torch.bfloat16 else 1
    shapes = realized_storage_shapes(name, B, T, H, D)
    if shapes is None:
        return None
    total = 0
    for shape, dtype in shapes:
        n = 1
        for s in shape:
            n *= s
        total += n * bytes_per(dtype)
    return total / (B * T)


def measure_realized_bpt(name, B, T, H, D, device):
    shapes = realized_storage_shapes(name, B, T, H, D)
    if shapes is None:
        return None
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated(device)
    keep = []
    for shape, dtype in shapes:
        keep.append(torch.empty(shape, dtype=dtype, device=device))
    torch.cuda.synchronize(device)
    after = torch.cuda.memory_allocated(device)
    realized = after - before
    del keep
    torch.cuda.empty_cache()
    return realized / (B * T)


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    H = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    D = int(sys.argv[4]) if len(sys.argv) > 4 else 96
    if not torch.cuda.is_available():
        print("CUDA required for memory_allocated() measurement.")
        sys.exit(1)
    device = torch.device("cuda")

    compressors = [
        "identity",
        "int8", "int4", "int2",
        "int4_g16", "int4_g8", "int4_g32",
        "int4_asym",
        "mixed_K8_V4", "mixed_K4_V8", "mixed_K8_V2", "mixed_K4_V2",
        "sliding_W64", "sliding_W128", "sliding_W256", "sliding_W512",
        "sink4_W64", "sink4_W128", "sink4_W256",
        "topk_knorm_25pct", "topk_knorm_50pct", "topk_knorm_75pct",
        "svd_r8", "svd_r16", "svd_r32",
        "headprune_1", "headprune_2",
        "hybrid_R64_int2", "hybrid_R64_int4", "hybrid_R128_int2",
    ]

    print(f"# Realized memory measurement: B={B}, T={T}, H={H}, D={D}")
    print(f"# Identity baseline = 4*H*D = {4*H*D} bytes/token-layer")
    print()
    print(f"{'compressor':<28}  {'closed_form':>11}  {'realized':>11}  {'Δ_bytes':>9}  {'overhead %':>10}")
    print("-" * 80)
    for name in compressors:
        cf = closed_form_bpt(name, B, T, H, D)
        rz = measure_realized_bpt(name, B, T, H, D, device)
        if cf is None or rz is None:
            print(f"{name:<28}  {'-':>11}  {'-':>11}  (skip)")
            continue
        delta = rz - cf
        pct = 100.0 * delta / cf if cf > 0 else 0.0
        print(f"{name:<28}  {cf:>11.3f}  {rz:>11.3f}  {delta:>+9.3f}  {pct:>+9.2f}%")


if __name__ == "__main__":
    main()
