"""Generate figures for the paper from results.tsv.

Reads results.tsv, plots:
  - figures/pareto.png            — Pareto front for each substrate
  - figures/family_pareto.png     — same but coloured by compressor family
  - figures/substrate_sweep.png   — compressor-by-substrate comparison
  - figures/score_trajectory.png  — research-progress plot
  - figures/method_comparison.png — per-method ratio + delta bar chart
"""
import os
import csv
import math
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


SUBSTRATES = ["small", "medium", "large"]
SUBSTRATE_COLORS = {"small": "#9c5fb6", "medium": "#1f77b4", "large": "#d62728"}

FAMILY_COLORS = {
    "quant":      "#1f77b4",
    "quant_grp":  "#aec7e8",
    "quant_asym": "#5b8ec7",
    "quant_mix":  "#003f88",
    "evict":      "#d62728",
    "evict_sink": "#ff9896",
    "evict_topk": "#ff7f0e",
    "lowrank":    "#9467bd",
    "headprune":  "#8c564b",
    "hybrid":     "#2ca02c",
    "stack":      "#17becf",
    "identity":   "#777777",
}


def family_of(name):
    n = name.lower()
    if n.startswith("identity"):
        return "identity"
    if n.startswith("stack_"):
        return "stack"
    if n.startswith("hybrid_"):
        return "hybrid"
    if "_asym" in n:
        return "quant_asym"
    if n.startswith("mixed_"):
        return "quant_mix"
    if "_group" in n or re.search(r"int\d+_g\d+", n):
        return "quant_grp"
    if re.match(r"int\d+", n):
        return "quant"
    if n.startswith("sliding_window"):
        return "evict"
    if n.startswith("sink"):
        return "evict_sink"
    if n.startswith("topk_"):
        return "evict_topk"
    if n.startswith("svd_") or n.startswith("randproj"):
        return "lowrank"
    if n.startswith("headprune"):
        return "headprune"
    return "identity"


def parse_row_substrate(desc):
    """Extract substrate tag (small/medium/large) from description, or 'small_legacy' for old rows."""
    m = re.search(r"\[(small|medium|large)", desc)
    if m:
        return m.group(1)
    return "small_legacy"


def parse_compressor_name(desc):
    """Extract compressor name from description (first token before '[' or first phrase)."""
    m = re.match(r"([a-zA-Z0-9_+:]+)", desc)
    if m:
        return m.group(1)
    return desc.split()[0] if desc else ""


def load_results(path="results.tsv"):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                if row["status"] == "marker":
                    continue
                if row["status"] == "crash":
                    continue
                ratio = float(row["compression_ratio"])
                if ratio <= 0:
                    continue
                desc = row["description"]
                substrate = parse_row_substrate(desc)
                name = parse_compressor_name(desc)
                rows.append({
                    "commit": row["commit"],
                    "compression_score": float(row["compression_score"]),
                    "compression_ratio": ratio,
                    "val_bpb_delta": float(row["val_bpb_delta"]),
                    "compressed_bpb": float(row["compressed_bpb"]),
                    "status": row["status"],
                    "description": desc,
                    "substrate": substrate,
                    "name": name,
                    "family": family_of(name),
                })
            except (KeyError, ValueError):
                continue
    return rows


def plot_substrate_pareto(rows, out_path):
    """Pareto colored by substrate, all on one axis."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for sub in ["small", "medium", "large"]:
        sub_rows = [r for r in rows if r["substrate"] == sub]
        if not sub_rows:
            continue
        keeps = [r for r in sub_rows if r["status"] == "keep"]
        discards = [r for r in sub_rows if r["status"] == "discard"]
        c = SUBSTRATE_COLORS[sub]
        if discards:
            ax.scatter([r["compression_ratio"] for r in discards],
                       [r["val_bpb_delta"] for r in discards],
                       marker="x", c=c, s=40, alpha=0.45, label=f"{sub} (discard)")
        if keeps:
            ax.scatter([r["compression_ratio"] for r in keeps],
                       [r["val_bpb_delta"] for r in keeps],
                       marker="o", c=c, s=90, edgecolors="black",
                       linewidths=0.7, label=f"{sub} (keep)")
            for r in keeps:
                ax.annotate(r["name"][:24], (r["compression_ratio"], r["val_bpb_delta"]),
                            fontsize=7, alpha=0.85, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, c="black", lw=0.4, alpha=0.5)
    ax.axhline(0.10, c="orange", lw=0.6, ls=":", alpha=0.6, label="Δ=0.10 (keep gate)")
    ax.set_xlabel("Compression ratio (uncompressed bytes / compressed bytes)")
    ax.set_ylabel(r"Quality loss $\Delta$val_bpb")
    ax.set_title("Pareto front by substrate scale (D=3 small, D=6 medium, D=10 large)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_family_pareto(rows, out_path, substrate="medium"):
    """Pareto colored by compressor family, single substrate."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sub_rows = [r for r in rows if r["substrate"] == substrate]
    if not sub_rows:
        plt.close(fig)
        return
    by_family = {}
    for r in sub_rows:
        by_family.setdefault(r["family"], []).append(r)
    for fam, frows in by_family.items():
        c = FAMILY_COLORS.get(fam, "#888888")
        ax.scatter([r["compression_ratio"] for r in frows],
                   [r["val_bpb_delta"] for r in frows],
                   c=c, s=70, edgecolors="black", linewidths=0.5,
                   alpha=0.85, label=fam)
        for r in frows:
            ax.annotate(r["name"][:22], (r["compression_ratio"], r["val_bpb_delta"]),
                        fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, c="black", lw=0.4, alpha=0.5)
    ax.axhline(0.10, c="orange", lw=0.6, ls=":", alpha=0.6, label="Δ=0.10 (keep gate)")
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel(r"$\Delta$val_bpb")
    ax.set_title(f"Family-resolved Pareto front ({substrate} substrate)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_substrate_sweep(rows, out_path):
    """For each compressor present on >=2 substrates, plot Δbpb vs substrate scale."""
    by_name = {}
    for r in rows:
        if r["substrate"] not in SUBSTRATES:
            continue
        by_name.setdefault(r["name"], {})[r["substrate"]] = r
    common = [n for n, d in by_name.items() if len(d) >= 2]
    if not common:
        return
    fig, (ax_d, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))
    x_pos = {"small": 0, "medium": 1, "large": 2}
    for n in sorted(common):
        d = by_name[n]
        xs = sorted([x_pos[s] for s in d.keys() if s in x_pos])
        labels_in_order = [s for s in SUBSTRATES if s in d]
        ys_d = [d[s]["val_bpb_delta"] for s in labels_in_order]
        ys_r = [d[s]["compression_ratio"] for s in labels_in_order]
        ax_d.plot(xs, ys_d, marker="o", lw=1.4, label=n[:28])
        ax_r.plot(xs, ys_r, marker="o", lw=1.4)
    for ax in (ax_d, ax_r):
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["small\n(D=3)", "medium\n(D=6)", "large\n(D=10)"])
        ax.grid(alpha=0.25)
    ax_d.axhline(0, c="black", lw=0.5)
    ax_d.axhline(0.10, c="orange", lw=0.6, ls=":")
    ax_d.set_ylabel(r"$\Delta$val_bpb")
    ax_r.set_ylabel("Compression ratio")
    ax_d.set_title("Quality loss across substrate scale")
    ax_r.set_title("Compression ratio across substrate scale")
    ax_d.legend(loc="upper right", fontsize=7, ncol=1)
    fig.suptitle("Substrate-scale sweep: ordering robustness of compressor leaderboard")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_score_trajectory(rows, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = list(range(1, len(rows) + 1))
    ys = [r["compression_score"] for r in rows]
    statuses = [r["status"] for r in rows]
    colors = {"keep": "seagreen", "discard": "lightcoral", "crash": "black"}
    cs = [colors.get(s, "gray") for s in statuses]
    ax.bar(xs, ys, color=cs, edgecolor="black", linewidth=0.4)
    running = []
    cur = -math.inf
    for s, st in zip(ys, statuses):
        if st == "keep":
            cur = max(cur, s)
        running.append(cur if cur != -math.inf else 0)
    ax.plot(xs, running, c="steelblue", lw=2, marker="o", ms=4, label="best so far")
    ax.axhline(1.0, c="gray", ls=":", lw=1, label="identity baseline")
    ax.set_xlabel("Experiment index")
    ax.set_ylabel(r"$S(\alpha=10) = $ratio$- 10 \max(\Delta\mathrm{bpb}, 0)$")
    ax.set_title("Compression score over experimentation trajectory")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_method_comparison(rows, out_path, substrate="medium"):
    sub_rows = [r for r in rows if r["substrate"] == substrate]
    keeps = [r for r in sub_rows if r["status"] == "keep"]
    if not keeps:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    labels = [r["name"][:40] for r in keeps]
    ratios = [r["compression_ratio"] for r in keeps]
    deltas = [r["val_bpb_delta"] for r in keeps]
    xs = range(len(labels))
    ax1.bar(xs, ratios, color="steelblue", edgecolor="black", linewidth=0.4)
    ax1.axhline(1.0, c="gray", ls=":", lw=1)
    ax1.set_ylabel("Compression ratio")
    ax1.set_title(f"Per-method comparison — kept experiments ({substrate} substrate)")
    ax1.grid(alpha=0.25, axis="y")
    ax2.bar(xs, deltas,
            color=["seagreen" if d <= 0 else "indianred" for d in deltas],
            edgecolor="black", linewidth=0.4)
    ax2.axhline(0, c="black", lw=0.5)
    ax2.set_ylabel(r"$\Delta$val_bpb")
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax2.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    rows = load_results()
    if not rows:
        print("No rows in results.tsv yet; nothing to plot.")
        raise SystemExit(0)
    plot_substrate_pareto(rows, os.path.join(FIG_DIR, "pareto.png"))
    plot_family_pareto(rows, os.path.join(FIG_DIR, "family_pareto.png"), substrate="medium")
    plot_substrate_sweep(rows, os.path.join(FIG_DIR, "substrate_sweep.png"))
    plot_score_trajectory(rows, os.path.join(FIG_DIR, "score_trajectory.png"))
    plot_method_comparison(rows, os.path.join(FIG_DIR, "method_comparison.png"))
    print(f"Wrote {len(rows)} rows -> "
          f"{FIG_DIR}/pareto.png, family_pareto.png, substrate_sweep.png, "
          f"score_trajectory.png, method_comparison.png")
