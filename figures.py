"""Generate figures for the paper from results.tsv.

Reads results.tsv, plots:
  - Pareto front: compression_ratio (x) vs val_bpb_delta (y), kept points highlighted
  - Bar chart of compression_score by experiment
  - Score vs experiment-index trajectory (research progress over time)

Outputs to figures/*.png.
"""
import os
import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def load_results(path="results.tsv"):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                rows.append({
                    "commit": row["commit"],
                    "compression_score": float(row["compression_score"]),
                    "compression_ratio": float(row["compression_ratio"]),
                    "val_bpb_delta": float(row["val_bpb_delta"]),
                    "compressed_bpb": float(row["compressed_bpb"]),
                    "status": row["status"],
                    "description": row["description"],
                })
            except (KeyError, ValueError):
                continue
    return rows


def plot_pareto(rows, out_path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    keeps = [r for r in rows if r["status"] == "keep" and r["compression_ratio"] > 0]
    discards = [r for r in rows if r["status"] == "discard" and r["compression_ratio"] > 0]
    if discards:
        ax.scatter([r["compression_ratio"] for r in discards],
                   [r["val_bpb_delta"] for r in discards],
                   marker="x", c="lightgray", s=50, label="discarded", zorder=2)
    if keeps:
        ax.scatter([r["compression_ratio"] for r in keeps],
                   [r["val_bpb_delta"] for r in keeps],
                   marker="o", c="steelblue", s=80, edgecolors="black",
                   linewidths=0.7, label="kept", zorder=3)
        for r in keeps:
            ax.annotate(r["description"][:32], (r["compression_ratio"], r["val_bpb_delta"]),
                        fontsize=7, alpha=0.85, xytext=(4, 4), textcoords="offset points")
    # Iso-score lines: ratio - 10*delta = const → delta = (ratio - const)/10
    if keeps or discards:
        all_r = [r["compression_ratio"] for r in keeps + discards]
        rmax = max(all_r + [1.0])
        for level in [1.0, 2.0, 3.0, 4.0]:
            xs = [1.0, rmax * 1.05]
            ys = [(x - level) / 10.0 for x in xs]
            ax.plot(xs, ys, ls="--", lw=0.6, c="gray", alpha=0.5)
            ax.text(xs[1], ys[1], f"score={level}", fontsize=7, c="gray", alpha=0.7)
    ax.axhline(0, c="black", lw=0.4, alpha=0.5)
    ax.set_xlabel("Compression ratio (uncompressed_bytes / compressed_bytes)")
    ax.set_ylabel(r"Quality loss $\Delta$val_bpb (compressed - baseline)")
    ax.set_title("Pareto front of KV-cache compressors\n(higher ratio + lower delta = better)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
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
    # Running max
    running = []
    cur = -math.inf
    for s, st in zip(ys, statuses):
        if st == "keep":
            cur = max(cur, s)
        running.append(cur if cur != -math.inf else 0)
    ax.plot(xs, running, c="steelblue", lw=2, marker="o", ms=4, label="best so far")
    ax.axhline(1.0, c="gray", ls=":", lw=1, label="identity baseline")
    ax.set_xlabel("Experiment index")
    ax.set_ylabel("compression_score (ratio - 10 * max(Δbpb, 0))")
    ax.set_title("Compression score over experimentation trajectory")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_method_comparison(rows, out_path):
    """Bar chart: compression_ratio side-by-side with val_bpb_delta per kept method."""
    keeps = [r for r in rows if r["status"] == "keep"]
    if not keeps:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    labels = [r["description"][:40] for r in keeps]
    ratios = [r["compression_ratio"] for r in keeps]
    deltas = [r["val_bpb_delta"] for r in keeps]
    xs = range(len(labels))
    ax1.bar(xs, ratios, color="steelblue", edgecolor="black", linewidth=0.4)
    ax1.axhline(1.0, c="gray", ls=":", lw=1)
    ax1.set_ylabel("Compression ratio")
    ax1.set_title("Per-method comparison (kept experiments)")
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
    plot_pareto(rows, os.path.join(FIG_DIR, "pareto.png"))
    plot_score_trajectory(rows, os.path.join(FIG_DIR, "score_trajectory.png"))
    plot_method_comparison(rows, os.path.join(FIG_DIR, "method_comparison.png"))
    print(f"Wrote {len(rows)} rows -> {FIG_DIR}/pareto.png, score_trajectory.png, method_comparison.png")
