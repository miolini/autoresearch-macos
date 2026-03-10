import argparse
import atexit
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parent
TRAIN_PY = ROOT / "train.py"
RESULTS_TSV = ROOT / "results.tsv"
RUN_LOG = ROOT / "run.log"
TIMEOUT_SECONDS = 1800
LOCKFILE = ROOT / ".auto_loop.lock"
STATE_FILE = ROOT / "auto_loop_state.json"
RECOMMENDATIONS_MD = ROOT / "auto_loop_recommendations.md"


def run(cmd, check=True):
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, capture_output=True)


def git_output(*args):
    return run(["git", *args]).stdout.strip()


def normalize_log(text):
    return text.replace("\r", "\n")


def parse_summary(log_text):
    summary = {}
    for key in (
        "val_bpb",
        "training_seconds",
        "total_seconds",
        "peak_vram_mb",
        "mfu_percent",
        "total_tokens_M",
        "num_steps",
        "num_params_M",
        "depth",
    ):
        match = re.search(rf"^{key}:\s+([0-9.]+)$", log_text, flags=re.M)
        if match:
            summary[key] = float(match.group(1))
    return summary


def infer_family(description):
    desc = description.lower()
    if any(token in desc for token in (
        "grouped block-affine",
        "group-mixed affine",
        "split affine",
        "translation reweighting",
        "translated branch reweighting",
    )):
        return "restricted_affine_history"
    if any(token in desc for token in ("lecun-scaled", "affine init", "identity-centered")):
        return "affine_init"
    if any(token in desc for token in (
        "exact exponential",
        "affine coupling",
        "channel-wise",
        "group-wise aeg",
        "low-rank affine lift",
        "full-matrix",
        "full matrix",
        "double aeg",
    )):
        return "heavy_high_dim_lift"
    if any(token in desc for token in (
        "value embeddings",
        "window pattern",
        "short windows",
        "5 layers",
        "ssl",
        "sl",
    )):
        return "backbone"
    if any(token in desc for token in (
        "warmdown",
        "weight decay",
        "final lr",
        "learning rate",
        "batch",
    )):
        return "optimization"
    return "misc"


def read_results():
    if not RESULTS_TSV.exists():
        return []
    lines = RESULTS_TSV.read_text(encoding="utf-8").splitlines()
    if len(lines) <= 1:
        return []
    rows = []
    for line in lines[1:]:
        commit, val_bpb, memory_gb, status, description = line.split("\t", 4)
        rows.append({
            "commit": commit,
            "val_bpb": float(val_bpb),
            "memory_gb": float(memory_gb),
            "status": status,
            "description": description,
            "family": infer_family(description),
        })
    return rows


def append_result(commit, val_bpb, memory_gb, status, description):
    with RESULTS_TSV.open("a", encoding="utf-8") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")


def pid_is_running(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock():
    if LOCKFILE.exists():
        owner_text = LOCKFILE.read_text(encoding="utf-8").strip()
        try:
            owner_pid = int(owner_text)
        except ValueError:
            owner_pid = None
        if owner_pid is None or not pid_is_running(owner_pid):
            LOCKFILE.unlink()
    try:
        fd = os.open(LOCKFILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        owner = LOCKFILE.read_text(encoding="utf-8").strip() if LOCKFILE.exists() else "unknown"
        raise RuntimeError(f"auto_loop.py is already running (lock owner: {owner})") from exc
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)

    def _cleanup():
        try:
            LOCKFILE.unlink()
        except FileNotFoundError:
            pass

    atexit.register(_cleanup)


def current_best(rows):
    keep_rows = [row for row in rows if row["status"] == "keep"]
    if not keep_rows:
        raise RuntimeError("results.tsv has no keep rows")
    return min(keep_rows, key=lambda row: row["val_bpb"])


def ensure_train_clean():
    status = git_output("status", "--short", "--", "train.py")
    if status:
        raise RuntimeError(f"train.py is dirty before experiment:\n{status}")


def replace_once(text, pattern, repl, flags=0):
    new_text, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"Pattern did not match exactly once:\n{pattern}")
    return new_text


def replace_block(text, start_marker, end_marker, replacement):
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"Missing start marker:\n{start_marker}")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"Missing end marker:\n{end_marker}")
    return text[:start] + replacement + text[end:]


def set_opt_aeg_class(text, class_body):
    return replace_block(
        text,
        "class OptAEGV3(nn.Module):\n",
        "\n\nclass CausalSelfAttention",
        class_body.rstrip() + "\n\n",
    )


def set_has_ve(text, expr):
    replacement = (
        "def has_ve(layer_idx, n_layer):\n"
        f"    {expr}\n\n"
    )
    return replace_block(
        text,
        "def has_ve(layer_idx, n_layer):\n",
        "\n\ndef apply_rotary_emb",
        replacement,
    )


def set_window_pattern(text, pattern_name):
    pattern = r'^WINDOW_PATTERN = ".*?"\s+# .*?$'
    replacement = f'WINDOW_PATTERN = "{pattern_name}"  # sliding-window schedule'
    return replace_once(text, pattern, replacement, flags=re.M)


def set_constant(text, name, value, comment):
    pattern = rf"^{name} = .*?$"
    replacement = f"{name} = {value}  # {comment}"
    return replace_once(text, pattern, replacement, flags=re.M)


def grouped_block_affine_class():
    return """class OptAEGV3(nn.Module):
    def __init__(self, width, groups=8):
        super().__init__()
        assert width % groups == 0
        self.groups = groups
        self.group_size = width // groups
        shape = (groups, 1)
        self.bx = nn.Parameter(torch.zeros(shape))
        self.by = nn.Parameter(torch.zeros(shape))
        self.mx = nn.Parameter(torch.zeros(shape))
        self.my = nn.Parameter(torch.zeros(shape))
        self.bfactor = nn.Parameter(torch.empty(shape))
        self.mfactor = nn.Parameter(torch.empty(shape))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.bx.zero_()
            self.by.zero_()
            self.mx.zero_()
            self.my.zero_()
            self.bfactor.normal_(0.0, 0.05)
            self.mfactor.normal_(0.0, 0.05)

    def forward(self, data):
        B, T, C = data.shape
        x = data.reshape(B, T, self.groups, self.group_size)
        bx = self.bx.view(1, 1, self.groups, 1)
        by = self.by.view(1, 1, self.groups, 1)
        mx = self.mx.view(1, 1, self.groups, 1)
        my = self.my.view(1, 1, self.groups, 1)
        bfactor = self.bfactor.view(1, 1, self.groups, 1)
        mfactor = self.mfactor.view(1, 1, self.groups, 1)
        group_state = x.mean(dim=-1, keepdim=True)
        trans_state = group_state * (1 + by) + bx
        linear_state = group_state * (1 + my) + mx

        trans = bfactor * trans_state * torch.sigmoid(linear_state)
        log_phi = mfactor * torch.tanh(linear_state)
        # Restricted Aff(V): block-diagonal commuting linear history plus block translation.
        delta = x * torch.expm1(log_phi) + trans
        return delta.reshape(B, T, C)"""


def group_mixed_affine_class():
    return """class OptAEGV3(nn.Module):
    def __init__(self, width, groups=8):
        super().__init__()
        assert width % groups == 0
        self.groups = groups
        self.group_size = width // groups
        shape = (groups, 1)
        self.trans_mix = nn.Linear(groups, groups, bias=False)
        self.linear_mix = nn.Linear(groups, groups, bias=False)
        self.bx = nn.Parameter(torch.zeros(shape))
        self.by = nn.Parameter(torch.zeros(shape))
        self.mx = nn.Parameter(torch.zeros(shape))
        self.my = nn.Parameter(torch.zeros(shape))
        self.bfactor = nn.Parameter(torch.empty(shape))
        self.mfactor = nn.Parameter(torch.empty(shape))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.trans_mix.weight.zero_()
            self.linear_mix.weight.zero_()
            self.bx.zero_()
            self.by.zero_()
            self.mx.zero_()
            self.my.zero_()
            self.bfactor.normal_(0.0, 0.05)
            self.mfactor.normal_(0.0, 0.05)

    def forward(self, data):
        B, T, C = data.shape
        x = data.reshape(B, T, self.groups, self.group_size)
        group_state = x.mean(dim=-1)
        mixed_trans_state = group_state + self.trans_mix(group_state)
        mixed_linear_state = group_state + self.linear_mix(group_state)
        bx = self.bx.view(1, 1, self.groups, 1)
        by = self.by.view(1, 1, self.groups, 1)
        mx = self.mx.view(1, 1, self.groups, 1)
        my = self.my.view(1, 1, self.groups, 1)
        bfactor = self.bfactor.view(1, 1, self.groups, 1)
        mfactor = self.mfactor.view(1, 1, self.groups, 1)
        trans_state = mixed_trans_state.unsqueeze(-1) * (1 + by) + bx
        linear_state = mixed_linear_state.unsqueeze(-1) * (1 + my) + mx

        trans = bfactor * trans_state * torch.sigmoid(linear_state)
        log_phi = mfactor * torch.tanh(linear_state)
        # Restricted Aff(V): commuting block-diagonal M with lightweight group-space history mixing.
        delta = x * torch.expm1(log_phi) + trans
        return delta.reshape(B, T, C)"""


def split_stats_affine_class():
    return """class OptAEGV3(nn.Module):
    def __init__(self, width, groups=8):
        super().__init__()
        assert width % groups == 0
        self.groups = groups
        self.group_size = width // groups
        shape = (groups, 1)
        self.bx = nn.Parameter(torch.zeros(shape))
        self.by = nn.Parameter(torch.zeros(shape))
        self.mx = nn.Parameter(torch.zeros(shape))
        self.my = nn.Parameter(torch.zeros(shape))
        self.bfactor = nn.Parameter(torch.empty(shape))
        self.mfactor = nn.Parameter(torch.empty(shape))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.bx.zero_()
            self.by.zero_()
            self.mx.zero_()
            self.my.zero_()
            self.bfactor.normal_(0.0, 0.05)
            self.mfactor.normal_(0.0, 0.05)

    def forward(self, data):
        B, T, C = data.shape
        x = data.reshape(B, T, self.groups, self.group_size)
        group_mean = x.mean(dim=-1)
        centered = x - group_mean.unsqueeze(-1)
        group_std = centered.square().mean(dim=-1).add(1e-6).sqrt()
        bx = self.bx.view(1, 1, self.groups, 1)
        by = self.by.view(1, 1, self.groups, 1)
        mx = self.mx.view(1, 1, self.groups, 1)
        my = self.my.view(1, 1, self.groups, 1)
        bfactor = self.bfactor.view(1, 1, self.groups, 1)
        mfactor = self.mfactor.view(1, 1, self.groups, 1)
        trans_state = group_mean.unsqueeze(-1) * (1 + by) + bx
        linear_state = group_std.unsqueeze(-1) * (1 + my) + mx

        trans = bfactor * trans_state * torch.sigmoid(linear_state)
        log_phi = mfactor * torch.tanh(linear_state)
        # Restricted Aff(V): mean drives translation history, intra-group spread drives linear history.
        delta = x * torch.expm1(log_phi) + trans
        return delta.reshape(B, T, C)"""


def aacs_reweight_affine_class():
    return """class OptAEGV3(nn.Module):
    def __init__(self, width, groups=8):
        super().__init__()
        assert width % groups == 0
        self.groups = groups
        self.group_size = width // groups
        shape = (groups, 1)
        self.bx = nn.Parameter(torch.zeros(shape))
        self.by = nn.Parameter(torch.zeros(shape))
        self.mx = nn.Parameter(torch.zeros(shape))
        self.my = nn.Parameter(torch.zeros(shape))
        self.bfactor = nn.Parameter(torch.empty(shape))
        self.mfactor = nn.Parameter(torch.empty(shape))
        self.trans_carry = nn.Parameter(torch.zeros(shape))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.bx.zero_()
            self.by.zero_()
            self.mx.zero_()
            self.my.zero_()
            self.bfactor.normal_(0.0, 0.05)
            self.mfactor.normal_(0.0, 0.05)
            self.trans_carry.zero_()

    def forward(self, data):
        B, T, C = data.shape
        x = data.reshape(B, T, self.groups, self.group_size)
        bx = self.bx.view(1, 1, self.groups, 1)
        by = self.by.view(1, 1, self.groups, 1)
        mx = self.mx.view(1, 1, self.groups, 1)
        my = self.my.view(1, 1, self.groups, 1)
        bfactor = self.bfactor.view(1, 1, self.groups, 1)
        mfactor = self.mfactor.view(1, 1, self.groups, 1)
        trans_carry = self.trans_carry.view(1, 1, self.groups, 1)
        group_state = x.mean(dim=-1, keepdim=True)
        trans_state = group_state * (1 + by) + bx
        linear_state = group_state * (1 + my) + mx

        log_phi = mfactor * torch.tanh(linear_state)
        base_trans = bfactor * trans_state * torch.sigmoid(linear_state)
        trans = base_trans * (1 + trans_carry * torch.tanh(log_phi))
        # Restricted Aff(V): residual e^Lambda dA-style reweighting of translation history.
        delta = x * torch.expm1(log_phi) + trans
        return delta.reshape(B, T, C)"""


def lecun_affine_class(with_reweight=False):
    extra_attr = "        self.trans_carry = nn.Parameter(torch.zeros(shape))\n" if with_reweight else ""
    extra_reset = "            self.trans_carry.zero_()\n" if with_reweight else ""
    extra_view = "        trans_carry = self.trans_carry.view(1, 1, self.groups, 1)\n" if with_reweight else ""
    trans_body = (
        "        base_trans = bfactor * trans_state * torch.sigmoid(linear_state)\n"
        "        trans = base_trans * (1 + trans_carry * torch.tanh(log_phi))\n"
        "        # Restricted Aff(V): identity-centered LeCun init with residual translation carry.\n"
        if with_reweight else
        "        trans = bfactor * trans_state * torch.sigmoid(linear_state)\n"
        "        # Restricted Aff(V): identity-centered LeCun init for affine generators.\n"
    )
    return f"""class OptAEGV3(nn.Module):
    def __init__(self, width, groups=8):
        super().__init__()
        assert width % groups == 0
        self.groups = groups
        self.group_size = width // groups
        shape = (groups, 1)
        self.bx = nn.Parameter(torch.zeros(shape))
        self.by = nn.Parameter(torch.zeros(shape))
        self.mx = nn.Parameter(torch.zeros(shape))
        self.my = nn.Parameter(torch.zeros(shape))
        self.bfactor = nn.Parameter(torch.empty(shape))
        self.mfactor = nn.Parameter(torch.empty(shape))
{extra_attr}
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            scale = 0.5 * (self.group_size ** -0.5)
            self.bx.zero_()
            self.by.zero_()
            self.mx.zero_()
            self.my.zero_()
            self.bfactor.normal_(0.0, scale)
            self.mfactor.normal_(0.0, scale)
{extra_reset}
    def forward(self, data):
        B, T, C = data.shape
        x = data.reshape(B, T, self.groups, self.group_size)
        bx = self.bx.view(1, 1, self.groups, 1)
        by = self.by.view(1, 1, self.groups, 1)
        mx = self.mx.view(1, 1, self.groups, 1)
        my = self.my.view(1, 1, self.groups, 1)
        bfactor = self.bfactor.view(1, 1, self.groups, 1)
        mfactor = self.mfactor.view(1, 1, self.groups, 1)
{extra_view}        group_state = x.mean(dim=-1, keepdim=True)
        trans_state = group_state * (1 + by) + bx
        linear_state = group_state * (1 + my) + mx

        log_phi = mfactor * torch.tanh(linear_state)
{trans_body}        delta = x * torch.expm1(log_phi) + trans
        return delta.reshape(B, T, C)"""


def experiment_all_layer_ve(text):
    return set_has_ve(text, "return True")


def experiment_window_ssl(text):
    return set_window_pattern(text, "SSL")


def experiment_grouped_block_affine(text):
    return set_opt_aeg_class(text, grouped_block_affine_class())


def experiment_group_mixed_affine(text):
    return set_opt_aeg_class(text, group_mixed_affine_class())


def experiment_split_stats_affine(text):
    return set_opt_aeg_class(text, split_stats_affine_class())


def experiment_aacs_reweight(text):
    return set_opt_aeg_class(text, aacs_reweight_affine_class())


def experiment_lecun_affine_init(text):
    return set_opt_aeg_class(text, lecun_affine_class(with_reweight=False))


def experiment_aacs_reweight_lecun(text):
    return set_opt_aeg_class(text, lecun_affine_class(with_reweight=True))


def experiment_shorter_warmdown(text):
    return set_constant(text, "WARMDOWN_RATIO", "0.35", "fraction of time budget for LR warmdown")


def experiment_nonzero_final_lr(text):
    return set_constant(text, "FINAL_LR_FRAC", "0.1", "final LR as fraction of initial")


def experiment_lower_weight_decay(text):
    return set_constant(text, "WEIGHT_DECAY", "0.1", "lighter weight decay for Muon")


@dataclass(frozen=True)
class Experiment:
    key: str
    description: str
    commit_message: str
    family: str
    priority: int
    cost: str
    rationale: str
    apply: Callable[[str], str]


EXPERIMENTS = [
    Experiment(
        key="all_layer_ve",
        description="value embeddings in every layer without GQA",
        commit_message="Enable value embeddings in every layer",
        family="backbone",
        priority=25,
        cost="low",
        rationale="Strengthen the backbone before affine-history experiments.",
        apply=experiment_all_layer_ve,
    ),
    Experiment(
        key="window_ssl",
        description="mixed local-global window pattern SSL",
        commit_message="Use mixed local-global window pattern SSL",
        family="backbone",
        priority=35,
        cost="low",
        rationale="Best-performing attention backbone before the grouped affine lift.",
        apply=experiment_window_ssl,
    ),
    Experiment(
        key="grouped_block_affine",
        description="grouped block-affine AEG step on SSL backbone",
        commit_message="Use grouped block-affine AEG step",
        family="restricted_affine_history",
        priority=90,
        cost="low",
        rationale="Current best theory-driven affine-history backbone.",
        apply=experiment_grouped_block_affine,
    ),
    Experiment(
        key="aacs_reweight",
        description="AACS-style translated branch reweighting",
        commit_message="Add translated-branch reweighting to grouped affine step",
        family="restricted_affine_history",
        priority=88,
        cost="low",
        rationale="Implements the note's e^Lambda dA idea while staying in the stable commuting sector.",
        apply=experiment_aacs_reweight,
    ),
    Experiment(
        key="lecun_affine_init",
        description="identity-centered LeCun init for grouped affine factors",
        commit_message="Use identity-centered LeCun init for grouped affine factors",
        family="affine_init",
        priority=84,
        cost="low",
        rationale="Treats affine init as a first-class axis: fan-in aware but still near identity.",
        apply=experiment_lecun_affine_init,
    ),
    Experiment(
        key="aacs_reweight_lecun",
        description="AACS-style reweighting with identity-centered LeCun init",
        commit_message="Combine translation reweighting with identity-centered LeCun init",
        family="restricted_affine_history",
        priority=82,
        cost="low",
        rationale="Tests whether the note-aligned branch needs a better variance-preserving init to win.",
        apply=experiment_aacs_reweight_lecun,
    ),
    Experiment(
        key="group_mixed_affine",
        description="group-mixed affine history step",
        commit_message="Add group-mixed affine history step",
        family="restricted_affine_history",
        priority=30,
        cost="medium",
        rationale="Allows tiny cross-group coupling while keeping channel-space M commuting.",
        apply=experiment_group_mixed_affine,
    ),
    Experiment(
        key="split_stats_affine",
        description="split affine translation and linear statistics",
        commit_message="Split affine translation and linear statistics",
        family="restricted_affine_history",
        priority=25,
        cost="low",
        rationale="Separates translation and linear summaries, but was weaker in practice.",
        apply=experiment_split_stats_affine,
    ),
    Experiment(
        key="shorter_warmdown",
        description="shorter LR warmdown ratio 0.35",
        commit_message="Shorten LR warmdown ratio",
        family="optimization",
        priority=40,
        cost="low",
        rationale="Cheap optimizer-side follow-up once the affine-history direction is fixed.",
        apply=experiment_shorter_warmdown,
    ),
    Experiment(
        key="nonzero_final_lr",
        description="nonzero final LR fraction 0.1",
        commit_message="Keep nonzero final LR fraction",
        family="optimization",
        priority=38,
        cost="low",
        rationale="Keeps late-stage updates alive after the best backbone stabilizes.",
        apply=experiment_nonzero_final_lr,
    ),
    Experiment(
        key="lower_weight_decay",
        description="lighter Muon weight decay 0.1",
        commit_message="Lower Muon weight decay",
        family="optimization",
        priority=36,
        cost="low",
        rationale="Reduces regularization pressure once structured affine history starts helping.",
        apply=experiment_lower_weight_decay,
    ),
]


def compute_family_stats(rows):
    stats = {}
    for row in rows:
        family_stats = stats.setdefault(row["family"], {
            "keep": 0,
            "discard": 0,
            "crash": 0,
            "best_val_bpb": None,
        })
        if row["status"] in family_stats:
            family_stats[row["status"]] += 1
        if row["status"] == "keep":
            best = family_stats["best_val_bpb"]
            if best is None or row["val_bpb"] < best:
                family_stats["best_val_bpb"] = row["val_bpb"]
    return stats


def score_experiment(experiment, family_stats, best_row):
    stats = family_stats.get(experiment.family, {})
    score = experiment.priority
    score += 18 * stats.get("keep", 0)
    score -= 8 * stats.get("discard", 0)
    score -= 15 * stats.get("crash", 0)
    if experiment.cost == "low":
        score += 5

    best_desc = best_row["description"].lower()
    if "grouped block-affine" in best_desc:
        if experiment.family in {"restricted_affine_history", "affine_init"}:
            score += 20
        if experiment.family == "optimization":
            score += 6
        if experiment.family == "backbone":
            score -= 20
    elif "ssl" in best_desc or "value embeddings" in best_desc:
        if experiment.family == "backbone":
            score += 15

    if experiment.family == "heavy_high_dim_lift":
        score -= 40
    return score


def rank_experiments(rows, best_row):
    tried = {row["description"] for row in rows}
    family_stats = compute_family_stats(rows)
    ranked = []
    for experiment in EXPERIMENTS:
        if experiment.description in tried:
            continue
        ranked.append((score_experiment(experiment, family_stats, best_row), experiment))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked, family_stats


def write_state(best_row, family_stats, ranked):
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "head": git_output("rev-parse", "--short", "HEAD"),
        "best": best_row,
        "family_stats": family_stats,
        "recommended": [
            {
                "score": score,
                "key": experiment.key,
                "description": experiment.description,
                "family": experiment.family,
                "cost": experiment.cost,
                "rationale": experiment.rationale,
            }
            for score, experiment in ranked[:5]
        ],
    }
    STATE_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_recommendations(best_row, family_stats, ranked):
    lines = [
        "# Auto Loop Recommendations",
        "",
        f"Current best: `{best_row['commit']}` / `val_bpb={best_row['val_bpb']:.6f}` / {best_row['description']}",
        "",
        "## Family Scoreboard",
    ]
    for family in sorted(family_stats):
        stats = family_stats[family]
        best_val = stats["best_val_bpb"]
        best_text = "-" if best_val is None else f"{best_val:.6f}"
        lines.append(
            f"- `{family}`: keep={stats['keep']} discard={stats['discard']} crash={stats['crash']} best={best_text}"
        )
    lines.extend(["", "## Next Candidates"])
    if ranked:
        for index, (score, experiment) in enumerate(ranked[:5], start=1):
            lines.append(f"{index}. `{experiment.description}`")
            lines.append(
                f"   score={score:.1f}, family={experiment.family}, cost={experiment.cost}. {experiment.rationale}"
            )
    else:
        lines.append("No eligible experiments remain.")
    RECOMMENDATIONS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_planning_artifacts(rows):
    best_row = current_best(rows)
    ranked, family_stats = rank_experiments(rows, best_row)
    write_state(best_row, family_stats, ranked)
    write_recommendations(best_row, family_stats, ranked)
    return best_row, ranked


def run_training():
    with RUN_LOG.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            [sys.executable, "train.py"],
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    log_text = normalize_log(RUN_LOG.read_text(encoding="utf-8"))
    return parse_summary(log_text), log_text


def revert_head():
    subprocess.run(["git", "revert", "--no-edit", "HEAD"], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suggest", action="store_true", help="Only refresh and print ranked recommendations.")
    parser.add_argument("--max-experiments", type=int, default=None, help="Maximum experiments to run in this invocation.")
    args = parser.parse_args()

    rows = read_results()
    best_row, ranked = refresh_planning_artifacts(rows)
    if args.suggest:
        print(RECOMMENDATIONS_MD.read_text(encoding="utf-8"), end="")
        return

    acquire_lock()
    runs_completed = 0

    while True:
        rows = read_results()
        best_row, ranked = refresh_planning_artifacts(rows)
        if not ranked:
            print("No eligible experiments remain.", flush=True)
            break
        if args.max_experiments is not None and runs_completed >= args.max_experiments:
            print(f"Reached max_experiments={args.max_experiments}.", flush=True)
            break

        score, experiment = ranked[0]
        print(
            f"Selected [{experiment.family}] score={score:.1f}: {experiment.description}",
            flush=True,
        )

        ensure_train_clean()
        original_text = TRAIN_PY.read_text(encoding="utf-8")
        updated_text = experiment.apply(original_text)
        if updated_text == original_text:
            print(f"No-op experiment, skipping: {experiment.description}", flush=True)
            append_result(git_output("rev-parse", "--short", "HEAD"), 0.0, 0.0, "crash", experiment.description)
            runs_completed += 1
            continue

        TRAIN_PY.write_text(updated_text, encoding="utf-8")
        subprocess.run([sys.executable, "-m", "py_compile", "train.py"], cwd=ROOT, check=True)
        subprocess.run(["git", "add", "train.py"], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-m", experiment.commit_message], cwd=ROOT, check=True)

        print(f"Running experiment: {experiment.description}", flush=True)
        status = "crash"
        val_bpb = 0.0
        memory_gb = 0.0
        try:
            summary, _ = run_training()
            val_bpb = summary.get("val_bpb", 0.0)
            memory_gb = summary.get("peak_vram_mb", 0.0) / 1024.0
            if val_bpb > 0:
                status = "keep" if val_bpb < best_row["val_bpb"] else "discard"
            if status == "crash":
                print("Experiment did not produce a valid summary. See run.log.", flush=True)
        except subprocess.TimeoutExpired:
            print("Experiment timed out. See run.log.", flush=True)

        commit = git_output("rev-parse", "--short", "HEAD")
        append_result(commit, val_bpb, memory_gb, status, experiment.description)
        runs_completed += 1

        if status == "keep":
            print(f"Kept {commit} with val_bpb={val_bpb:.6f}", flush=True)
        else:
            revert_head()
            print(f"Discarded {commit} with val_bpb={val_bpb:.6f}", flush=True)


if __name__ == "__main__":
    main()
