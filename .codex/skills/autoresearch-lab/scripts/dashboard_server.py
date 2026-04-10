#!/usr/bin/env python3
"""Local dashboard server for the autoresearch-lab skill."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import threading
import tomllib
import webbrowser
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from autoresearch_ops import parse_summary_lines, repo_root


REPO_ROOT = repo_root()
SKILL_DIR = Path(__file__).resolve().parents[1]
ASSET_PATH = SKILL_DIR / "assets" / "dashboard.html"
HELPER_SCRIPT = SKILL_DIR / "scripts" / "autoresearch_ops.py"
CURRENT_OMX_TASK = REPO_ROOT / "OMX_TASK.md"
CURRENT_OMX_NOTES = REPO_ROOT / "# OMX.md"
CURRENT_POMDP_GUIDE = REPO_ROOT / "POMDP-autoresearch" / "POMDP-autoresearch fork.md"
CURRENT_SEED_GUIDE = REPO_ROOT / "POMDP-autoresearch" / "Standard seed sets.md"
CURRENT_TOMX_SKILL = Path("/Users/stephenbeale/.codex/skills/tomx/SKILL.md")
EXTERNAL_TOMX_ROOT = Path("/Users/stephenbeale/Projects/ToM_AI_Research_Team")
EXTERNAL_AGENTS_DIR = EXTERNAL_TOMX_ROOT / ".codex" / "agents"
EXTERNAL_AGENTS_README = EXTERNAL_AGENTS_DIR / "README.md"
CHAT_HISTORY_PATH = REPO_ROOT / ".omx" / "state" / "dashboard-chat-history.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def slugify(text: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")[:48] or "experiment"


def maybe_git_status(path: Path) -> dict[str, Any]:
    status = {"current_branch": "", "worktree_dirty": None}
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=path,
            check=True,
            text=True,
            capture_output=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            check=True,
            text=True,
            capture_output=True,
        )
        status["current_branch"] = branch.stdout.strip()
        status["worktree_dirty"] = bool(dirty.stdout.strip())
    except Exception:
        pass
    return status


def ensure_chat_history_store() -> dict[str, list[dict[str, Any]]]:
    CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CHAT_HISTORY_PATH.exists():
        CHAT_HISTORY_PATH.write_text(json.dumps({}, indent=2), encoding="utf-8")
        return {}
    try:
        data = json.loads(CHAT_HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {}
    if isinstance(data, dict):
        return data
    return {}


def load_chat_history() -> dict[str, list[dict[str, Any]]]:
    return ensure_chat_history_store()


def save_chat_history(history: dict[str, list[dict[str, Any]]]) -> None:
    CHAT_HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def append_chat_entry(
    profile_id: str,
    role: str,
    title: str,
    text: str,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    history = load_chat_history()
    entries = history.setdefault(profile_id, [])
    entries.append(
        {
            "id": f"{profile_id}-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "role": role,
            "title": title,
            "text": text,
            "timestamp": now_iso(),
            "meta": meta or {},
        }
    )
    history[profile_id] = entries[-80:]
    save_chat_history(history)
    return history


def format_job_summary(job: "JobRecord") -> tuple[str, str]:
    title = f"{job.action_label} {job.status}"
    summary = job.summary or {}
    if not summary:
        return title, f"{job.profile_title}: {job.action_label} finished with status `{job.status}`."

    if job.profile_id == "autoresearch":
        lines = [
            f"Status: `{job.status}`",
            f"val_bpb: `{summary.get('val_bpb', 'n/a')}`",
            f"peak_vram_mb: `{summary.get('peak_vram_mb', 'n/a')}`",
            f"log: `{summary.get('log_path', 'run.log')}`",
        ]
        return title, "\n".join(lines)

    lines = [
        f"Status: `{job.status}`",
        f"decision: `{summary.get('decision', 'n/a')}`",
    ]
    for key in (
        "mean_candidate_ToMCoordScore",
        "mean_candidate_deadlock_rate",
        "candidate_ToMCoordScore",
        "candidate_deadlock_rate",
        "selection_path",
    ):
        value = summary.get(key)
        if value is not None:
            lines.append(f"{key}: `{value}`")
    return title, "\n".join(lines)


def codex_app_available() -> bool:
    return Path("/Applications/Codex.app").exists()


def send_prompt_to_codex_app(prompt: str, *, dry_run: bool = False) -> dict[str, Any]:
    if not codex_app_available():
        raise RuntimeError("Codex.app is not installed at /Applications/Codex.app")

    if dry_run:
        return {
            "sent": False,
            "copied_to_clipboard": False,
            "target_app": "Codex",
            "dry_run": True,
            "mode": "dry_run",
        }

    subprocess.run(["pbcopy"], input=prompt, text=True, check=True)
    subprocess.run(["open", "-a", "Codex"], check=False)
    script = """
tell application "Codex" to activate
delay 0.25
tell application "System Events"
    keystroke "v" using command down
    delay 0.15
    key code 36
end tell
"""
    completed = subprocess.run(
        ["osascript", "-e", script],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return {
            "sent": False,
            "copied_to_clipboard": True,
            "target_app": "Codex",
            "opened_app": True,
            "mode": "clipboard_only",
            "automation_error": completed.stderr.strip()
            or "AppleScript could not drive Codex automatically.",
        }
    return {
        "sent": True,
        "copied_to_clipboard": True,
        "target_app": "Codex",
        "opened_app": True,
        "mode": "submitted",
    }


def parse_tom_selection(output_root: Path) -> dict[str, Any]:
    selection_path = output_root / "selection" / "selection.json"
    if not selection_path.exists():
        return {}
    data = json.loads(selection_path.read_text(encoding="utf-8"))
    summary = {
        "selection_path": str(selection_path),
        "decision": data.get("decision"),
        "baseline_ToMCoordScore": data.get("baseline_ToMCoordScore"),
        "candidate_ToMCoordScore": data.get("candidate_ToMCoordScore"),
        "baseline_deadlock_rate": data.get("baseline_deadlock_rate"),
        "candidate_deadlock_rate": data.get("candidate_deadlock_rate"),
    }
    baseline_metrics = data.get("baseline_metrics") or {}
    candidate_metrics = data.get("candidate_metrics") or {}
    for key in ("CollisionRate", "SuccessRate", "StrategySwitchAccuracy"):
        if key in baseline_metrics:
            summary[f"baseline_{key}"] = baseline_metrics.get(key)
        if key in candidate_metrics:
            summary[f"candidate_{key}"] = candidate_metrics.get(key)
    return summary


def aggregate_tom_quick_gate(step_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in step_summaries if item and item.get("candidate_ToMCoordScore") is not None]
    if not valid:
        return {}

    def mean(key: str) -> float:
        values = [float(item[key]) for item in valid if item.get(key) is not None]
        return round(sum(values) / len(values), 4) if values else 0.0

    max_deadlock_delta = max(
        float(item["candidate_deadlock_rate"]) - float(item["baseline_deadlock_rate"])
        for item in valid
    )
    decision = (
        "keep"
        if mean("candidate_ToMCoordScore") > mean("baseline_ToMCoordScore")
        and mean("candidate_deadlock_rate") <= mean("baseline_deadlock_rate")
        and max_deadlock_delta <= 0.10
        else "discard"
    )
    return {
        "decision": decision,
        "mean_baseline_ToMCoordScore": mean("baseline_ToMCoordScore"),
        "mean_candidate_ToMCoordScore": mean("candidate_ToMCoordScore"),
        "mean_baseline_deadlock_rate": mean("baseline_deadlock_rate"),
        "mean_candidate_deadlock_rate": mean("candidate_deadlock_rate"),
        "max_deadlock_delta": round(max_deadlock_delta, 4),
        "seed_count": len(valid),
    }


def load_external_agents() -> list[dict[str, Any]]:
    preferred_order = {
        "tom-train-tuner": 0,
        "tom-results-judge": 1,
        "tom-incumbent-curator": 2,
        "tom-research-policy-evolver": 3,
    }
    templates = {
        "tom-train-tuner": "Work in {{workspace}}.\nRead {{task_path}}.\nMake one train.py-only change aimed at: {{focus}}. Then run:\n  python scripts/local_runner.py --train-episodes 5 --seed {{seed}} --output-root logs/{{run_label}}_smoke\nand if that passes, run:\n  python scripts/local_runner.py --train-episodes 800 --seed {{seed}} --output-root logs/{{run_label}}\nSummarize the patch, metrics, and keep/discard recommendation.",
        "tom-results-judge": "Work in {{workspace}}.\nRead the relevant selection files and compare the candidate against the incumbent for: {{focus}}.\nUse the current decision rules to state:\n1. whether ToMCoordScore improved\n2. whether deadlock worsened\n3. whether collisions worsened\n4. whether switching improved or held\n5. keep/discard or promote/do-not-promote, including any single-seed weak spots.",
        "tom-incumbent-curator": "Work in {{workspace}}.\nSnapshot and document this promoted candidate: {{focus}}.\nCopy the exact winning train.py, create a concise incumbent note, and produce a per-seed Markdown table linked back to the source log directories.",
        "tom-research-policy-evolver": "Work in {{workspace}}.\nReview these experiment artifacts or policy notes: {{focus}}.\nClassify the line as promote, keep active, deprioritize, or retire-for-now, and draft the smallest useful policy update grounded in the existing artifact set.",
    }
    defaults = {
        "tom-train-tuner": [
            {
                "id": "postevidence-assertive-beliefs",
                "title": "Train post-evidence commitment timing",
                "focus": "post-evidence commitment timing and sharper use of assertive/cooperative belief comparisons",
                "run_label": "v3omx_postevidence_reengage_seed7",
                "seed": 7,
            },
            {
                "id": "deadlock-micro-improvement",
                "title": "Reduce residual deadlock",
                "focus": "reducing residual deadlock without weakening ambiguity handling or belief-guided switching",
                "run_label": "v3omx_deadlock_micro_seed7",
                "seed": 7,
            },
            {
                "id": "reengage-after-evidence",
                "title": "Improve re-engagement",
                "focus": "re-engagement after partial evidence instead of repeated soft-action streaks",
                "run_label": "v3omx_reengage_seed7",
                "seed": 7,
            },
        ],
        "tom-results-judge": [
            {
                "id": "judge-quick-gate",
                "title": "Judge a quick gate",
                "focus": "three fresh 800-episode seed runs versus the current incumbent quick gate",
                "run_label": "v3omx_quick_gate",
                "seed": 7,
            }
        ],
        "tom-incumbent-curator": [
            {
                "id": "snapshot-promoted-candidate",
                "title": "Snapshot a promoted candidate",
                "focus": "the latest promoted candidate run set and its winning per-seed evidence",
                "run_label": "incumbent_snapshot",
                "seed": 7,
            }
        ],
        "tom-research-policy-evolver": [
            {
                "id": "policy-review",
                "title": "Review an experiment family",
                "focus": "whether a recent experiment family should stay active, be deprioritized, or be retired-for-now",
                "run_label": "policy_review",
                "seed": 7,
            }
        ],
    }

    agents = []
    for path in sorted(
        EXTERNAL_AGENTS_DIR.glob("tom-*.toml"),
        key=lambda item: preferred_order.get(item.stem, 99),
    ):
      data = tomllib.loads(path.read_text(encoding="utf-8"))
      agent_id = data["name"]
      agents.append(
          {
              "id": agent_id,
              "name": data["name"],
              "description": data["description"],
              "model": data.get("model", "unknown"),
              "reasoning_effort": data.get("model_reasoning_effort", "unknown"),
              "source_path": str(path),
              "prompt_template": templates.get(agent_id, ""),
              "suggestions": defaults.get(agent_id, []),
          }
      )
    return agents


LOCAL_AGENTS = [
    {
        "id": "lab-train-operator",
        "name": "lab-train-operator",
        "description": "Train.py-only experiment operator for the local autoresearch repo",
        "model": "workspace",
        "reasoning_effort": "n/a",
        "source_path": str(SKILL_DIR / "SKILL.md"),
        "prompt_template": "Work in {{workspace}}.\nRead README.md, prepare.py, train.py, program.md, and .codex/skills/autoresearch-lab/references/protocol.md.\nMake one train.py-only change aimed at: {{focus}}.\nThen run:\n  uv run train.py > run.log 2>&1\nAfter it finishes, parse run.log and summarize the patch, val_bpb, peak_vram_mb, and keep/discard recommendation.",
        "suggestions": [
            {
                "id": "baseline-smallest-move",
                "title": "Establish the cleanest next move",
                "focus": "establishing the current baseline and proposing the smallest train.py change likely to improve val_bpb without ugly complexity",
                "run_label": "baseline",
                "seed": 7,
            },
            {
                "id": "simplify-attention-surface",
                "title": "Simplify a costly surface",
                "focus": "simplifying the attention or value-embedding surface while preserving or improving fixed-budget val_bpb",
                "run_label": "simplify_attention",
                "seed": 7,
            },
            {
                "id": "budget-aware-hyper-iteration",
                "title": "Tune for the 5-minute budget",
                "focus": "a train.py-only change that is specifically justified by the 5-minute budget instead of generic scale-up instincts",
                "run_label": "budget_iter",
                "seed": 7,
            },
        ],
    }
]


@dataclass
class JobStep:
    label: str
    command: list[str]
    cwd: str
    log_path: str | None = None
    output_root: str | None = None
    summary_kind: str | None = None
    status: str = "pending"
    returncode: int | None = None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobRecord:
    id: str
    profile_id: str
    profile_title: str
    action_id: str
    action_label: str
    cwd: str
    status: str
    started_at: str
    finished_at: str | None = None
    steps: list[JobStep] = field(default_factory=list)
    log_tail: deque[str] = field(default_factory=lambda: deque(maxlen=220))
    summary: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [asdict(step) for step in self.steps]
        payload["log_tail"] = list(self.log_tail)
        return payload


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: JobRecord | None = None
        self._current_process: subprocess.Popen[str] | None = None
        self._stop_requested = False

    def current(self) -> JobRecord | None:
        with self._lock:
            return self._current

    def start(self, job: JobRecord) -> JobRecord:
        with self._lock:
            if self._current and self._current.status in {"running", "stopping"}:
                raise RuntimeError("A job is already running.")
            self._current = job
            self._stop_requested = False
        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        thread.start()
        return job

    def stop(self) -> JobRecord | None:
        with self._lock:
            if not self._current or self._current.status not in {"running", "stopping"}:
                return self._current
            self._stop_requested = True
            self._current.status = "stopping"
            process = self._current_process
        if process and process.poll() is None:
            process.terminate()
        return self.current()

    def _run_job(self, job: JobRecord) -> None:
        try:
            for step in job.steps:
                with self._lock:
                    if self._stop_requested:
                        job.status = "stopped"
                        break
                    step.status = "running"
                    self._current_process = None
                self._run_step(job, step)
                if step.status != "completed":
                    job.status = "failed" if step.status != "stopped" else "stopped"
                    break
            else:
                job.status = "completed"

            if job.action_id == "tomx_quick_gate_3seed":
                job.summary = aggregate_tom_quick_gate([step.summary for step in job.steps])
            elif job.steps:
                for step in reversed(job.steps):
                    if step.summary:
                        job.summary = step.summary
                        break
            title, text = format_job_summary(job)
            append_chat_entry(job.profile_id, "system", title, text, meta={"job_id": job.id})
        finally:
            with self._lock:
                if job.finished_at is None:
                    job.finished_at = now_iso()
                self._current_process = None
                if self._current is job:
                    self._current = job

    def _run_step(self, job: JobRecord, step: JobStep) -> None:
        log_handle = None
        try:
            if step.log_path:
                log_path = Path(step.log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("w", encoding="utf-8")

            process = subprocess.Popen(
                step.command,
                cwd=step.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self._lock:
                self._current_process = process
            assert process.stdout is not None
            for line in process.stdout:
                if log_handle:
                    log_handle.write(line)
                    log_handle.flush()
                with self._lock:
                    job.log_tail.append(line)
                    if self._stop_requested and process.poll() is None:
                        process.terminate()
            process.wait()
            step.returncode = process.returncode
            step.status = "completed" if process.returncode == 0 else "failed"
            if step.status == "completed":
                step.summary = self._summarize_step(step)
            if step.status != "completed":
                with self._lock:
                    if self._stop_requested:
                        step.status = "stopped"
                        job.status = "stopped"
            with self._lock:
                self._current_process = None
            if step.status != "completed":
                job.log_tail.append(f"\n[runner] step failed: {step.label} (exit {step.returncode})\n")
        finally:
            if log_handle:
                log_handle.close()
            if job.finished_at is None and step.status in {"failed", "stopped"}:
                job.finished_at = now_iso()

    def _summarize_step(self, step: JobStep) -> dict[str, Any]:
        if step.summary_kind == "autoresearch_log" and step.log_path:
            log_path = Path(step.log_path)
            if log_path.exists():
                summary = parse_summary_lines(log_path.read_text(encoding="utf-8", errors="replace"))
                summary["log_path"] = str(log_path)
                return summary
        if step.summary_kind == "tom_selection" and step.output_root:
            return parse_tom_selection(Path(step.output_root))
        return {}


JOB_MANAGER = JobManager()


def autoresearch_status() -> dict[str, Any]:
    command = [sys.executable, str(HELPER_SCRIPT), "check-setup", "--json"]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    data = json.loads(completed.stdout or "{}")
    data["ready"] = data.get("ready_for_training", False)
    return data


def tomx_status() -> dict[str, Any]:
    data = {
        "workspace_path": str(EXTERNAL_TOMX_ROOT),
        "task_contract_path": str(CURRENT_OMX_TASK),
        "train_py_exists": (EXTERNAL_TOMX_ROOT / "train.py").exists(),
        "runner_exists": (EXTERNAL_TOMX_ROOT / "scripts" / "local_runner.py").exists(),
        "venv_python_exists": (EXTERNAL_TOMX_ROOT / ".venv" / "bin" / "python").exists(),
        "agents_ready": all(path.exists() for path in sorted(EXTERNAL_AGENTS_DIR.glob("tom-*.toml"))),
        "ready": False,
    }
    data.update(maybe_git_status(EXTERNAL_TOMX_ROOT))
    data["ready"] = (
        data["train_py_exists"]
        and data["runner_exists"]
        and data["venv_python_exists"]
        and data["agents_ready"]
        and CURRENT_OMX_TASK.exists()
    )
    return data


def build_profiles() -> list[dict[str, Any]]:
    external_agents = load_external_agents()
    tomx_prompt_templates = {"default": external_agents[0]["prompt_template"] if external_agents else ""}
    tomx_suggestions = []
    for agent in external_agents:
        tomx_prompt_templates[agent["id"]] = agent["prompt_template"]
        tomx_suggestions.extend(agent["suggestions"])

    local_prompt_templates = {
        "default": LOCAL_AGENTS[0]["prompt_template"],
        LOCAL_AGENTS[0]["id"]: LOCAL_AGENTS[0]["prompt_template"],
    }
    local_suggestions = LOCAL_AGENTS[0]["suggestions"]

    return [
        {
            "id": "autoresearch",
            "title": "Autoresearch MacOS",
            "summary": "Current repo loop for fixed-budget train.py experimentation with val_bpb as the keep/discard metric.",
            "workspace_path": str(REPO_ROOT),
            "task_contract_path": str(REPO_ROOT / "program.md"),
            "guardrails": [
                "train.py only",
                "5-minute fixed budget",
                "prepare.py stays fixed",
                "use git for keep/discard",
            ],
            "docs": [
                {"label": "README", "path": str(REPO_ROOT / "README.md"), "note": "Repo context and training contract."},
                {"label": "Program", "path": str(REPO_ROOT / "program.md"), "note": "Legacy prompt surface for the loop."},
                {"label": "Protocol", "path": str(SKILL_DIR / "references" / "protocol.md"), "note": "Reusable experiment policy."},
            ],
            "agents": LOCAL_AGENTS,
            "suggestions": local_suggestions,
            "default_agent_id": "lab-train-operator",
            "default_seed": 7,
            "branch_tag_hint": datetime.now().strftime("%b%d").lower(),
            "prompt_templates": local_prompt_templates,
            "actions": [
                {"id": "refresh_status", "label": "Refresh Status", "note": "Re-check readiness and git state."},
                {"id": "ensure_results", "label": "Ensure results.tsv", "note": "Create the canonical header if needed."},
                {"id": "create_branch", "label": "Create Run Branch", "note": "Uses the Branch Tag field and the autoresearch/<tag> convention."},
                {"id": "run_autoresearch_train", "label": "Run Baseline Train", "note": "Streams uv run train.py and parses run.log at the end.", "primary": True},
            ],
        },
        {
            "id": "tomx",
            "title": "ToMX Local Quality",
            "summary": "Train.py-only local quality mode for the external ToM workspace, using the current OMX task contract and imported repo-local agent roles.",
            "workspace_path": str(EXTERNAL_TOMX_ROOT),
            "task_contract_path": str(CURRENT_OMX_TASK),
            "guardrails": [
                "train.py only",
                "smoke for breakage",
                "800 episodes is the gate",
                "deadlock is the veto signal",
            ],
            "docs": [
                {"label": "OMX Task Contract", "path": str(CURRENT_OMX_TASK), "note": "Train.py-only Variant 1 operating rules."},
                {"label": "ToMX Skill", "path": str(CURRENT_TOMX_SKILL), "note": "Long-run research context and frontier guidance."},
                {"label": "Agent README", "path": str(EXTERNAL_AGENTS_README), "note": "Imported ToMX agent roles for tune/judge/curate/policy."},
                {"label": "Seed Set Guide", "path": str(CURRENT_SEED_GUIDE), "note": "3-seed quick gate and 5-seed promotion policy."},
                {"label": "POMDP Guide", "path": str(CURRENT_POMDP_GUIDE), "note": "Why autoresearch maps well onto POMDP-style loops."},
                {"label": "OMX Notes", "path": str(CURRENT_OMX_NOTES), "note": "Historical prompt examples and incumbent notes."},
            ],
            "agents": external_agents,
            "suggestions": tomx_suggestions,
            "default_agent_id": "tom-train-tuner",
            "default_seed": 7,
            "branch_tag_hint": "v3omx",
            "prompt_templates": tomx_prompt_templates,
            "actions": [
                {"id": "refresh_status", "label": "Refresh Status", "note": "Check runner, venv, imported agents, and git state."},
                {"id": "tomx_smoke_then_gate", "label": "Run Smoke Then Gate", "note": "5 episodes, then 800 if smoke passes.", "primary": True},
                {"id": "tomx_quick_gate_3seed", "label": "Run 3-Seed Quick Gate", "note": "Runs seeds 7, 11, and 17 at 800 episodes."},
            ],
        },
    ]


PROFILES = build_profiles()
PROFILE_MAP = {profile["id"]: profile for profile in PROFILES}


def initial_status_by_profile() -> dict[str, dict[str, Any]]:
    return {
        "autoresearch": autoresearch_status(),
        "tomx": tomx_status(),
    }


STATUS_BY_PROFILE = initial_status_by_profile()
CHAT_HISTORY = load_chat_history()


def make_autoresearch_job(profile: dict[str, Any], action_id: str, params: dict[str, Any]) -> JobRecord:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = REPO_ROOT / "run.log"
    return JobRecord(
        id=f"job-{timestamp}",
        profile_id=profile["id"],
        profile_title=profile["title"],
        action_id=action_id,
        action_label="Run Baseline Train",
        cwd=str(REPO_ROOT),
        status="running",
        started_at=now_iso(),
        steps=[
            JobStep(
                label="uv run train.py",
                command=["uv", "run", "train.py"],
                cwd=str(REPO_ROOT),
                log_path=str(log_path),
                summary_kind="autoresearch_log",
            )
        ],
    )


def make_tomx_job(profile: dict[str, Any], action_id: str, params: dict[str, Any]) -> JobRecord:
    focus = params.get("focus", "")
    base_label = params.get("run_label") or slugify(focus)
    seed = int(params.get("seed") or profile["default_seed"])
    python_path = str(EXTERNAL_TOMX_ROOT / ".venv" / "bin" / "python")
    runner_path = str(EXTERNAL_TOMX_ROOT / "scripts" / "local_runner.py")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if action_id == "tomx_smoke_then_gate":
        smoke_root = EXTERNAL_TOMX_ROOT / "logs" / f"{base_label}_smoke"
        gate_root = EXTERNAL_TOMX_ROOT / "logs" / base_label
        steps = [
            JobStep(
                label=f"Smoke seed {seed}",
                command=[python_path, runner_path, "--train-episodes", "5", "--seed", str(seed), "--output-root", str(smoke_root)],
                cwd=str(EXTERNAL_TOMX_ROOT),
                output_root=str(smoke_root),
                summary_kind="tom_selection",
            ),
            JobStep(
                label=f"Gate seed {seed}",
                command=[python_path, runner_path, "--train-episodes", "800", "--seed", str(seed), "--output-root", str(gate_root)],
                cwd=str(EXTERNAL_TOMX_ROOT),
                output_root=str(gate_root),
                summary_kind="tom_selection",
            ),
        ]
        action_label = "Run Smoke Then Gate"
    elif action_id == "tomx_quick_gate_3seed":
        seeds = [7, 11, 17]
        steps = []
        for item in seeds:
            output_root = EXTERNAL_TOMX_ROOT / "logs" / f"{base_label}_seed{item}"
            steps.append(
                JobStep(
                    label=f"Quick gate seed {item}",
                    command=[python_path, runner_path, "--train-episodes", "800", "--seed", str(item), "--output-root", str(output_root)],
                    cwd=str(EXTERNAL_TOMX_ROOT),
                    output_root=str(output_root),
                    summary_kind="tom_selection",
                )
            )
        action_label = "Run 3-Seed Quick Gate"
    else:
        raise ValueError(f"Unsupported action: {action_id}")

    return JobRecord(
        id=f"job-{timestamp}",
        profile_id=profile["id"],
        profile_title=profile["title"],
        action_id=action_id,
        action_label=action_label,
        cwd=str(EXTERNAL_TOMX_ROOT),
        status="running",
        started_at=now_iso(),
        steps=steps,
    )


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "AutoresearchLab/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_bytes(HTTPStatus.OK, ASSET_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if self.path == "/api/state":
            self._send_json(
                {
                    "profiles": PROFILES,
                    "default_profile_id": "autoresearch",
                    "current_job": JOB_MANAGER.current().to_payload() if JOB_MANAGER.current() else None,
                    "status_by_profile": STATUS_BY_PROFILE,
                    "chat_history": load_chat_history(),
                    "codex_app_available": codex_app_available(),
                }
            )
            return
        if self.path == "/api/job":
            self._send_json(
                {
                    "current_job": JOB_MANAGER.current().to_payload() if JOB_MANAGER.current() else None,
                    "status_by_profile": STATUS_BY_PROFILE,
                    "chat_history": load_chat_history(),
                }
            )
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/action":
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json()
            response = self._handle_action(payload)
            self._send_json(response)
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _handle_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id = payload.get("profile_id")
        action_id = payload.get("action_id")
        params = payload.get("params") or {}
        profile = PROFILE_MAP.get(profile_id)
        if not profile:
            raise ValueError(f"Unknown profile: {profile_id}")

        if action_id == "refresh_status":
            status = autoresearch_status() if profile_id == "autoresearch" else tomx_status()
            STATUS_BY_PROFILE[profile_id] = status
            return {"message": "Status refreshed.", "status": status}

        if action_id == "stop_job":
            job = JOB_MANAGER.stop()
            return {
                "message": "Stop signal sent." if job else "No running job.",
                "job": job.to_payload() if job else None,
            }

        if profile_id == "autoresearch":
            if action_id == "ensure_results":
                completed = subprocess.run(
                    [sys.executable, str(HELPER_SCRIPT), "ensure-results"],
                    cwd=REPO_ROOT,
                    check=False,
                    text=True,
                    capture_output=True,
                )
                STATUS_BY_PROFILE[profile_id] = autoresearch_status()
                return {
                    "message": completed.stdout.strip() or completed.stderr.strip() or "results.tsv checked.",
                    "status": STATUS_BY_PROFILE[profile_id],
                }
            if action_id == "create_branch":
                tag = params.get("branch_tag") or datetime.now().strftime("%b%d").lower()
                completed = subprocess.run(
                    [sys.executable, str(HELPER_SCRIPT), "create-run-branch", "--tag", tag],
                    cwd=REPO_ROOT,
                    check=False,
                    text=True,
                    capture_output=True,
                )
                STATUS_BY_PROFILE[profile_id] = autoresearch_status()
                if completed.returncode != 0:
                    raise ValueError(completed.stderr.strip() or "Branch creation failed.")
                return {
                    "message": completed.stdout.strip() or f"Created autoresearch/{tag}",
                    "status": STATUS_BY_PROFILE[profile_id],
                }
            if action_id == "send_prompt_to_codex":
                prompt = params.get("prompt", "").strip()
                if not prompt:
                    raise ValueError("No prompt provided.")
                history = append_chat_entry(
                    profile_id,
                    "user",
                    "Prompt sent to Codex",
                    prompt,
                    meta={
                        "agent_id": params.get("agent_id"),
                        "run_label": params.get("run_label"),
                    },
                )
                delivery = send_prompt_to_codex_app(prompt, dry_run=bool(params.get("dry_run")))
                history = append_chat_entry(
                    profile_id,
                    "system",
                    "Codex delivery",
                    (
                        "Prompt delivery prepared for Codex.app."
                        if delivery.get("dry_run")
                        else "Prompt copied to the clipboard, Codex was opened, and the prompt was submitted."
                        if delivery.get("mode") == "submitted"
                        else "Prompt copied to the clipboard and Codex was opened. If it did not auto-submit, press Cmd-V then Enter."
                    ),
                    meta=delivery,
                )
                return {
                    "message": (
                        "Prompt sent to Codex."
                        if delivery.get("mode") == "submitted" or delivery.get("dry_run")
                        else "Prompt copied and Codex opened. If needed, just press Cmd-V then Enter."
                    ),
                    "chat_history": history,
                    "status": STATUS_BY_PROFILE.get(profile_id),
                }
            if action_id == "run_autoresearch_train":
                job = make_autoresearch_job(profile, action_id, params)
                append_chat_entry(
                    profile_id,
                    "system",
                    "Job started",
                    f"Started `{job.action_label}` in `{job.cwd}`.",
                    meta={"job_id": job.id},
                )
                JOB_MANAGER.start(job)
                return {
                    "message": "Autoresearch run started.",
                    "job": job.to_payload(),
                    "chat_history": load_chat_history(),
                }

        if profile_id == "tomx":
            if action_id == "send_prompt_to_codex":
                prompt = params.get("prompt", "").strip()
                if not prompt:
                    raise ValueError("No prompt provided.")
                history = append_chat_entry(
                    profile_id,
                    "user",
                    "Prompt sent to Codex",
                    prompt,
                    meta={
                        "agent_id": params.get("agent_id"),
                        "run_label": params.get("run_label"),
                    },
                )
                delivery = send_prompt_to_codex_app(prompt, dry_run=bool(params.get("dry_run")))
                history = append_chat_entry(
                    profile_id,
                    "system",
                    "Codex delivery",
                    (
                        "Prompt delivery prepared for Codex.app."
                        if delivery.get("dry_run")
                        else "Prompt copied to the clipboard, Codex was opened, and the prompt was submitted."
                        if delivery.get("mode") == "submitted"
                        else "Prompt copied to the clipboard and Codex was opened. If it did not auto-submit, press Cmd-V then Enter."
                    ),
                    meta=delivery,
                )
                return {
                    "message": (
                        "Prompt sent to Codex."
                        if delivery.get("mode") == "submitted" or delivery.get("dry_run")
                        else "Prompt copied and Codex opened. If needed, just press Cmd-V then Enter."
                    ),
                    "chat_history": history,
                    "status": STATUS_BY_PROFILE.get(profile_id),
                }
            if action_id in {"tomx_smoke_then_gate", "tomx_quick_gate_3seed"}:
                job = make_tomx_job(profile, action_id, params)
                append_chat_entry(
                    profile_id,
                    "system",
                    "Job started",
                    f"Started `{job.action_label}` in `{job.cwd}`.",
                    meta={"job_id": job.id},
                )
                JOB_MANAGER.start(job)
                return {
                    "message": f"{job.action_label} started.",
                    "job": job.to_payload(),
                    "chat_history": load_chat_history(),
                }

        raise ValueError(f"Unsupported action: {action_id}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length) if length else b"{}"
        return json.loads(data.decode("utf-8"))

    def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")


def find_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError("Could not find an open port near the requested value.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Preferred port (default: 8765)")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically")
    args = parser.parse_args()

    port = find_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), DashboardHandler)
    url = f"http://{args.host}:{port}/"
    print(f"Autoresearch Lab running at {url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
