#!/usr/bin/env python3
"""Deterministic helpers for the repo-local autoresearch skill."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


RESULTS_HEADER = ["commit", "val_bpb", "memory_gb", "status", "description"]
SUMMARY_FIELDS = {
    "val_bpb": float,
    "training_seconds": float,
    "total_seconds": float,
    "peak_vram_mb": float,
    "mfu_percent": float,
    "total_tokens_M": float,
    "num_steps": int,
    "num_params_M": float,
    "depth": int,
}
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "autoresearch"


def sanitize_description(text: str) -> str:
    return " ".join(text.replace("\t", " ").split())


def run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root(),
        check=check,
        text=True,
        capture_output=True,
    )


def parse_summary_lines(text: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if key not in SUMMARY_FIELDS:
            continue
        caster = SUMMARY_FIELDS[key]
        summary[key] = caster(raw_value.strip())
    if "peak_vram_mb" in summary:
        summary["memory_gb"] = round(summary["peak_vram_mb"] / 1024, 1)
    return summary


def print_payload(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


def command_check_setup(args: argparse.Namespace) -> int:
    root = repo_root()
    cache_dir = default_cache_dir()
    data_dir = cache_dir / "data"
    tokenizer_dir = cache_dir / "tokenizer"
    tokenizer_files = {
        "tokenizer.pkl": (tokenizer_dir / "tokenizer.pkl").exists(),
        "token_bytes.pt": (tokenizer_dir / "token_bytes.pt").exists(),
    }
    required_repo_files = {
        name: (root / name).exists()
        for name in ("README.md", "prepare.py", "train.py", "pyproject.toml", "program.md")
    }
    git_ok = shutil.which("git") is not None
    uv_ok = shutil.which("uv") is not None
    parquet_count = len(list(data_dir.glob("*.parquet"))) if data_dir.exists() else 0

    current_branch = ""
    dirty = None
    if git_ok:
        current_branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
        dirty = bool(run_git(["status", "--porcelain"]).stdout.strip())

    payload = {
        "repo_root": str(root),
        "current_branch": current_branch,
        "worktree_dirty": dirty,
        "has_uv": uv_ok,
        "has_git": git_ok,
        "has_venv": (root / ".venv").exists(),
        "results_tsv_exists": (root / "results.tsv").exists(),
        "cache_dir": str(cache_dir),
        "data_shards": parquet_count,
        "tokenizer_files": tokenizer_files,
        "required_repo_files": required_repo_files,
    }
    payload["ready_for_training"] = (
        uv_ok
        and git_ok
        and all(required_repo_files.values())
        and parquet_count >= 2
        and all(tokenizer_files.values())
    )
    print_payload(payload, args.json)
    return 0 if payload["ready_for_training"] else 1


def command_create_run_branch(args: argparse.Namespace) -> int:
    if not TAG_RE.match(args.tag):
        print(
            "Invalid tag. Use lowercase letters, digits, dots, underscores, or hyphens.",
            file=sys.stderr,
        )
        return 2

    branch_name = f"{args.prefix}/{args.tag}" if args.prefix else args.tag
    dirty = bool(run_git(["status", "--porcelain"]).stdout.strip())
    if dirty and not args.allow_dirty:
        print(
            "Worktree is dirty. Commit or stash changes first, or rerun with --allow-dirty.",
            file=sys.stderr,
        )
        return 1

    existing = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch_name}"],
        cwd=repo_root(),
        text=True,
        capture_output=True,
    )
    if existing.returncode == 0:
        print(f"Branch already exists: {branch_name}", file=sys.stderr)
        return 1

    base_ref = args.base or "HEAD"
    subprocess.run(
        ["git", "checkout", "-b", branch_name, base_ref],
        cwd=repo_root(),
        check=True,
    )
    print(branch_name)
    return 0


def ensure_results_file(results_path: Path, *, quiet: bool = False) -> int:
    header_line = "\t".join(RESULTS_HEADER)
    if not results_path.exists() or results_path.stat().st_size == 0:
        results_path.write_text(header_line + "\n", encoding="utf-8")
        if not quiet:
            print(f"Created {results_path}")
        return 0

    first_line = results_path.read_text(encoding="utf-8").splitlines()[0].strip()
    if first_line != header_line:
        print(
            f"{results_path} exists but does not have the expected header.",
            file=sys.stderr,
        )
        return 1

    if not quiet:
        print(f"Header already present in {results_path}")
    return 0


def command_ensure_results(args: argparse.Namespace) -> int:
    results_path = repo_root() / args.results
    return ensure_results_file(results_path)


def command_run(args: argparse.Namespace) -> int:
    root = repo_root()
    log_path = root / args.log
    command = [args.uv_cmd, "run", args.train_path]
    status = "completed"
    exit_code = 0

    with log_path.open("w", encoding="utf-8") as handle:
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            exit_code = completed.returncode
            if exit_code != 0:
                status = "nonzero_exit"
        except subprocess.TimeoutExpired:
            status = "timeout"
            exit_code = 124
            handle.write(
                f"\n[autoresearch_ops] timed out after {args.timeout} seconds running {' '.join(command)}\n"
            )

    payload = {
        "status": status,
        "exit_code": exit_code,
        "log": str(log_path),
        "command": command,
    }
    print_payload(payload, args.json)
    return 0 if status == "completed" else exit_code


def command_parse_log(args: argparse.Namespace) -> int:
    log_path = repo_root() / args.log
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        return 1

    text = log_path.read_text(encoding="utf-8", errors="replace")
    summary = parse_summary_lines(text)
    if "val_bpb" not in summary:
        lines = text.splitlines()
        tail = lines[-args.tail_lines :] if lines else []
        payload = {
            "ok": False,
            "log": str(log_path),
            "reason": "summary_not_found",
            "tail": tail,
        }
        print_payload(payload, args.json)
        return 1

    payload = {"ok": True, "log": str(log_path), **summary}
    print_payload(payload, args.json)
    return 0


def command_append_result(args: argparse.Namespace) -> int:
    results_path = repo_root() / args.results
    ensure_code = ensure_results_file(results_path, quiet=args.json)
    if ensure_code != 0:
        return ensure_code

    metrics: dict[str, Any]
    if args.log:
        log_path = repo_root() / args.log
        metrics = parse_summary_lines(log_path.read_text(encoding="utf-8", errors="replace"))
    else:
        metrics = {}

    if args.status == "crash" and "val_bpb" not in metrics:
        metrics["val_bpb"] = 0.0
        metrics["memory_gb"] = 0.0
    elif "val_bpb" not in metrics or "memory_gb" not in metrics:
        print(
            "Need either --log with a parseable summary or explicit crash status without metrics.",
            file=sys.stderr,
        )
        return 1

    row = {
        "commit": args.commit,
        "val_bpb": f"{float(metrics['val_bpb']):.6f}",
        "memory_gb": f"{float(metrics['memory_gb']):.1f}",
        "status": args.status,
        "description": sanitize_description(args.description),
    }

    with results_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_HEADER, delimiter="\t")
        writer.writerow(row)

    print_payload(row, args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-setup", help="Verify repo and cache prerequisites")
    check.add_argument("--json", action="store_true", help="Emit JSON output")
    check.set_defaults(func=command_check_setup)

    create_branch = subparsers.add_parser(
        "create-run-branch",
        help="Create a dedicated autoresearch branch from HEAD or a supplied base ref",
    )
    create_branch.add_argument("--tag", required=True, help="Run tag, e.g. apr10")
    create_branch.add_argument(
        "--prefix",
        default="autoresearch",
        help="Branch prefix (default: autoresearch)",
    )
    create_branch.add_argument(
        "--base",
        help="Base ref to branch from (default: HEAD)",
    )
    create_branch.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow branch creation with a dirty worktree",
    )
    create_branch.set_defaults(func=command_create_run_branch)

    ensure = subparsers.add_parser("ensure-results", help="Create results.tsv header if missing")
    ensure.add_argument(
        "--results",
        default="results.tsv",
        help="Results path relative to the repo root",
    )
    ensure.set_defaults(func=command_ensure_results)

    run = subparsers.add_parser("run", help="Run train.py under uv with a bounded timeout")
    run.add_argument("--log", default="run.log", help="Log path relative to the repo root")
    run.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds (default: 600)",
    )
    run.add_argument("--uv-cmd", default="uv", help="uv executable to invoke")
    run.add_argument(
        "--train-path",
        default="train.py",
        help="Training entrypoint relative to the repo root",
    )
    run.add_argument("--json", action="store_true", help="Emit JSON output")
    run.set_defaults(func=command_run)

    parse = subparsers.add_parser("parse-log", help="Parse the training summary from a run log")
    parse.add_argument("log", help="Log path relative to the repo root")
    parse.add_argument("--json", action="store_true", help="Emit JSON output")
    parse.add_argument(
        "--tail-lines",
        type=int,
        default=50,
        help="Tail lines to include when the summary is missing",
    )
    parse.set_defaults(func=command_parse_log)

    append = subparsers.add_parser("append-result", help="Append a row to results.tsv")
    append.add_argument("--commit", required=True, help="Short git commit hash")
    append.add_argument(
        "--status",
        required=True,
        choices=("keep", "discard", "crash"),
        help="Experiment status",
    )
    append.add_argument("--description", required=True, help="Short experiment description")
    append.add_argument(
        "--results",
        default="results.tsv",
        help="Results path relative to the repo root",
    )
    append.add_argument(
        "--log",
        help="Run log to parse for metrics",
    )
    append.add_argument("--json", action="store_true", help="Emit JSON output")
    append.set_defaults(func=command_append_result)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
