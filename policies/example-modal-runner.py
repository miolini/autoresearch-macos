from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

try:
    import modal
except ImportError as exc:  # pragma: no cover - optional dependency path
    raise RuntimeError("Install Modal first: pip install modal") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_INCUMBENT_ROOT = (
    PROJECT_ROOT / "modal" / "tom-experiment-incumbent" / "auxhead-lite-v2-local800"
)
REMOTE_PROJECT_ROOT = Path("/root/project")
REMOTE_INCUMBENT_ROOT = Path("/root/incumbent/auxhead-lite-v2-local800")
REMOTE_OUTPUT_ROOT = Path("/root/outputs")

DEFAULT_VOLUME_NAME = "tom-auxhead-lite-runs-v2b-20260409"
APP_NAME = "tom-auxhead-lite-runner-v2b-20260409"
DEFAULT_SAVE_EVERY = 2000
DEFAULT_COMMIT_INTERVAL_SECONDS = 60

DEFAULT_VOLUME_NAME = os.environ.get(
    "TOM_PREVIOUS_V2_DUPLICATE_VOLUME_NAME",
    DEFAULT_VOLUME_NAME,
)
APP_NAME = os.environ.get(
    "TOM_PREVIOUS_V2_DUPLICATE_APP_NAME",
    APP_NAME,
)
WAIT_FOR_RESULTS = os.environ.get("TOM_PREVIOUS_V2_WAIT_FOR_RESULTS", "1").lower() not in {
    "0",
    "false",
    "no",
}


def _parse_prefixed_line(stdout: str, prefix: str) -> str | None:
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    return None


def _read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_streaming_subprocess(
    cmd: list[str],
    cwd: Path,
    stdout_log_path: Path,
    stderr_log_path: Path,
    commit_interval_seconds: int,
) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    selector = selectors.DefaultSelector()
    assert proc.stdout is not None
    assert proc.stderr is not None
    selector.register(proc.stdout, selectors.EVENT_READ, data="stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, data="stderr")

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    last_commit = time.monotonic()

    with stdout_log_path.open("a", encoding="utf-8") as stdout_log, stderr_log_path.open(
        "a", encoding="utf-8"
    ) as stderr_log:
        while selector.get_map():
            events = selector.select(timeout=1.0)
            for key, _ in events:
                stream_name = key.data
                line = key.fileobj.readline()
                if line == "":
                    selector.unregister(key.fileobj)
                    continue

                if stream_name == "stdout":
                    stdout_lines.append(line)
                    stdout_log.write(line)
                    stdout_log.flush()
                    print(line, end="")
                else:
                    stderr_lines.append(line)
                    stderr_log.write(line)
                    stderr_log.flush()
                    print(line, end="", file=sys.stderr)

            if time.monotonic() - last_commit >= commit_interval_seconds:
                outputs.commit()
                last_commit = time.monotonic()

        returncode = proc.wait()

    outputs.commit()
    return returncode, "".join(stdout_lines), "".join(stderr_lines)


image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirements.txt"))
    .add_local_file(PROJECT_ROOT / "train.py", remote_path=str(REMOTE_PROJECT_ROOT / "train.py"))
    .add_local_file(PROJECT_ROOT / "env.py", remote_path=str(REMOTE_PROJECT_ROOT / "env.py"))
    .add_local_file(PROJECT_ROOT / "eval.py", remote_path=str(REMOTE_PROJECT_ROOT / "eval.py"))
    .add_local_file(
        LOCAL_INCUMBENT_ROOT / "seed7" / "selected_model.pt",
        remote_path=str(REMOTE_INCUMBENT_ROOT / "seed7" / "selected_model.pt"),
    )
    .add_local_file(
        LOCAL_INCUMBENT_ROOT / "seed11" / "selected_model.pt",
        remote_path=str(REMOTE_INCUMBENT_ROOT / "seed11" / "selected_model.pt"),
    )
)

app = modal.App(APP_NAME)
outputs = modal.Volume.from_name(DEFAULT_VOLUME_NAME, create_if_missing=True)


@app.function(image=image, cpu=4.0, timeout=12 * 60 * 60, volumes={str(REMOTE_OUTPUT_ROOT): outputs})
def run_auxhead_seed(
    seed: int,
    target_total_episodes: int = 140000,
    save_every: int = DEFAULT_SAVE_EVERY,
    commit_interval_seconds: int = DEFAULT_COMMIT_INTERVAL_SECONDS,
) -> dict[str, object]:
    if seed not in (7, 11):
        raise ValueError("This v2b runner packages warm starts for seed 7 and seed 11 only.")

    checkpoint_path = REMOTE_INCUMBENT_ROOT / f"seed{seed}" / "selected_model.pt"

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    start_episodes = int(checkpoint_args.get("train_episodes", 0))
    if target_total_episodes <= start_episodes:
        raise ValueError(
            f"target_total_episodes={target_total_episodes} must exceed checkpoint train_episodes={start_episodes}"
        )

    output_dir = REMOTE_OUTPUT_ROOT / "auxhead-lite" / f"seed{seed}" / f"target-{target_total_episodes}"
    checkpoint_dir = output_dir / "checkpoints"
    curve_dir = output_dir / "curves"
    analysis_dir = output_dir / "analysis"
    progress_json = output_dir / "progress.json"
    run_status_json = output_dir / "run_status.json"
    stdout_log_path = output_dir / "stdout.log"
    stderr_log_path = output_dir / "stderr.log"
    latest_checkpoint_path = checkpoint_dir / f"latest-tom_seed{seed}.pt"
    latest_curve_path = curve_dir / f"latest-curve-tom-seed{seed}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    curve_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    existing_summary = _read_json_if_exists(output_dir / "run_summary.json")
    if existing_summary and existing_summary.get("returncode") == 0:
        existing_summary["resume_mode"] = "already_complete"
        return existing_summary

    existing_progress = _read_json_if_exists(progress_json) or {}
    completed_total_episodes = int(existing_progress.get("completed_total_episodes", start_episodes))
    completed_additional_episodes = max(0, completed_total_episodes - start_episodes)
    resume_checkpoint = latest_checkpoint_path if latest_checkpoint_path.exists() else checkpoint_path
    remaining_episodes = target_total_episodes - completed_total_episodes
    if remaining_episodes <= 0:
        payload = {
            "seed": seed,
            "initial_checkpoint": str(checkpoint_path),
            "initial_checkpoint_train_episodes": start_episodes,
            "completed_total_episodes": completed_total_episodes,
            "completed_additional_episodes": completed_additional_episodes,
            "target_total_episodes": target_total_episodes,
            "output_dir": str(output_dir),
            "resume_checkpoint": str(resume_checkpoint),
            "resume_mode": "completed_without_summary",
            "returncode": 0,
        }
        _write_json(output_dir / "run_summary.json", payload)
        outputs.commit()
        return payload

    status_payload = {
        "seed": seed,
        "state": "running",
        "target_total_episodes": target_total_episodes,
        "initial_checkpoint": str(checkpoint_path),
        "resume_checkpoint": str(resume_checkpoint),
        "base_train_episodes": start_episodes,
        "completed_total_episodes": completed_total_episodes,
        "completed_additional_episodes": completed_additional_episodes,
        "remaining_episodes": remaining_episodes,
        "save_every": save_every,
        "commit_interval_seconds": commit_interval_seconds,
    }
    _write_json(run_status_json, status_payload)
    outputs.commit()

    cmd = [
        sys.executable,
        "-u",
        "train.py",
        "--variant",
        "tom",
        "--seed",
        str(seed),
        "--train-episodes",
        str(remaining_episodes),
        "--base-train-episodes",
        str(start_episodes),
        "--episode-offset",
        str(completed_total_episodes),
        "--reported-train-episodes",
        str(target_total_episodes),
        "--artifact-episode-tag",
        str(target_total_episodes),
        "--init-checkpoint",
        str(resume_checkpoint),
        "--hidden-dim",
        str(checkpoint_args.get("hidden_dim", 64)),
        "--lr",
        str(checkpoint_args.get("lr", 3e-4)),
        "--gamma",
        str(checkpoint_args.get("gamma", 0.98)),
        "--entropy-coef",
        str(checkpoint_args.get("entropy_coef", 0.01)),
        "--aux-loss-weight",
        str(checkpoint_args.get("aux_loss_weight", 0.25)),
        "--max-steps",
        str(checkpoint_args.get("max_steps", 20)),
        "--tom-experiment",
        str(checkpoint_args.get("tom_experiment", "contextual_right_of_way_switch")),
        "--tom-experiment-strength",
        str(checkpoint_args.get("tom_experiment_strength", 0.25)),
        "--tom-belief-uncertainty-threshold",
        str(checkpoint_args.get("tom_belief_uncertainty_threshold", 0.60)),
        "--tom-context-tag-threshold",
        str(checkpoint_args.get("tom_context_tag_threshold", 0.55)),
        "--tom-proceed-safety-penalty",
        str(checkpoint_args.get("tom_proceed_safety_penalty", 0.12)),
        "--tom-conflict-dist-threshold",
        str(checkpoint_args.get("tom_conflict_dist_threshold", 0.55)),
        "--tom-bottleneck-dist-threshold",
        str(checkpoint_args.get("tom_bottleneck_dist_threshold", 0.45)),
        "--analysis-conflict-dist-threshold",
        str(checkpoint_args.get("analysis_conflict_dist_threshold", 0.2)),
        "--analysis-bottleneck-dist-threshold",
        str(checkpoint_args.get("analysis_bottleneck_dist_threshold", 0.35)),
        "--analysis-urgency-threshold",
        str(checkpoint_args.get("analysis_urgency_threshold", 0.5)),
        "--save-dir",
        str(checkpoint_dir),
        "--curve-dir",
        str(curve_dir),
        "--analysis-dir",
        str(analysis_dir),
        "--resume-curve-from",
        str(latest_curve_path if latest_curve_path.exists() else ""),
        "--progress-json",
        str(progress_json),
        "--save-every",
        str(save_every),
    ]

    returncode, stdout_text, stderr_text = _run_streaming_subprocess(
        cmd=cmd,
        cwd=REMOTE_PROJECT_ROOT,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
        commit_interval_seconds=commit_interval_seconds,
    )

    saved_checkpoint = _parse_prefixed_line(stdout_text, "saved_checkpoint")
    learning_curve_csv = _parse_prefixed_line(stdout_text, "learning_curve_csv")
    choice_analysis_json = _parse_prefixed_line(stdout_text, "choice_analysis_json")
    eval_metrics_raw = _parse_prefixed_line(stdout_text, "eval_metrics")
    eval_metrics = json.loads(eval_metrics_raw) if eval_metrics_raw else None

    summary = {
        "seed": seed,
        "initial_checkpoint": str(checkpoint_path),
        "initial_checkpoint_train_episodes": start_episodes,
        "additional_train_episodes": target_total_episodes - start_episodes,
        "target_total_episodes": target_total_episodes,
        "output_dir": str(output_dir),
        "resume_checkpoint": str(resume_checkpoint),
        "resume_mode": "resumed" if resume_checkpoint != checkpoint_path else "fresh_from_incumbent",
        "saved_checkpoint": saved_checkpoint,
        "learning_curve_csv": learning_curve_csv,
        "choice_analysis_json": choice_analysis_json,
        "eval_metrics": eval_metrics,
        "returncode": returncode,
    }

    status_payload["state"] = "completed" if returncode == 0 else "failed"
    _write_json(run_status_json, status_payload)
    _write_json(output_dir / "run_summary.json", summary)
    outputs.commit()

    if returncode != 0:
        raise RuntimeError(f"train.py failed for seed {seed} with exit code {returncode}")

    return summary


@app.local_entrypoint()
def main(
    target_total_episodes: int = 140000,
    save_every: int = DEFAULT_SAVE_EVERY,
    commit_interval_seconds: int = DEFAULT_COMMIT_INTERVAL_SECONDS,
) -> None:
    calls: list[tuple[int, object]] = []
    for seed in (7, 11):
        function_call = run_auxhead_seed.spawn(
            seed,
            target_total_episodes=target_total_episodes,
            save_every=save_every,
            commit_interval_seconds=commit_interval_seconds,
        )
        calls.append((seed, function_call))
        print(json.dumps({"seed": seed, "submitted": True}, indent=2, sort_keys=True))

    print(f"app_name={APP_NAME}")
    print(f"volume_name={DEFAULT_VOLUME_NAME}")

    if not WAIT_FOR_RESULTS:
        return

    results = modal.FunctionCall.gather(*[function_call for _, function_call in calls])
    for result in results:
        print(json.dumps(result, indent=2, sort_keys=True))
