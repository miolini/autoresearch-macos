"""Local shell execution adapter."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time

from .base import BaseAdapter, ConfigurationError, ExecutionError
from .models import BackendCapabilities, PreparedRun, RunResult, RunSpec


class LocalCLIAdapter(BaseAdapter):
    """Run declared shell commands locally with structured results."""

    name = "local_cli"

    def validate_config(self, run_spec: RunSpec) -> None:
        self._normalize_command(run_spec.command)

        working_directory = self._resolve_working_directory(run_spec)
        if working_directory is not None and not working_directory.exists():
            raise ConfigurationError(
                "missing_working_directory",
                "Local CLI adapter requires an existing working directory.",
                {"working_directory": str(working_directory)},
            )

        shell_executable = Path(run_spec.context.shell_executable).expanduser()
        if not shell_executable.exists() and shutil.which(str(shell_executable)) is None:
            raise ConfigurationError(
                "missing_shell",
                "Local CLI adapter could not find the configured shell executable.",
                {"shell_executable": run_spec.context.shell_executable},
            )

    def prepare_run(self, run_spec: RunSpec) -> PreparedRun:
        normalized_command = self._normalize_command(run_spec.command)
        steps = self._build_command_steps(run_spec, normalized_command)
        shell_script = "\n".join(["set -euo pipefail", *run_spec.context.setup_commands, normalized_command])

        return PreparedRun(
            backend=self.name,
            command=normalized_command,
            command_steps=steps,
            working_directory=str(self._resolve_working_directory(run_spec)) if self._resolve_working_directory(run_spec) else None,
            environment=dict(run_spec.context.environment),
            artifact_paths=run_spec.artifact_paths,
            dry_run=run_spec.context.dry_run,
            metadata={"shell_script": shell_script},
        )

    def run(self, run_spec: RunSpec) -> RunResult:
        try:
            self.validate_config(run_spec)
            prepared = self.prepare_run(run_spec)
        except ConfigurationError as error:
            prepared = self._placeholder_prepared_run(run_spec)
            return self._error_result(prepared=prepared, error=error)

        if prepared.dry_run:
            return self._result(
                status="dry_run",
                prepared=prepared,
                stdout=prepared.metadata["shell_script"],
                artifacts=self.collect_artifacts(run_spec, expected_only=True),
            )

        environment = os.environ.copy()
        environment.update(prepared.environment)

        started_at = self.utc_now()
        started_clock = time.monotonic()
        try:
            completed = subprocess.run(
                [run_spec.context.shell_executable, "-lc", prepared.metadata["shell_script"]],
                cwd=prepared.working_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=run_spec.context.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            finished_at = self.utc_now()
            duration = time.monotonic() - started_clock
            return self._error_result(
                prepared=prepared,
                error=ExecutionError(
                    "timeout",
                    "Local CLI command exceeded the configured timeout.",
                    {
                        "timeout_seconds": run_spec.context.timeout_seconds,
                        "shell_executable": run_spec.context.shell_executable,
                    },
                ),
                stdout=error.stdout or "",
                stderr=error.stderr or "",
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=duration,
                artifacts=self.collect_artifacts(run_spec),
            )

        finished_at = self.utc_now()
        duration = time.monotonic() - started_clock
        artifacts = self.collect_artifacts(run_spec)

        if completed.returncode != 0:
            return self._result(
                status="failed",
                prepared=prepared,
                stdout=completed.stdout,
                stderr=completed.stderr,
                exit_code=completed.returncode,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=duration,
                artifacts=artifacts,
                error=ExecutionError(
                    "non_zero_exit",
                    "Local CLI command exited with a non-zero status.",
                    {"exit_code": completed.returncode},
                ).to_error_info(),
            )

        return self._result(
            status="success",
            prepared=prepared,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration,
            artifacts=artifacts,
        )

    def collect_artifacts(self, run_spec: RunSpec, *, expected_only: bool = False) -> tuple:
        return self._collect_generic_artifacts(run_spec, expected_only=expected_only)

    def describe_backend(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            dry_run=True,
            local_execution=True,
            remote_execution=False,
            artifact_collection=True,
            setup_hooks=True,
            notes=(
                "Executes declared shell commands locally.",
                "Captures stdout, stderr, exit code, timing, and artifact presence.",
            ),
        )
