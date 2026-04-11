"""Codex-oriented planning adapter."""

from __future__ import annotations

from typing import Any

from .base import BaseAdapter, ConfigurationError
from .models import BackendCapabilities, PreparedRun, RunResult, RunSpec


class CodexAdapter(BaseAdapter):
    """Plan-oriented backend for agent-driven terminal execution.

    This adapter keeps the run visible as an explicit command plan. It does not
    perform hidden execution or hidden fallback.
    """

    name = "codex"

    def validate_config(self, run_spec: RunSpec) -> None:
        normalized_command = self._normalize_command(run_spec.command)
        working_directory = self._resolve_working_directory(run_spec)
        if working_directory is not None and not working_directory.exists():
            raise ConfigurationError(
                "missing_working_directory",
                "Codex adapter requires an existing working directory.",
                {"working_directory": str(working_directory)},
            )
        if not normalized_command:
            raise ConfigurationError("missing_command", "Codex adapter requires a non-empty command.")

    def prepare_run(self, run_spec: RunSpec) -> PreparedRun:
        normalized_command = self._normalize_command(run_spec.command)
        policy_constraints = tuple(str(item) for item in run_spec.metadata.get("policy_constraints", ()))
        return PreparedRun(
            backend=self.name,
            command=normalized_command,
            command_steps=self._build_command_steps(run_spec, normalized_command),
            working_directory=str(self._resolve_working_directory(run_spec)) if self._resolve_working_directory(run_spec) else None,
            environment=dict(run_spec.context.environment),
            artifact_paths=run_spec.artifact_paths,
            dry_run=run_spec.context.dry_run,
            metadata={
                "execution_mode": "plan_only",
                "policy_constraints": policy_constraints,
            },
        )

    def run(self, run_spec: RunSpec) -> RunResult:
        try:
            self.validate_config(run_spec)
            prepared = self.prepare_run(run_spec)
        except ConfigurationError as error:
            prepared = self._placeholder_prepared_run(run_spec)
            return self._error_result(prepared=prepared, error=error)

        status = "dry_run" if prepared.dry_run else "planned"
        artifacts = self.collect_artifacts(run_spec, expected_only=True)
        stdout_lines = ["Codex execution plan:"] + list(prepared.command_steps)
        if prepared.metadata.get("policy_constraints"):
            stdout_lines.append("Policy constraints:")
            stdout_lines.extend(f"- {item}" for item in prepared.metadata["policy_constraints"])

        return self._result(
            status=status,
            prepared=prepared,
            stdout="\n".join(stdout_lines),
            artifacts=artifacts,
            metadata={
                "codex_request": {
                    "working_directory": prepared.working_directory,
                    "command_steps": prepared.command_steps,
                    "policy_constraints": prepared.metadata.get("policy_constraints", ()),
                }
            },
        )

    def collect_artifacts(self, run_spec: RunSpec, *, expected_only: bool = False) -> tuple:
        return self._collect_generic_artifacts(run_spec, expected_only=True)

    def describe_backend(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            dry_run=True,
            local_execution=False,
            remote_execution=False,
            artifact_collection=True,
            setup_hooks=True,
            plan_only=True,
            notes=(
                "Builds a reproducible command plan for Codex-style execution.",
                "Does not execute commands directly or apply hidden fallback behavior.",
            ),
        )
