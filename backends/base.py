"""Base classes and helpers for backend adapters.

Adapters in this layer must stay generic. They may plan or execute commands,
collect artifacts, and report structured results, but they must not encode
benchmark semantics, thresholds, or scientific interpretation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
import shlex
from typing import Any, Mapping, Sequence

from .models import ArtifactRef, BackendCapabilities, ErrorInfo, PreparedRun, RunResult, RunSpec


class AdapterError(Exception):
    """Base class for structured adapter failures."""

    def __init__(self, code: str, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_error_info(self) -> ErrorInfo:
        return ErrorInfo(code=self.code, message=self.message, details=self.details)


class ConfigurationError(AdapterError):
    """Raised when a run spec cannot be executed by a backend."""


class ExecutionError(AdapterError):
    """Raised when command execution fails before completion."""


class UnsupportedBackendError(AdapterError):
    """Raised when a requested backend is unknown."""


class BackendAdapter(ABC):
    """Small interface implemented by all execution backends."""

    name: str

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    @abstractmethod
    def validate_config(self, run_spec: RunSpec) -> None:
        """Validate that the run spec is suitable for this backend."""

    @abstractmethod
    def prepare_run(self, run_spec: RunSpec) -> PreparedRun:
        """Translate a normalized run spec into backend-ready steps."""

    @abstractmethod
    def run(self, run_spec: RunSpec) -> RunResult:
        """Execute or simulate the run and return a normalized result."""

    @abstractmethod
    def collect_artifacts(self, run_spec: RunSpec, *, expected_only: bool = False) -> tuple[ArtifactRef, ...]:
        """Collect or describe the artifacts referenced by the run spec."""

    @abstractmethod
    def describe_backend(self) -> BackendCapabilities:
        """Describe backend capabilities and execution style."""


class BaseAdapter(BackendAdapter):
    """Helper base class shared by concrete adapters."""

    def _normalize_command(self, command: str | Sequence[str]) -> str:
        if isinstance(command, str):
            normalized = command.strip()
        else:
            normalized = shlex.join(list(command))
        if not normalized:
            raise ConfigurationError("missing_command", "RunSpec.command must not be empty.")
        return normalized

    def _resolve_working_directory(self, run_spec: RunSpec) -> Path | None:
        return run_spec.context.resolved_working_directory()

    def _build_command_steps(self, run_spec: RunSpec, normalized_command: str) -> tuple[str, ...]:
        steps: list[str] = []
        working_directory = self._resolve_working_directory(run_spec)
        if working_directory is not None:
            steps.append(f"cd {shlex.quote(str(working_directory))}")
        steps.extend(run_spec.context.setup_commands)
        steps.append(normalized_command)
        return tuple(steps)

    def _collect_generic_artifacts(
        self,
        run_spec: RunSpec,
        *,
        expected_only: bool = False,
    ) -> tuple[ArtifactRef, ...]:
        working_directory = self._resolve_working_directory(run_spec)
        artifacts: list[ArtifactRef] = []
        for raw_path in run_spec.artifact_paths:
            path = Path(raw_path).expanduser()
            if not path.is_absolute() and working_directory is not None:
                path = working_directory / path
            exists = None if expected_only else path.exists()
            metadata: dict[str, Any] = {
                "expected_only": expected_only,
                "is_absolute": path.is_absolute(),
            }
            if exists:
                metadata["is_dir"] = path.is_dir()
                if path.is_file():
                    metadata["size_bytes"] = path.stat().st_size
            artifacts.append(
                ArtifactRef(
                    name=path.name or raw_path,
                    path=str(path),
                    exists=exists,
                    metadata=metadata,
                )
            )
        return tuple(artifacts)

    def _result(
        self,
        *,
        status: str,
        prepared: PreparedRun,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        duration_seconds: float | None = None,
        artifacts: tuple[ArtifactRef, ...] = (),
        metadata: Mapping[str, Any] | None = None,
        error: ErrorInfo | None = None,
    ) -> RunResult:
        result_metadata = dict(prepared.metadata)
        if metadata:
            result_metadata.update(metadata)
        result_metadata.setdefault("command_steps", prepared.command_steps)
        return RunResult(
            backend=prepared.backend,
            status=status,
            command=prepared.command,
            working_directory=prepared.working_directory,
            dry_run=prepared.dry_run,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            artifacts=artifacts,
            metadata=result_metadata,
            error=error,
        )

    def _error_result(
        self,
        *,
        prepared: PreparedRun,
        error: AdapterError,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        duration_seconds: float | None = None,
        artifacts: tuple[ArtifactRef, ...] = (),
    ) -> RunResult:
        return self._result(
            status="error",
            prepared=prepared,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            artifacts=artifacts,
            error=error.to_error_info(),
        )

    def _placeholder_prepared_run(self, run_spec: RunSpec) -> PreparedRun:
        normalized_command = self._normalize_command(run_spec.command)
        return PreparedRun(
            backend=self.name,
            command=normalized_command,
            command_steps=self._build_command_steps(run_spec, normalized_command),
            working_directory=str(self._resolve_working_directory(run_spec)) if self._resolve_working_directory(run_spec) else None,
            environment=dict(run_spec.context.environment),
            artifact_paths=run_spec.artifact_paths,
            dry_run=run_spec.context.dry_run,
        )

    @staticmethod
    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
