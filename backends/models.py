"""Shared models for backend-agnostic execution requests and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

CommandLike = str | Sequence[str]


@dataclass(slots=True)
class ArtifactRef:
    """Generic reference to an output artifact."""

    name: str
    path: str
    exists: bool | None
    kind: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionContext:
    """Execution-time settings shared across backends."""

    working_directory: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    setup_commands: tuple[str, ...] = ()
    timeout_seconds: float | None = None
    dry_run: bool = False
    shell_executable: str = "/bin/bash"

    def resolved_working_directory(self) -> Path | None:
        if not self.working_directory:
            return None
        return Path(self.working_directory).expanduser()


@dataclass(slots=True)
class BackendCapabilities:
    """Description of what a backend can do."""

    name: str
    dry_run: bool
    local_execution: bool
    remote_execution: bool
    artifact_collection: bool
    setup_hooks: bool
    plan_only: bool = False
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class RunSpec:
    """Normalized request passed into any backend adapter."""

    command: CommandLike
    context: ExecutionContext = field(default_factory=ExecutionContext)
    artifact_paths: tuple[str, ...] = ()
    run_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PreparedRun:
    """Backend-specific preparation output derived from a RunSpec."""

    backend: str
    command: str
    command_steps: tuple[str, ...]
    working_directory: str | None
    environment: dict[str, str]
    artifact_paths: tuple[str, ...]
    dry_run: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ErrorInfo:
    """Structured error payload returned by adapters."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunResult:
    """Normalized backend execution result."""

    backend: str
    status: str
    command: str
    working_directory: str | None
    dry_run: bool
    exit_code: int | None
    stdout: str
    stderr: str
    started_at: str | None
    finished_at: str | None
    duration_seconds: float | None
    artifacts: tuple[ArtifactRef, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    error: ErrorInfo | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "success"
