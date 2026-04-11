"""Backend adapters for the orchestration layer."""

from .base import AdapterError, BackendAdapter, ConfigurationError, ExecutionError, UnsupportedBackendError
from .codex import CodexAdapter
from .factory import available_backends, create_adapter
from .local_cli import LocalCLIAdapter
from .modal import ModalAdapter
from .models import (
    ArtifactRef,
    BackendCapabilities,
    ErrorInfo,
    ExecutionContext,
    PreparedRun,
    RunResult,
    RunSpec,
)

__all__ = [
    "AdapterError",
    "ArtifactRef",
    "BackendAdapter",
    "BackendCapabilities",
    "CodexAdapter",
    "ConfigurationError",
    "ErrorInfo",
    "ExecutionContext",
    "ExecutionError",
    "LocalCLIAdapter",
    "ModalAdapter",
    "PreparedRun",
    "RunResult",
    "RunSpec",
    "UnsupportedBackendError",
    "available_backends",
    "create_adapter",
]
