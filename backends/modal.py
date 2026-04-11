"""Optional Modal execution adapter."""

from __future__ import annotations

from importlib.util import find_spec
import shutil
from typing import Any, Mapping

from .base import ConfigurationError
from .local_cli import LocalCLIAdapter
from .models import BackendCapabilities, PreparedRun, RunSpec


class ModalAdapter(LocalCLIAdapter):
    """Optional backend for explicit Modal submission commands."""

    name = "modal"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self.modal_command = str(self.config.get("modal_command", "modal"))
        self.enabled = bool(self.config.get("enabled", True))

    def _modal_available(self) -> bool:
        return shutil.which(self.modal_command) is not None or find_spec("modal") is not None

    def validate_config(self, run_spec: RunSpec) -> None:
        super().validate_config(run_spec)

        if not self.enabled:
            raise ConfigurationError(
                "backend_disabled",
                "Modal adapter is disabled by configuration.",
                {"backend": self.name},
            )

        normalized_command = self._normalize_command(run_spec.command)
        expected_prefix = f"{self.modal_command} "
        if normalized_command != self.modal_command and not normalized_command.startswith(expected_prefix):
            raise ConfigurationError(
                "modal_command_required",
                "Modal adapter requires an explicit modal CLI command in RunSpec.command.",
                {
                    "expected_prefix": self.modal_command,
                    "received_command": normalized_command,
                },
            )

        if not run_spec.context.dry_run and not self._modal_available():
            raise ConfigurationError(
                "missing_dependency",
                "Modal adapter is unavailable because the modal CLI or Python package is not installed.",
                {"modal_command": self.modal_command},
            )

    def prepare_run(self, run_spec: RunSpec) -> PreparedRun:
        prepared = super().prepare_run(run_spec)
        prepared.backend = self.name
        prepared.metadata["modal_command"] = self.modal_command
        prepared.metadata["modal_available"] = self._modal_available()
        return prepared

    def describe_backend(self) -> BackendCapabilities:
        availability_note = (
            "Modal CLI or Python package detected."
            if self._modal_available()
            else "Modal support is optional and currently unavailable."
        )
        return BackendCapabilities(
            name=self.name,
            dry_run=True,
            local_execution=False,
            remote_execution=True,
            artifact_collection=True,
            setup_hooks=True,
            notes=(
                "Executes explicitly declared Modal commands only.",
                availability_note,
            ),
        )
