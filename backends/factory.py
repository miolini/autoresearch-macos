"""Factory helpers for backend adapter selection."""

from __future__ import annotations

from typing import Any, Mapping

from .base import BackendAdapter, UnsupportedBackendError
from .codex import CodexAdapter
from .local_cli import LocalCLIAdapter
from .modal import ModalAdapter

_BACKENDS = {
    "codex": CodexAdapter,
    "local": LocalCLIAdapter,
    "local_cli": LocalCLIAdapter,
    "modal": ModalAdapter,
}


def available_backends() -> tuple[str, ...]:
    """Return normalized backend names accepted by the factory."""

    return tuple(sorted(_BACKENDS))


def create_adapter(name: str | None = None, config: Mapping[str, Any] | None = None) -> BackendAdapter:
    """Create a backend adapter by normalized name.

    Local CLI is the default practical backend.
    """

    normalized = (name or "local_cli").strip().lower()
    adapter_class = _BACKENDS.get(normalized)
    if adapter_class is None:
        raise UnsupportedBackendError(
            "unsupported_backend",
            f"Unsupported backend '{name}'.",
            {"requested_backend": name, "available_backends": available_backends()},
        )
    return adapter_class(config=config)
