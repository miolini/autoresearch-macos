from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from backends.base import UnsupportedBackendError
from backends.codex import CodexAdapter
from backends.factory import create_adapter
from backends.local_cli import LocalCLIAdapter
from backends.modal import ModalAdapter
from backends.models import ExecutionContext, RunSpec


class BackendFactoryTests(unittest.TestCase):
    def test_factory_defaults_to_local_cli(self) -> None:
        adapter = create_adapter()
        self.assertIsInstance(adapter, LocalCLIAdapter)

    def test_factory_returns_named_backend(self) -> None:
        adapter = create_adapter("codex")
        self.assertIsInstance(adapter, CodexAdapter)

    def test_factory_rejects_unknown_backend(self) -> None:
        with self.assertRaises(UnsupportedBackendError) as context:
            create_adapter("unknown-backend")
        self.assertEqual(context.exception.code, "unsupported_backend")
        self.assertIn("unknown-backend", context.exception.message)


class CodexAdapterTests(unittest.TestCase):
    def test_codex_dry_run_returns_structured_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = CodexAdapter()
            spec = RunSpec(
                command=[sys.executable, "-c", "print('hello')"],
                context=ExecutionContext(
                    working_directory=temp_dir,
                    setup_commands=("source .venv/bin/activate",),
                    dry_run=True,
                ),
                artifact_paths=("artifacts/result.json",),
                metadata={"policy_constraints": ["no hidden threshold logic"]},
            )

            result = adapter.run(spec)

            self.assertEqual(result.status, "dry_run")
            self.assertTrue(result.dry_run)
            self.assertIn("Codex execution plan:", result.stdout)
            self.assertIn("source .venv/bin/activate", result.stdout)
            self.assertIn("policy_constraints", result.metadata["codex_request"])
            self.assertEqual(result.artifacts[0].exists, None)


class LocalCLIAdapterTests(unittest.TestCase):
    def test_local_cli_captures_output_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = LocalCLIAdapter()
            artifact_path = Path(temp_dir) / "artifact.txt"
            spec = RunSpec(
                command=[
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "print('hello from local cli'); "
                        "Path('artifact.txt').write_text('artifact-ready', encoding='utf-8')"
                    ),
                ],
                context=ExecutionContext(working_directory=temp_dir),
                artifact_paths=("artifact.txt",),
            )

            result = adapter.run(spec)

            self.assertEqual(result.status, "success")
            self.assertEqual(result.exit_code, 0)
            self.assertIn("hello from local cli", result.stdout)
            self.assertTrue(artifact_path.exists())
            self.assertEqual(result.artifacts[0].exists, True)

    def test_local_cli_dry_run_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = LocalCLIAdapter()
            spec = RunSpec(
                command=[sys.executable, "-c", "print('dry run')"],
                context=ExecutionContext(working_directory=temp_dir, dry_run=True),
                artifact_paths=("artifact.txt",),
            )

            result = adapter.run(spec)

            self.assertEqual(result.status, "dry_run")
            self.assertIsNone(result.exit_code)
            self.assertEqual(result.artifacts[0].exists, None)
            self.assertIn("set -euo pipefail", result.stdout)


class ModalAdapterTests(unittest.TestCase):
    def test_modal_adapter_disabled_cleanly_when_dependency_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = ModalAdapter(config={"modal_command": "__missing_modal__"})
            spec = RunSpec(
                command="__missing_modal__ run app.py",
                context=ExecutionContext(working_directory=temp_dir),
            )

            result = adapter.run(spec)

            self.assertEqual(result.status, "error")
            self.assertIsNotNone(result.error)
            self.assertEqual(result.error.code, "missing_dependency")


class BoundaryTests(unittest.TestCase):
    def test_adapter_code_does_not_embed_benchmark_metric_names(self) -> None:
        forbidden_tokens = (
            "ToMCoordScore",
            "DeadlockRate",
            "CollisionRate",
            "SuccessRate",
            "AmbiguityEfficiency",
            "IntentionPredictionF1",
            "StrategySwitchAccuracy",
        )
        adapter_sources = sorted(Path("backends").glob("*.py"))
        self.assertTrue(adapter_sources)
        for path in adapter_sources:
            content = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                self.assertNotIn(token, content, msg=f"{token} leaked into {path}")


if __name__ == "__main__":
    unittest.main()
