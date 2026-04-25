"""Tests for shell and Python script integrity.

Validates:
- Shell scripts pass bash -n (syntax check)
- Python scripts pass ast.parse (syntax check)
- No deprecated gcsfuse CLI flags in shell scripts
- smoke_test_gcs.py does not swallow import errors
"""

import ast
import pathlib
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

SHELL_SCRIPTS = [
    "scripts/bootstrap_vm.sh",
    "scripts/setup_python_env.sh",
    "scripts/gcp_train_f1_3s.sh",
    "scripts/unwrap_checkpoint.sh",
]

PYTHON_SCRIPTS = [
    "scripts/smoke_test_gcs.py",
    "scripts/ds_zero_to_pl_ckpt.py",
    "pre_encode.py",
    "train.py",
    "run_gradio.py",
    "unwrap_model.py",
]


class TestShellSyntax:
    """All shell scripts must pass bash -n."""

    @pytest.mark.parametrize("script", SHELL_SCRIPTS)
    def test_bash_syntax(self, script):
        path = REPO_ROOT / script
        if not path.exists():
            pytest.skip(f"{script} does not exist")
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"bash -n failed for {script}:\n{result.stderr}"
        )


class TestPythonSyntax:
    """All Python scripts must parse without SyntaxError."""

    @pytest.mark.parametrize("script", PYTHON_SCRIPTS)
    def test_python_syntax(self, script):
        path = REPO_ROOT / script
        if not path.exists():
            pytest.skip(f"{script} does not exist")
        source = path.read_text()
        try:
            ast.parse(source, filename=script)
        except SyntaxError as e:
            pytest.fail(f"SyntaxError in {script}: {e}")


class TestNoDeprecatedGcsfuseFlags:
    """Deprecated gcsfuse CLI flags must not appear in executable lines."""

    DEPRECATED_FLAGS = [
        "--file-cache-capacity-mb",
        "--stat-cache-ttl",
        "--type-cache-ttl",
    ]

    @pytest.mark.parametrize("script", SHELL_SCRIPTS)
    def test_no_deprecated_flags(self, script):
        path = REPO_ROOT / script
        if not path.exists():
            pytest.skip(f"{script} does not exist")
        content = path.read_text()
        lines = content.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for flag in self.DEPRECATED_FLAGS:
                assert flag not in stripped, (
                    f"{script} line {i+1}: deprecated gcsfuse flag '{flag}' — "
                    "use a config file with --config-file instead"
                )


class TestSmokeTestImportGuard:
    """smoke_test_gcs.py must not swallow import errors with a generic message."""

    def test_no_bare_except_import_error(self):
        path = REPO_ROOT / "scripts" / "smoke_test_gcs.py"
        if not path.exists():
            pytest.skip("smoke_test_gcs.py does not exist")
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type and isinstance(node.type, ast.Name):
                    if node.type.id == "ImportError" and node.name is None:
                        pytest.fail(
                            "smoke_test_gcs.py has a bare 'except ImportError:' "
                            "without capturing the exception — this swallows the "
                            "real error message"
                        )
