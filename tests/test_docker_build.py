"""Integration test: Docker image build.

This test actually builds the Docker image and verifies the sanity check
passes.  It is SLOW (~30-40 min) and requires Docker, so it is skipped
by default.  Enable it with:

    pytest tests/test_docker_build.py --run-docker

Or in CI by setting the RUN_DOCKER_TESTS=1 environment variable.
"""

import os
import pathlib
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def docker_available() -> bool:
    """Check if Docker daemon is reachable."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture
def require_docker(request):
    run_docker = (
        request.config.getoption("--run-docker", default=False)
        or os.environ.get("RUN_DOCKER_TESTS", "0") == "1"
    )
    if not run_docker:
        pytest.skip("Docker tests disabled — use --run-docker or RUN_DOCKER_TESTS=1")
    if not docker_available():
        pytest.skip("Docker daemon not available")


class TestDockerBuild:
    """Build the Docker image and verify the sanity check passes."""

    IMAGE_TAG = "dnl-f1-training:test"

    def test_docker_build_succeeds(self, require_docker):
        """Full Docker build — this is the ultimate integration test."""
        result = subprocess.run(
            [
                "docker", "build",
                "-t", self.IMAGE_TAG,
                "--progress=plain",
                str(REPO_ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        assert result.returncode == 0, (
            f"Docker build failed:\n"
            f"STDOUT (last 50 lines):\n"
            + "\n".join(result.stdout.splitlines()[-50:])
            + f"\nSTDERR (last 50 lines):\n"
            + "\n".join(result.stderr.splitlines()[-50:])
        )

    def test_docker_import_stable_audio_tools(self, require_docker):
        """Verify stable_audio_tools imports correctly inside the built image."""
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                self.IMAGE_TAG,
                "python3", "-c",
                "from stable_audio_tools.models.factory import create_model_from_config; "
                "print('stable_audio_tools.models.factory: OK')",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"stable_audio_tools import failed inside container:\n{result.stderr}"
        )
        assert "OK" in result.stdout
