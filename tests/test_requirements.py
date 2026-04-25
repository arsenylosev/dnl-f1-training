"""Tests for requirements file consistency.

Validates that:
- All requirements files exist and are parseable
- Known package-name mistakes are not reintroduced
- torch is pinned to a stable CUDA 12.x build (not unpinned NGC nightly)
- torchmetrics is >= 1.0 (old 0.11.x imports torchvision at load time)
"""

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REQUIREMENTS_DIR = REPO_ROOT / "requirements"

EXPECTED_FILES = ["base.txt", "encode.txt", "train.txt", "smoke.txt", "augment.txt"]


class TestRequirementsExist:
    """All expected requirements files must be present."""

    @pytest.mark.parametrize("filename", EXPECTED_FILES)
    def test_file_exists(self, filename):
        path = REQUIREMENTS_DIR / filename
        assert path.exists(), f"Missing requirements file: {path}"

    @pytest.mark.parametrize("filename", EXPECTED_FILES)
    def test_file_not_empty(self, filename):
        path = REQUIREMENTS_DIR / filename
        content = path.read_text().strip()
        lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
        assert len(lines) > 0, f"Requirements file is empty: {path}"


class TestPackageNames:
    """Catch common PyPI package name mistakes."""

    WRONG_TO_CORRECT = {
        "audiotools": "descript-audiotools",
    }

    @pytest.mark.parametrize("filename", EXPECTED_FILES)
    def test_no_wrong_package_names(self, filename):
        path = REQUIREMENTS_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} does not exist")
        content = path.read_text()
        for wrong, correct in self.WRONG_TO_CORRECT.items():
            pattern = rf"(?<![a-zA-Z\-]){re.escape(wrong)}(?=[>=<\s\[]|$)"
            matches = re.findall(pattern, content, re.MULTILINE)
            assert not matches, (
                f"{filename} uses '{wrong}' — the correct PyPI name is '{correct}'"
            )


class TestTorchPin:
    """torch must be pinned to a specific version, not left unpinned."""

    def test_torch_pinned_in_dockerfile(self):
        dockerfile = (REPO_ROOT / "Dockerfile").read_text()
        assert re.search(r"torch==\d+\.\d+\.\d+", dockerfile), (
            "torch is not pinned to a specific version in the Dockerfile — "
            "unpinned installs pull the latest NGC nightly (CUDA 13.0)"
        )

    def test_torch_uses_cu124_index(self):
        dockerfile = (REPO_ROOT / "Dockerfile").read_text()
        assert "download.pytorch.org/whl/cu124" in dockerfile, (
            "torch must be installed from the cu124 index, not the NGC index"
        )


class TestTorchmetricsVersion:
    """torchmetrics must be >= 1.0 to avoid torchvision import at load time."""

    @pytest.mark.parametrize("filename", ["encode.txt", "train.txt"])
    def test_torchmetrics_not_pinned_below_1(self, filename):
        path = REQUIREMENTS_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} does not exist")
        content = path.read_text()
        if "torchmetrics" in content:
            assert not re.search(r"torchmetrics[>=<]*0\.\d+", content), (
                f"{filename} pins torchmetrics to 0.x — must be >= 1.0.0 "
                "(old 0.11.x imports torchvision at module load time)"
            )
