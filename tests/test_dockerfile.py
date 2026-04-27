"""Tests for Dockerfile structural integrity.

These tests parse the Dockerfile as text and validate that common pitfalls
(multi-line python -c indentation, editable installs, missing shadow cleanup,
heredoc syntax on Cloud Build) are not reintroduced.
"""

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text()
DOCKERFILE_LINES = DOCKERFILE.splitlines()
DOCKERIGNORE = (REPO_ROOT / ".dockerignore").read_text()
DOCKERIGNORE_LINES = [
    l.split("#", 1)[0].rstrip() for l in DOCKERIGNORE.splitlines()
]
GITIGNORE = (REPO_ROOT / ".gitignore").read_text()
GCLOUDIGNORE = (REPO_ROOT / ".gcloudignore").read_text()


# ── Base image ──────────────────────────────────────────────────────────────

class TestDockerignore:
    """.dockerignore must not drop stable_audio_tools/{data,models} from the context.

    Unanchored ``data/`` and ``models/`` match *any* directory with that name
    in the build context (like .gitignore), so they used to exclude the package
    tree and break ``pip install /workspace/`` and the Dockerfile copy step.
    """

    def test_data_and_models_patterns_are_root_anchored(self):
        for line in DOCKERIGNORE_LINES:
            if not line or line.strip().startswith("!"):
                continue
            assert line not in ("data/", "data", "models/", "models"), (
                f"Use /data/ and /models/ (root-anchored), not {line!r} — see "
                "comment in .dockerignore"
            )

    def test_gitignore_models_rule_is_root_anchored(self):
        """gcloud builds submit applies .gitignore; bare models/* drops package code."""
        # Must not have unanchored `models/*` (excludes stable_audio_tools/models/ in gcloud)
        for line in GITIGNORE.splitlines():
            s = line.split("#", 1)[0].strip()
            if s in ("models/*", "models/"):
                pytest.fail(
                    "Use /models/ (root-anchored) in .gitignore — see comment there"
                )
        assert "/models/*" in GITIGNORE, "Root /models/* pattern required for gcloud"
        assert "stable_audio_tools" in GCLOUDIGNORE, ".gcloudignore should backstop package"


class TestBaseImage:
    """Validate the FROM instruction."""

    def test_base_image_is_nvidia_pytorch(self):
        from_lines = [l for l in DOCKERFILE_LINES if l.startswith("FROM ")]
        assert len(from_lines) == 1, "Expected exactly one FROM instruction"
        assert "nvcr.io/nvidia/pytorch" in from_lines[0]

    def test_base_image_not_past_24_04(self):
        """Vertex AI A100 VMs ship driver 525.x; images past 24.04 need >= 550."""
        from_line = [l for l in DOCKERFILE_LINES if l.startswith("FROM ")][0]
        match = re.search(r"pytorch:(\d+)\.(\d+)", from_line)
        assert match, f"Cannot parse tag from: {from_line}"
        year, month = int(match.group(1)), int(match.group(2))
        assert (year, month) <= (24, 4), (
            f"Base image tag {year}.{month:02d} exceeds 24.04 — "
            "Vertex AI A100 driver 525.x cannot run CUDA 12.5+"
        )


# ── Python path shadowing ───────────────────────────────────────────────────

class TestShadowCleanup:
    """Ensure the source stable_audio_tools/ dir is removed after install."""

    def test_rm_rf_stable_audio_tools_present(self):
        # Accept both trailing-slash and no-trailing-slash variants
        assert (
            "rm -rf /workspace/stable_audio_tools/" in DOCKERFILE
            or "rm -rf /workspace/stable_audio_tools" in DOCKERFILE
        ), (
            "Missing 'rm -rf /workspace/stable_audio_tools' — the source "
            "directory will shadow the site-packages install"
        )

    def test_remove_site_pkgs_copy_before_install(self):
        """Pre-existing site-packages/stable_audio_tools makes cp -r nest the tree."""
        assert "rm -rf" in DOCKERFILE and "SITE_PKGS" in DOCKERFILE
        assert "${SITE_PKGS}/stable_audio_tools" in DOCKERFILE
        assert "stable_audio_tools/models/factory.py" in DOCKERFILE

    def test_no_editable_install(self):
        """Editable installs create .pth files that break in containers."""
        code_lines = [l for l in DOCKERFILE_LINES if not l.strip().startswith("#")]
        for line in code_lines:
            assert "pip install" not in line or "-e " not in line, (
                f"Found editable install (-e) in Dockerfile: {line.strip()}"
            )


# ── Heredoc / multi-line python -c ──────────────────────────────────────────

class TestNoBrokenHeredoc:
    """Cloud Build's older Docker daemon does not support heredoc syntax."""

    def test_no_heredoc_syntax(self):
        code_lines = [l for l in DOCKERFILE_LINES if not l.strip().startswith("#")]
        for i, line in enumerate(code_lines):
            assert "<<" not in line or "EOF" not in line, (
                f"Line {i+1}: heredoc syntax found — Cloud Build cannot parse this"
            )

    def test_no_bare_import_at_line_start(self):
        """Docker treats 'import' at the start of a line as an instruction.

        Lines that are continuations of a previous line (i.e. the previous
        non-empty line ends with '\\') are part of the same RUN instruction
        and are safe.
        """
        for i, line in enumerate(DOCKERFILE_LINES):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if not re.match(r"^import\s", stripped):
                continue
            # Check if this is a continuation of the previous line
            is_continuation = False
            for j in range(i - 1, -1, -1):
                prev = DOCKERFILE_LINES[j].rstrip()
                if not prev:  # skip blank lines
                    continue
                if prev.endswith("\\"):
                    is_continuation = True
                break
            assert is_continuation, (
                f"Line {i+1}: bare 'import' at start of line — Docker will "
                "misparse this as an unknown instruction"
            )


# ── Sanity check step ──────────────────────────────────────────────────────

class TestSanityCheck:
    """Verify the sanity-check RUN imports the critical packages."""

    REQUIRED_IMPORTS = [
        "torch",
        "torchaudio",
        "google.cloud.storage",
        "soundfile",
        "librosa",
        "pytorch_lightning",
        "clearml",
        "stable_audio_tools",
    ]

    def test_sanity_check_imports_all_required(self):
        for pkg in self.REQUIRED_IMPORTS:
            assert pkg in DOCKERFILE, (
                f"Sanity-check step is missing 'import {pkg}'"
            )


# ── pip cache purge ─────────────────────────────────────────────────────────

class TestNoPipCachePurge:
    """Cloud Build sets PIP_NO_CACHE_DIR=1; pip cache purge exits 1."""

    def test_no_pip_cache_purge_in_code(self):
        code_lines = [l for l in DOCKERFILE_LINES if not l.strip().startswith("#")]
        joined = "\n".join(code_lines)
        assert "pip cache purge" not in joined, (
            "pip cache purge will fail on Cloud Build (cache is disabled)"
        )
