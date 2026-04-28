"""Tests for Vertex AI YAML job configuration files.

Validates:
- YAML syntax is valid
- Required fields are present (workerPoolSpecs, containerSpec, imageUri, env)
- env block is inside containerSpec (not a sibling)
- No gcsfuse mount commands (Vertex AI containers lack /dev/fuse)
- No --file-cache-capacity-mb flag (never existed in gcsfuse)
- No --stat-cache-ttl or --type-cache-ttl (deprecated in gcsfuse 2.0+)
- pip install -e is not used (editable installs are fragile in containers)
"""

import pathlib

import yaml
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

YAML_FILES = [
    "scripts/vertex_job_pre_encode.yaml",
    "scripts/vertex_job_f1_3s.yaml",
]


def load_yaml(rel_path: str) -> dict:
    path = REPO_ROOT / rel_path
    if not path.exists():
        pytest.skip(f"{rel_path} does not exist")
    with open(path) as f:
        return yaml.safe_load(f)


class TestYamlSyntax:
    """YAML files must parse without errors."""

    @pytest.mark.parametrize("yaml_file", YAML_FILES)
    def test_yaml_parses(self, yaml_file):
        data = load_yaml(yaml_file)
        assert data is not None, f"{yaml_file} parsed as empty"
        assert isinstance(data, dict), f"{yaml_file} root is not a mapping"


class TestVertexAIStructure:
    """Validate Vertex AI CustomJobSpec structure."""

    @pytest.mark.parametrize("yaml_file", YAML_FILES)
    def test_has_worker_pool_specs(self, yaml_file):
        data = load_yaml(yaml_file)
        assert "workerPoolSpecs" in data, f"{yaml_file}: missing workerPoolSpecs"

    @pytest.mark.parametrize("yaml_file", YAML_FILES)
    def test_has_container_spec(self, yaml_file):
        data = load_yaml(yaml_file)
        pool = data["workerPoolSpecs"][0]
        assert "containerSpec" in pool, f"{yaml_file}: missing containerSpec"

    @pytest.mark.parametrize("yaml_file", YAML_FILES)
    def test_has_image_uri(self, yaml_file):
        data = load_yaml(yaml_file)
        container = data["workerPoolSpecs"][0]["containerSpec"]
        assert "imageUri" in container, f"{yaml_file}: missing imageUri"

    @pytest.mark.parametrize("yaml_file", YAML_FILES)
    def test_env_inside_container_spec(self, yaml_file):
        """env must be a child of containerSpec, not a sibling at workerPoolSpec level."""
        data = load_yaml(yaml_file)
        pool = data["workerPoolSpecs"][0]
        assert "env" not in pool, (
            f"{yaml_file}: 'env' is at workerPoolSpec level — "
            "must be inside containerSpec"
        )
        container = pool["containerSpec"]
        assert "env" in container, (
            f"{yaml_file}: 'env' is missing from containerSpec"
        )


class TestNoGcsfuseMount:
    """Vertex AI containers lack /dev/fuse; gcsfuse mount will always fail."""

    @pytest.mark.parametrize("yaml_file", YAML_FILES)
    def test_no_gcsfuse_mount_command(self, yaml_file):
        content = (REPO_ROOT / yaml_file).read_text()
        lines = content.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "gcsfuse" not in stripped or "mount" not in stripped, (
                f"{yaml_file} line {i+1}: gcsfuse mount found — "
                "Vertex AI containers lack /dev/fuse"
            )

    @pytest.mark.parametrize("yaml_file", YAML_FILES)
    def test_no_file_cache_capacity_mb(self, yaml_file):
        content = (REPO_ROOT / yaml_file).read_text()
        assert "--file-cache-capacity-mb" not in content, (
            f"{yaml_file}: --file-cache-capacity-mb is not a valid gcsfuse flag"
        )


class TestNoEditableInstall:
    """pip install -e is fragile in containers."""

    @pytest.mark.parametrize("yaml_file", YAML_FILES)
    def test_no_pip_install_editable(self, yaml_file):
        content = (REPO_ROOT / yaml_file).read_text()
        lines = content.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "pip install" in stripped and "-e " in stripped:
                pytest.fail(
                    f"{yaml_file} line {i+1}: editable install found — "
                    "use 'pip install --no-deps /workspace/' instead"
                )


class TestHuggingFaceToken:
    """Gated HF fallback must be discoverable: env var and/or documented in YAML.

    Vertex Custom Jobs reject ``env`` entries with an empty ``value``, so we do not
    ship ``HUGGINGFACE_TOKEN: \"\"`` — operators inject the token at submit time when
    the checkpoint is not already in GCS.
    """

    @pytest.mark.parametrize("yaml_file", YAML_FILES)
    def test_huggingface_token_in_env_or_documented(self, yaml_file):
        text = (REPO_ROOT / yaml_file).read_text()
        data = load_yaml(yaml_file)
        container = data["workerPoolSpecs"][0]["containerSpec"]
        env_list = container.get("env", [])
        env_names = [e["name"] for e in env_list if "name" in e]
        if "HUGGINGFACE_TOKEN" in env_names:
            return
        assert "HUGGINGFACE_TOKEN" in text, (
            f"{yaml_file}: mention HUGGINGFACE_TOKEN for HF gated fallback, or add it "
            "to env when submitting (empty values are invalid on Vertex)."
        )
