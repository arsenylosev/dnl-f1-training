"""Contract tests for Vertex ML jobs and Docker training image paths.

Catches failures where CI passes but Vertex fails (e.g. missing model JSON excluded
from Docker context) — see README_DNL / .github/workflows/ci.yml notes.
"""

import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Paths referenced by Vertex YAML / scripts must exist in-repo and ship in Docker context.
VERTEX_REQUIRED_FILES = [
    "models/foundation1_3s/model_config_3s.json",
    "stable_audio_tools/configs/dataset_configs/gcs_f1_3s_pre_encode.json",
    "stable_audio_tools/configs/dataset_configs/pre_encoded_f1_3s.json",
]

# Each Vertex job YAML must mention these substrings (dataset differs per job).
VERTEX_YAML_FRAGMENTS = {
    "scripts/vertex_job_pre_encode.yaml": (
        "models/foundation1_3s/model_config_3s.json",
        "gcs_f1_3s_pre_encode.json",
    ),
    "scripts/vertex_job_f1_3s.yaml": (
        "models/foundation1_3s/model_config_3s.json",
        "pre_encoded_f1_3s.json",
    ),
}


class TestVertexArtifactPathsExist:
    def test_required_files_exist(self):
        missing = [p for p in VERTEX_REQUIRED_FILES if not (REPO_ROOT / p).is_file()]
        assert not missing, f"Missing files required by Vertex jobs: {missing}"

    def test_model_config_json_is_valid_diffusion_cond(self):
        path = REPO_ROOT / "models/foundation1_3s/model_config_3s.json"
        data = json.loads(path.read_text())
        assert data.get("model_type") == "diffusion_cond"
        assert "model" in data
        assert data.get("sample_rate") == 44100
        assert data.get("training", {}).get("pre_encoded") is True

    def test_pre_encoded_dataset_config(self):
        path = REPO_ROOT / "stable_audio_tools/configs/dataset_configs/pre_encoded_f1_3s.json"
        data = json.loads(path.read_text())
        assert data.get("dataset_type") == "pre_encoded"
        assert data["datasets"][0]["path"] == "/workspace/pre_encoded_3s"


class TestVertexYamlReferencesResolvableFiles:
    """YAML command blocks mention paths — ensure those files exist."""

    @pytest.mark.parametrize("yaml_rel", list(VERTEX_YAML_FRAGMENTS.keys()))
    def test_yaml_contains_expected_path_fragments(self, yaml_rel):
        text = (REPO_ROOT / yaml_rel).read_text()
        for fragment in VERTEX_YAML_FRAGMENTS[yaml_rel]:
            assert fragment in text, f"{yaml_rel} must mention {fragment!r}"


class TestCiDockerContextVerifyPresent:
    """Thin Dockerfile used on every PR to mirror Cloud Build context rules."""

    def test_context_verify_dockerfile_exists(self):
        path = REPO_ROOT / "ci" / "Dockerfile.context-verify"
        assert path.is_file(), "Add ci/Dockerfile.context-verify (see .github/workflows/ci.yml)"


class TestDockerignoreAllowsFoundationConfig:
    """Regression: blanket /models/ excluded model_config_3s.json from images."""

    def test_dockerignore_has_models_exception(self):
        di = (REPO_ROOT / ".dockerignore").read_text()
        assert "model_config_3s.json" in di, (
            ".dockerignore must allow models/foundation1_3s/model_config_3s.json "
            "(see exception patterns near /models/)"
        )
        assert "/models/**" in di or "/models/" in di, (
            ".dockerignore should exclude heavy files under /models/ with a root-anchored pattern"
        )
