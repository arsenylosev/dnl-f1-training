"""Shared fixtures for the DNL F1 training test suite."""

import pathlib

import pytest

# Root of the repository
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> pathlib.Path:
    """Return the absolute path to the repository root."""
    return REPO_ROOT


def get_repo_root() -> pathlib.Path:
    """Module-level helper — use this instead of importing REPO_ROOT directly."""
    return REPO_ROOT
