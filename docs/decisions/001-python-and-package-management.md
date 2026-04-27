# ADR-001: Python version matrix and package management

## Status
Accepted (2026-04-27)

## Context

- The upstream [RC-stable-audio-tools](https://github.com/RoyalCities/RC-stable-audio-tools) README was written when **Python 3.10** and tight pins (e.g. older `scipy` wheels) were the norm, and it warns that **3.11+** can break resolution.
- This DNL fork **relaxed** the `scipy` constraint in `setup.py` and `requirements/` (`scipy>=1.8.1,<1.14`) specifically so pre-built wheels exist on **Python 3.11+**, matching current Debian 12 and Ubuntu 22+ defaults used by our VM bootstrap.
- The **Vertex training Docker image** uses the NVIDIA NGC `pytorch:24.04` base, which currently ships **Python 3.10** — a different, intentional pin driven by the CUDA/NGC support matrix, not by library incompatibility in our application code.
- The repository is packaged with **setuptools** via `setup.py` (runtime dependency list) plus **PEP 621** metadata in `pyproject.toml` (name, version, `tool.setuptools.packages.find`). The Docker image installs dependencies from `requirements/*.txt` and **copies** (not `pip install -e`) the `stable_audio_tools` tree into `site-packages` to avoid known container edge cases; local workflows may still use `pip install -e . --no-deps` after the same `requirements/`.

**Alternatives considered (uv / single manifest)**

| Approach | Pros | Cons |
|----------|------|------|
| Keep `pip` + `requirements/*.txt` + `setup.py` in CI/Docker | One story for production; matches NGC/Vertex; already works | Slower local installs; no built-in lock by default |
| **Optional `uv` for local dev** (`uv venv`, `uv pip install -r ...`) | Much faster resolution/install; can pin interpreter with `uv python pin` / `-p` | Team must not confuse “local venv” with the **3.10** image inside Docker |
| Full `uv` project with `uv.lock` as the only source of truth | Reproducible by hash | **Large migration** — need to move `install_requires` from `setup.py` into `pyproject.toml` (or generate), change CI/Docker, and revalidate GPU stacks; not justified while Docker remains pip-based |
| **Deprecation** of `setup.py` | Cleaner single file | `install_requires` still lives in `setup.py` today; removing it without a completed migration to `[project] dependencies` would break conventions and `pip install .`; defer until a dedicated effort |

## Decision

1. **Python support band (this fork):** `requires-python = ">=3.8"` in `pyproject.toml`. **First-class** validation targets are **3.10** (Docker/NGC) and **3.11** (GitHub Actions `lint-and-test`, GCE `bootstrap_vm.sh` / `setup_python_env.sh` default). We run the full non-Docker `pytest` suite on **both**; packaging builds (`pip wheel` / `pip install`, `--no-deps`) are verified to succeed on 3.11+ for the sdist/bdist of `stable-audio-tools`.
2. **Documentation:** Root `README.md` retains upstream text where possible; a pointer to this ADR and to `README_DNL.md` clarifies the fork. **Do not** treat the upstream "use only 3.10" note as the single source of truth for DNL.
3. **`uv` (adoption):** Treat **`uv` as an optional** accelerator for **local** environments, not a replacement for the production install path. Recommended pattern:
   - `uv venv -p 3.11` (or 3.10) then `uv pip install -U pip setuptools wheel` and install `requirements/*.txt` in the same order as `setup_python_env.sh` / the Dockerfile, then `pip install -e . --no-deps` (or a wheel of the project). This preserves **one** source of requirements (`setup.py` + `requirements/`) and avoids a half-migrated `uv.lock` that drifts from Docker.
4. **No compulsory deprecation** of `setup.py` in this change set; if we later standardize on `uv.lock`, do it as a **gated** migration with CI + Docker both switched and `install_requires` consolidated into `pyproject` under review.

## Consequences

- Engineers may use **Python 3.11** locally and in CI; **3.10** remains what runs in the NGC training container until the base image changes.
- Optional `uv` usage is documented; production remains **pip + `requirements/*.txt` + Dockerfile copy step**.
- New dependency pins still belong in `setup.py` / `requirements/*.txt` until a lockfile migration is completed end-to-end.

## Verification (repeatable)

```bash
# With uv and Python 3.11 and 3.10 interpreters available:
uv venv /tmp/venv-311 -p 3.11
uv pip install --python /tmp/venv-311/bin/python pip pytest pyyaml
# From repo root:
uv run --python /tmp/venv-311/bin/python -m pytest tests/ --ignore=tests/test_docker_build.py

# Or after: uv pip install --python /tmp/venv-311/bin/python pip setuptools wheel
# /tmp/venv-311/bin/python -m pip wheel . --no-deps -w /tmp/wheel-out
```

CI already runs the same test suite (excluding the optional long Docker test) on **3.11**; run the `uv` commands above to confirm locally.
