# Agent Guidelines for DNL F1 Training Pipeline

This document provides comprehensive context and guidelines for AI agents working on the `dnl-f1-training` repository. It outlines the repository structure, CI/CD pipeline, training infrastructure, and best practices to ensure consistency and prevent regressions.

## 1. Repository Overview

The `dnl-f1-training` repository contains the MLOps infrastructure and configuration for fine-tuning the Foundation-1 (F1) Diffusion Transformer (DiT) model on Google Cloud Vertex AI.

### Key Directories and Files

- **`scripts/`**: Contains the core Vertex AI custom job YAML configurations.
  - `vertex_job_pre_encode.yaml`: Configuration for the VAE pre-encoding step.
  - `vertex_job_f1_3s.yaml`: Configuration for the main DiT fine-tuning step.
  - `gcp_train_f1_3s.sh`: Legacy/alternative shell script for local or direct VM training.
- **`models/`**: Contains model-specific configurations.
  - `foundation1_3s/model_config_3s.json`: The primary model configuration file defining architecture, conditioning, and training parameters.
- **`requirements/`**: Python dependency specifications.
  - `base.txt`: Core dependencies.
  - `train.txt`: Training-specific dependencies.
  - `encode.txt`: Pre-encoding specific dependencies.
- **`tests/`**: PyTest suite for structural validation.
  - `test_yaml_configs.py`: Validates YAML structure, preventing common Vertex AI configuration errors.
  - `test_docker_build.py`: Validates the Docker build process.
- **`.github/workflows/`**: CI/CD automation.
  - `ci.yml`: GitHub Actions workflow for linting, testing, and Docker build verification.
- **`Dockerfile`**: The unified container definition used for both pre-encoding and training on Vertex AI.

## 2. Infrastructure and Compute Configuration

The training pipeline is designed to run on Google Cloud Vertex AI Custom Training Jobs in the `europe-west4` region.

### Hardware Selection Strategy

Due to quota constraints and high costs associated with H100 GPUs, the pipeline is optimized for more accessible and cost-effective alternatives:

1.  **Pre-Encoding (`vertex_job_pre_encode.yaml`)**:
    -   **Target VM**: `a2-highgpu-1g` (1× NVIDIA A100 40GB).
    -   **Rationale**: A100 40GB is widely available in `europe-west4-a` and `europe-west4-b` without special quota requests. It provides sufficient VRAM for the pre-encoding batch size (32) at a reasonable cost (~$4.42/hr on-demand).
    -   **Alternative**: `g2-standard-16` (1× NVIDIA L4 24GB) can be used for further cost reduction (~$1.25/hr), requiring a batch size reduction to 16.

2.  **Fine-Tuning (`vertex_job_f1_3s.yaml`)**:
    -   **Target VM**: `g2-standard-48` (2× NVIDIA L4 24GB).
    -   **Rationale**: Replaces the original `a3-highgpu-2g` (2× H100 80GB) configuration. L4 GPUs are readily available in `europe-west4-b` without special quota. This configuration reduces costs by ~5× (on-demand) to ~14× (Spot) compared to H100s.
    -   **Adjustments**: Batch size per GPU is reduced from 16 to 8 to accommodate the 24GB VRAM limit of the L4 (compared to 80GB on H100). The number of workers is adjusted to 6.

## 3. CI/CD Pipeline (`ci.yml`)

The GitHub Actions CI pipeline is critical for maintaining repository integrity. It consists of two main jobs:

1.  **`lint-and-test`**: Runs on every push and PR. Executes the PyTest suite (`tests/`) to validate YAML configurations, dependency consistency, and Docker context inclusion.
2.  **`docker-build`**: Runs **only on pushes to the `main` branch**. Builds the full Docker image and performs sanity checks, including verifying critical imports and ensuring Vertex pre-encode paths exist within the image.

### Known CI Pitfalls and Solutions

-   **Apt-Get 404 Errors**: The NVIDIA PyTorch base image (Ubuntu 22.04) occasionally encounters transient `404 Not Found` errors during `apt-get install` due to stale security mirrors.
    -   *Solution*: The `Dockerfile` must use `apt-get install -y --no-install-recommends --fix-missing` to ensure resilience against these transient mirror issues.
-   **Shell Quoting in YAML**: Inline Python scripts within the `ci.yml` `run` blocks (which use `bash -c '...'`) must avoid using literal single quotes (`'`) for dictionary access or string literals, as this terminates the outer bash string and causes syntax errors (e.g., `NameError`).
    -   *Solution*: Always use double quotes (`"`) or properly escaped quotes within inline Python scripts in the CI YAML.

## 4. Agent Workflow Guidelines

When modifying this repository, agents MUST adhere to the following guidelines:

1.  **Test-Driven Modifications**: Before changing configuration files, review the tests in `tests/test_yaml_configs.py`. Ensure modifications do not violate established structural rules (e.g., no `gcsfuse` mounts, proper environment variable placement). Run `pytest tests/` locally to verify.
2.  **Surgical Changes**: Follow the Karpathy Guidelines. Touch only what is necessary to accomplish the goal. Do not perform unsolicited refactoring of adjacent code or configurations.
3.  **Cost Awareness**: Always consider the cost implications of compute configurations. Default to the most cost-effective GPU option (e.g., L4) that meets the technical requirements, and document the rationale in the YAML comments.
4.  **CI Verification**: After making changes, verify that the CI pipeline passes. If the `docker-build` job fails, investigate the logs carefully, keeping in mind the known pitfalls mentioned above.
5.  **Documentation**: Update this `AGENTS.md` file if new architectural decisions, CI workarounds, or infrastructure strategies are introduced.
