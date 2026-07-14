# Installing Isaac Lab Reference

## Install Method Routing

Consult `docs/source/setup/installation/index.rst` from the Isaac Lab checkout first, then route the user to the specific install page for the chosen method.

| User context | Docs page |
| --- | --- |
| First-time full-feature user (docs-Recommended) | `docs/source/setup/installation/pip_installation.rst` |
| Older Linux distro (GLIBC below the pip minimum) | `docs/source/setup/installation/binaries_installation.rst` |
| Isaac Sim contributor building from source | `docs/source/setup/installation/source_installation.rst` |
| External Isaac Lab extension author using pip only | `docs/source/setup/installation/isaaclab_pip_installation.rst` |
| Newton physics only, no Isaac Sim | `docs/source/setup/installation/kitless_installation.rst` |
| Zero-env experimental workflow | `docs/source/setup/installation/uv_run.rst` |
| Containerized deployment | `docs/source/deployment/docker.rst` |

## Prerequisite Minimums

Read `docs/source/setup/installation/index.rst` from the checkout for current minimums rather than relying on cached values. Common gates:

- NVIDIA driver minimum (varies by platform; documented in the install index).
- Python version required by the current Isaac Sim series.
- GLIBC minimum for pip-based Isaac Sim install on Linux.
- Visual Studio Build Tools 2022 or newer for Windows source builds.
- Docker Engine, Docker Compose, and NVIDIA Container Toolkit for the Docker path.

## Minimal Verification

Run the docs-defined minimal verification command after every install, before larger tests.

For uv-managed environments (most paths):

```bash
uv run python -c "import isaaclab; print('ok')"
```

For the binary Isaac Sim path with bundled Python, use the install page's documented verification instead. For Docker, run verification inside the container as documented in `docs/source/deployment/docker.rst`.

## Common Install-Time Failure Routing

| Symptom during install | First reference |
| --- | --- |
| `GLIBC` version too low | `docs/source/setup/installation/binaries_installation.rst` (switch install method) |
| `nvidia-smi` missing or driver too old | `docs/source/setup/installation/index.rst` (driver minimums) |
| Windows path-too-long errors | Windows tab of the chosen install page (long-path support) |
| Docker step fails with runtime error | `docs/source/deployment/docker.rst` (NVIDIA Container Toolkit setup) |
| Import fails after install completes | Hand off to `isaaclab-setup-troubleshooting` |

## Cross-Skill Routing

- Post-install import failures, backend setup, or other setup diagnostics belong in `isaaclab-setup-troubleshooting`. Route there as soon as the install commands finish and the failure is not in a documented install step.
- Environment authoring, task creation, and RL training belong in the environment and training skills. Route there after verification passes.
