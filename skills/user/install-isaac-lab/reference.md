# Installing Isaac Lab Reference

## Preflight Detection

Read-only commands to gather routing facts on Linux; nothing changes system state:

```bash
grep PRETTY_NAME /etc/os-release && uname -m && ldd --version | head -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
command -v python3.12 uv conda; free -g | head -2; df -h .
```

Also note existing install artifacts: `.venv/` or an env directory in the checkout, and `~/.isaaclab/install_profile.yaml`. On Windows, collect the equivalents (driver via `nvidia-smi`, Python version, free disk).

## Express Route Mapping

After preflight detection, read the minimums from `docs/source/setup/installation/index.rst` in the checkout, then route on the detected facts:

| Detected facts | Route | Docs page |
| --- | --- | --- |
| Linux, NVIDIA GPU, GLIBC at or above the documented pip minimum | pip (docs-Recommended) | `docs/source/setup/installation/pip_installation.rst` |
| Linux, NVIDIA GPU, GLIBC below the pip minimum or undetectable | binary Isaac Sim | `docs/source/setup/installation/binaries_installation.rst` |
| Windows 11 | guided per-tab flow | `docs/source/setup/installation/index.rst` |
| No NVIDIA GPU/driver, driver below minimum, or insufficient disk | blocked — report fixes, offer kit-less | `docs/source/setup/installation/index.rst` |

Distro version is not a routing key — GLIBC is. Ubuntu 22.04 (GLIBC 2.35) and Ubuntu 24.04 (GLIBC 2.39) both take the pip route because they meet the pip minimum. Ubuntu 20.04 (GLIBC 2.31) falls below the minimum and routes to binary. Read the current minimum from `docs/source/setup/installation/index.rst` for the OS support matrix rather than assuming a specific distro is on or off the list.

User-stated preferences override the routing and map directly:

| Stated preference | Docs page |
| --- | --- |
| Isaac Sim contributor building from source | `docs/source/setup/installation/source_installation.rst` |
| External extension author, Isaac Lab pip package only | `docs/source/setup/installation/isaaclab_pip_installation.rst` |
| Newton physics only, no Isaac Sim | `docs/source/setup/installation/kitless_installation.rst` |
| Zero-env experimental workflow | `docs/source/setup/installation/uv_run.rst` |
| Containerized deployment | `docs/source/deployment/docker.rst` |

## Express Flow Rules

- At most one question: the consolidated go/no-go before execution. The binary route adds one pause for the manual Isaac Sim download.
- Env manager: uv if present, else conda if present, else install uv via the docs pip page step. Never ask.
- Defaults: install into the current checkout, docs-default env name, docs-Recommended options. Never ask.
- Log every executed command and its output to `~/.isaaclab/logs/install-<timestamp>.log`.
- After success, write facts, route, and commands run to `~/.isaaclab/install_profile.yaml`.

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

## Install-Time Failure Routing

Apply at most one documented fix per failed step, retry once, then stop and hand off with the log path.

| Symptom during install | First reference |
| --- | --- |
| `GLIBC` version too low | `docs/source/setup/installation/binaries_installation.rst` (switch install method) |
| `nvidia-smi` missing or driver too old | `docs/source/setup/installation/index.rst` (driver minimums) |
| Network timeout fetching wheels | Retry the step once; then check proxy/firewall for github.com, pypi.org, pypi.nvidia.com, download.pytorch.org |
| Windows path-too-long errors | Windows tab of the chosen install page (long-path support) |
| Docker step fails with runtime error | `docs/source/deployment/docker.rst` (NVIDIA Container Toolkit setup) |
| Import fails after install completes | Hand off to `isaaclab-setup-troubleshooting` |

## Cross-Skill Routing

- Existing install artifacts detected by preflight, post-install import failures, backend setup, or other setup diagnostics belong in `isaaclab-setup-troubleshooting`.
- Environment authoring, task creation, and RL training belong in the environment and training skills. Route there after verification passes.
