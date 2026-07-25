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

After preflight detection, read the "System requirements" and per-method sections from `docs/source/setup/installation/index.rst` in the checkout, then route on the detected facts to a section anchor:

| Detected facts | Route | Docs section |
| --- | --- | --- |
| Linux, NVIDIA GPU, GLIBC at or above the documented pip minimum | automatic uv from checkout (docs-Recommended) | `docs/source/setup/installation/index.rst` (`installation-method-uv`) |
| Linux, NVIDIA GPU, GLIBC below the pip minimum or undetectable | downloaded Isaac Sim package | `docs/source/setup/installation/index.rst` (`installation-method-binary`) |
| Windows 11 | guided per-tab flow via the automatic uv or managed-env sections | `docs/source/setup/installation/index.rst` (`installation-method-uv` or `installation-method-python-env`) |
| No NVIDIA GPU/driver, driver below minimum, or insufficient disk | blocked — report fixes, offer the legacy Newton-only installer as the no-Isaac-Sim alternative | `docs/source/setup/installation/index.rst` (`installation-legacy-installer`) |

Distro version is not a routing key — GLIBC is. Ubuntu 22.04 (GLIBC 2.35) and Ubuntu 24.04 (GLIBC 2.39) both take the automatic uv route because they meet the pip minimum. Ubuntu 20.04 (GLIBC 2.31) falls below the minimum and routes to the downloaded Isaac Sim package. Read the current minimum from `docs/source/setup/installation/index.rst` for the OS support matrix rather than assuming a specific distro is on or off the list.

User-stated preferences override the routing and map directly:

| Stated preference | Docs section |
| --- | --- |
| Isaac Sim contributor building from source | `docs/source/setup/installation/index.rst` (`installation-method-source`) |
| External extension author, Isaac Lab wheel only | `docs/source/setup/installation/index.rst` (`installation-method-wheel`) |
| Newton-only workflow, no Isaac Sim | `docs/source/setup/installation/index.rst` (`installation-legacy-installer`) |
| Managed venv or conda with pip Isaac Sim | `docs/source/setup/installation/index.rst` (`installation-method-python-env`) |
| Containerized deployment | `docs/source/setup/installation/index.rst` (`installation-method-container`); deep-dive `docs/source/features/docker_cloud.rst` |
| Cloud-hosted GPU workstation | `docs/source/setup/installation/index.rst` (`installation-method-cloud`) |

## Express Flow Rules

- At most one question: the consolidated go/no-go before execution. The downloaded-Isaac-Sim route adds one pause for the manual package download.
- Env manager: uv if present, else conda if present, else install uv via the docs `installation-method-uv` step. Never ask.
- Defaults: install into the current checkout, docs-default env name, docs-Recommended options. Never ask.
- Log every executed command and its output to `~/.isaaclab/logs/install-<timestamp>.log`.
- After success, write facts, route, and commands run to `~/.isaaclab/install_profile.yaml`.

## Prerequisite Minimums

Read `docs/source/setup/installation/index.rst` "System requirements" from the checkout for current minimums rather than relying on cached values. Common gates:

- NVIDIA driver minimum (varies by platform; documented in the "System requirements" section of `index.rst`).
- Python 3.12 for the current Isaac Sim series.
- GLIBC minimum for pip-based Isaac Sim install on Linux.
- Ubuntu 22.04 or newer for the Isaac Sim source build (Linux only).
- Windows long-path support for any Windows install path.
- Docker Engine, Docker Compose, and NVIDIA Container Toolkit for the Docker path.

## Minimal Verification

Run the docs-defined minimal verification command after every install, before larger tests. The command varies by route:

- Automatic uv (`installation-method-uv`), legacy installer (`installation-legacy-installer`), managed Python env (`installation-method-python-env`), Isaac Lab wheel (`installation-method-wheel`), and Isaac Sim source build (`installation-method-source`) verify Isaac Lab via the tutorial script documented in the section's included verification snippet:

```bash
uv run python scripts/tutorials/00_sim/create_empty.py --viz kit
```

- Downloaded Isaac Sim package (`installation-method-binary`) uses the bundled-Python verification documented in the section (launch via `${ISAACSIM_PATH}/isaac-sim.sh`, then run the tutorial script from the checkout).
- Docker (`installation-method-container`) runs the same tutorial verification inside the container as documented in `docs/source/features/docker_cloud.rst`.

## Install-Time Failure Routing

Apply at most one documented fix per failed step, retry once, then stop and hand off with the log path.

| Symptom during install | First reference |
| --- | --- |
| `GLIBC` version too low | `docs/source/setup/installation/index.rst` (`installation-method-binary` — switch to the downloaded Isaac Sim package) |
| `nvidia-smi` missing or driver too old | `docs/source/setup/installation/index.rst` ("System requirements" — driver minimums) |
| Network timeout fetching wheels | Retry the step once; then check proxy/firewall for github.com, pypi.org, pypi.nvidia.com, download.pytorch.org |
| Windows path-too-long errors | Windows tab of the chosen section (long-path support) |
| Docker step fails with runtime error | `docs/source/features/docker_cloud.rst` (NVIDIA Container Toolkit setup) |
| Import fails after install completes | Hand off to `isaaclab-setup-troubleshooting` |

## Cross-Skill Routing

- Existing install artifacts detected by preflight, post-install import failures, backend setup, or other setup diagnostics belong in `isaaclab-setup-troubleshooting`.
- Environment authoring, task creation, and RL training belong in the environment and training skills. Route there after verification passes.
