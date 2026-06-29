# Compatibility Matrix

Pinned versions and the minimums for each install combination. These are sourced from `docs/source/setup/installation/` in this repository. Update them in `resources/combos.py` (the canonical location) when Isaac Lab bumps support.

## Pins (Isaac Lab 3.x / Isaac Sim 6.x)

| Component         | Pinned value                              |
| ----------------- | ----------------------------------------- |
| Isaac Sim         | `6.0.1` (pypi wheel `6.0.1.0`)            |
| Newton physics    | `1.2.1`                                   |
| Python            | `3.12`                                    |
| PyTorch (x86_64)  | `torch==2.10.0 torchvision==0.25.0` (cu128) |
| PyTorch (aarch64) | `torch==2.10.0 torchvision==0.25.0` (cu130) |
| NVIDIA driver (x86_64) | `>= 580.95.05`                       |
| NVIDIA driver (aarch64 / DGX Spark) | `>= 580.142`              |
| NVIDIA driver (Windows) | `>= 581.42.00` *(out of scope for v1)* |
| GLIBC (pip Isaac Sim only) | `>= 2.35`                        |

## Hardware minimums

| Resource | Recommended    |
| -------- | -------------- |
| RAM      | 32 GB          |
| VRAM     | 16 GB (more for rendering) |
| Disk     | 30 GB free (80 GB for source builds) |

## Distros

| Distro            | Recommended combo            | Notes                                       |
| ----------------- | ---------------------------- | ------------------------------------------- |
| Ubuntu 22.04 LTS  | `pip-uv-source`              | First-class supported.                      |
| Ubuntu 24.04      | `pip-uv-source`              | Works; less battle-tested.                  |
| Ubuntu 20.04      | `binary-uv-source` (or conda) | GLIBC 2.31 — pip Isaac Sim won't run.      |
| RHEL / Rocky 9    | `binary-*`                   | pip Isaac Sim may work; binary is safest.   |
| Debian 12         | Try `pip-*`; fall back to binary | GLIBC 2.36 — should work.               |

## Hard requirements per combo

| Combo                  | Arch          | Min GLIBC | Min driver  | Sudo? | Manual download? | Disk |
| ---------------------- | ------------- | --------- | ----------- | ----- | ---------------- | ---- |
| `pip-uv-source`        | x86_64 / arm64 | 2.35      | 580.95.05   | yes   | no               | 30 G |
| `pip-conda-source`     | x86_64 / arm64 | 2.35      | 580.95.05   | yes   | no               | 30 G |
| `pip-venv-source`      | x86_64 / arm64 | 2.35      | 580.95.05   | yes   | no               | 30 G |
| `binary-uv-source`     | x86_64        | —         | 580.95.05   | yes   | yes              | 35 G |
| `binary-conda-source`  | x86_64        | —         | 580.95.05   | yes   | yes              | 35 G |
| `binary-venv-source`   | x86_64        | —         | 580.95.05   | yes   | yes              | 35 G |
| `source-uv-source`     | x86_64        | —         | 580.95.05   | yes   | no               | 80 G |
| `source-conda-source`  | x86_64        | —         | 580.95.05   | yes   | no               | 80 G |
| `pip-only-uv`          | x86_64 / arm64 | 2.35      | 580.95.05   | no    | no               | 20 G |
| `pip-only-conda`       | x86_64 / arm64 | 2.35      | 580.95.05   | no    | no               | 20 G |
| `pip-only-venv`        | x86_64 / arm64 | 2.35      | 580.95.05   | no    | no               | 20 G |
| `kitless-uv`           | x86_64 / arm64 | —         | 580.95.05   | yes   | no               | 15 G |
| `kitless-conda`        | x86_64 / arm64 | —         | 580.95.05   | yes   | no               | 15 G |

## Which Isaac Sim works with which Isaac Lab?

This repo currently targets Isaac Sim **6.0.1** on the Isaac Lab `release/3.0.0-beta2` branch. Isaac Sim versions ≤ 5.1 are no longer supported.

The pip wheel string is `isaacsim[all,extscache]==6.0.1.0` (the `.0` suffix is the wheel revision; humans see "6.0.1"). When the wheel revision bumps (e.g. `6.0.1.1`), update `DEFAULT_ISAACSIM_VERSION` in `resources/combos.py`.

> A recent breaking change on the Isaac Lab `develop` branch is not compatible with the `develop` branch of Isaac Sim on GitHub. To run Isaac Lab against the Isaac Sim GitHub `develop` branch, use Isaac Lab commit `f0234a82e432e2a0b0f0a26ca3c5b59e527ddaaa` or earlier, or the `v3.0.0-beta` tag.

## DGX Spark (aarch64) specifics

- Architecture: aarch64.
- Driver: 580.142+.
- PyTorch index: cu130.
- Extra apt packages are mandatory for `imgui-bundle` / `quadprog` to build from source (no aarch64 wheels published):
  `sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev`.
- Some features are not supported on Spark; see `docs/source/setup/installation/index.rst` for the full list.
