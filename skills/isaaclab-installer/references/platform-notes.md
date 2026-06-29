# Platform Notes

Quirks the skill knows about and handles automatically. Listed here for reference and for users debugging unusual setups.

## aarch64 (DGX Spark)

- Architecture string: `aarch64`.
- PyTorch wheels come from `https://download.pytorch.org/whl/cu130` (not cu128).
- `imgui-bundle` and `quadprog` have no aarch64 wheels and are compiled from source. The combo's apt step adds the required headers automatically:

  ```bash
  sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev
  ```

- Common runtime warnings to ignore (or fix as described):
  - `ERROR: ld.so: object '.../libgomp-...so' cannot be preloaded: ignored.` — fix by exporting `LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1`.
  - `ImportError: .../libcarb.so: cannot allocate memory in static TLS block` — when invoking python directly, preload `libcarb.so`. `./isaaclab.sh -p` handles this automatically; the conda activation hook also does. Pure `uv` runs may need it set manually.

- Features that don't work on Spark (per `docs/source/setup/installation/index.rst`):
  - SkillGen (cuRobo dependency).
  - OpenXR teleoperation.
  - SKRL with JAX GPU (only CPU JAX is available).
  - Livestream and Hub Workstation Cache.
  - Cosmos Transfer1.
  - Newton VBD deformables (no `pytetwild` aarch64 wheel).

## Linux x86_64 — Ubuntu 22.04 (recommended)

- Primary supported platform.
- All combos work.
- Driver upgrade path: <https://www.nvidia.com/en-us/drivers/unix/>.

## Linux x86_64 — Ubuntu 20.04

- GLIBC 2.31. Pip Isaac Sim does NOT work — the recommender automatically picks a `binary-*` combo.
- All other combos work.

## Linux x86_64 — RHEL / Rocky 9 / Fedora

- GLIBC typically 2.34+. Pip Isaac Sim may work; the recommender will pick it if GLIBC ≥ 2.35.
- `apt-get` is not available — the combos that include the apt step will fail. The skill prompts before each sudo step; user can substitute `dnf install -y cmake gcc-c++ make` and continue. (A future v2 may auto-detect and translate.)

## Docker / containers

- `preflight.py` detects when it's running in a container (`/.dockerenv` or cgroup match) and surfaces this.
- If the container does not have `--gpus all` (or equivalent), the verify step will fail because nvidia-smi will be missing.
- Recommended: do the install in the host shell, mount the IsaacLab repo into the container.

## WSL2

- Detected via `WSL_DISTRO_NAME` or "microsoft" in the kernel release string.
- WSLg is required for any GUI smoke test. Headless verification works without it.
- Driver behavior differs from native Linux — install the **Windows** NVIDIA driver, not a Linux one. The WSL Linux kernel exposes the GPU automatically.

## Display vs headless

- The skill's default verification is **headless**. This is intentional: it works on servers and in containers, and it's a strict superset of "can Isaac Sim launch at all".
- If the user wants a viewport, the doctor and verify commands tell them how to retry with `--viz kit`.

## Network egress

`preflight.py` probes:

- `github.com:443` (clone)
- `pypi.org:443` (pip)
- `pypi.nvidia.com:443` (Isaac Sim wheels)
- `download.pytorch.org:443` (PyTorch wheels)

If any are unreachable the recommender warns. Users behind corporate proxies should configure `https_proxy` / `pip.conf` before running.
