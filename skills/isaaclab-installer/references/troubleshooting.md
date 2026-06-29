# Troubleshooting

Symptom → likely cause → fix. Use `scripts/doctor.py` to diagnose automatically.

## Imports

### `ModuleNotFoundError: No module named 'isaacsim'`

- **Cause**: Environment not activated, or the `install_isaacsim_pip` step was skipped, or you cloned the repo but never ran `./isaaclab.sh -i`.
- **Fix**:
  ```bash
  # Activate the env (uv)
  source $ISAACLAB_DIR/env_isaaclab/bin/activate
  # Or (conda)
  conda activate env_isaaclab
  # Then re-run the Isaac Sim install
  uv pip install "isaacsim[all,extscache]==6.0.1.0" \
      --extra-index-url https://pypi.nvidia.com \
      --index-strategy unsafe-best-match --prerelease=allow
  ```

### `ModuleNotFoundError: No module named 'isaaclab'`

- **Cause**: `./isaaclab.sh -i` never finished, or you're using a different Python from the one the install ran with.
- **Fix**: re-run from the IsaacLab repo root with the env activated:
  ```bash
  ./isaaclab.sh -i all
  ```

### `ImportError: .../libcarb.so: cannot allocate memory in static TLS block`

- **Cause**: aarch64 specific — `libcarb.so` uses initial-exec TLS and the linker's TLS surplus is exhausted before it loads.
- **Fix**: preload `libcarb.so` before launching python. `./isaaclab.sh -p` handles this; for direct `python` invocations on aarch64:
  ```bash
  export LD_PRELOAD=$(python -c "import sys,os;[print(os.path.join(p,'omni','client','libcarb.so')) for p in sys.path if os.path.isfile(os.path.join(p,'omni','client','libcarb.so'))]" 2>/dev/null | head -1)${LD_PRELOAD:+:$LD_PRELOAD}
  ```

### `ld.so: object '.../libgomp-...so' cannot be preloaded: ignored.`

- **Cause**: aarch64 specific — system and PyTorch each bundle a `libgomp`.
- **Fix**:
  ```bash
  unset LD_PRELOAD
  export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
  ```

## GLIBC

### `version 'GLIBC_2.35' not found`

- **Cause**: pip-installed Isaac Sim needs GLIBC ≥ 2.35; your distro is older (e.g. Ubuntu 20.04).
- **Fix**: switch to a `binary-*` combo. The recommender does this automatically once it sees the GLIBC version.

## Driver / CUDA

### `nvidia-smi: command not found`

- **Cause**: No NVIDIA driver installed (or you're inside a container without GPU passthrough).
- **Fix**: install the latest production-branch driver from <https://www.nvidia.com/en-us/drivers/unix/> using the `.run` installer. For containers, ensure `--gpus all` (Docker) or `--device nvidia.com/gpu=all` (Podman).

### Driver too old (`580.x < 580.95.05`)

- **Cause**: docs require >= 580.95.05 (x86_64) / >= 580.142 (aarch64).
- **Fix**: upgrade. The recommender will block until you do, but you can override (`--force`, not implemented yet — for now, upgrade is mandatory).

### PyTorch reports CUDA unavailable

- **Cause**: wrong wheel architecture; cu128 wheel on aarch64 (need cu130), or CPU-only torch sneaked in.
- **Fix**: reinstall with the correct index URL. The skill's `install_torch` step uses the right URL for the arch.

## Environment management

### `conda activate` says "command not found"

- **Cause**: conda not initialized in the current shell.
- **Fix**: `source ~/miniconda3/etc/profile.d/conda.sh` (path may vary). Or restart your shell after running `conda init`.

### `uv: command not found`

- **Cause**: uv not on `PATH`.
- **Fix**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # restart shell or:
  source ~/.bashrc
  ```

### Wrong Python in the env

- **Cause**: env created with Python 3.10 or 3.11 instead of 3.12.
- **Fix**: delete the env and recreate it with the right Python:
  ```bash
  rm -rf $ISAACLAB_DIR/env_isaaclab
  uv venv --python 3.12 --seed $ISAACLAB_DIR/env_isaaclab
  ```

## Filesystem

### `_isaac_sim` is a dangling symlink

- **Cause**: you moved or deleted the Isaac Sim directory; the link still points at the old path.
- **Fix**:
  ```bash
  cd $ISAACLAB_DIR
  ln -sfn /current/path/to/isaacsim _isaac_sim
  ```

### Disk full during install

- **Cause**: source builds need ~80 GB; binary combos need ~35 GB; pip combos need ~30 GB.
- **Fix**: prune docker images / pip cache, or install to a different filesystem with `--isaaclab-dir`.

## Build failures (aarch64)

### `imgui-bundle` / `quadprog` build fails with missing `OPENGL_*` libraries

- **Cause**: missing OpenGL / X11 dev headers (no aarch64 prebuilt wheels exist).
- **Fix**:
  ```bash
  sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev
  ```

## Verify step

### Headless verify exits non-zero but install seems fine

- **Cause**: container without GPU access, or no driver, or env not activated for the verify shell.
- **Fix**: re-run with the env activated; if running in Docker, ensure `--gpus all`.

### `EULA prompt blocks first run`

- **Cause**: Isaac Sim shows the NVIDIA Omniverse EULA the first time you launch it.
- **Fix**: it's interactive — type `Yes` when prompted. The skill currently does NOT auto-accept; users must accept it themselves.

## Remote install (SSH) issues

### `Remote install requires the 'paramiko' package`

- **Cause**: paramiko is not installed on the workstation running the skill.
- **Fix**: `pip install --user paramiko`, or `sudo apt install python3-paramiko`.

### `Authentication failed for user@host`

- **Cause**: wrong SSH password, or password auth disabled on the remote sshd.
- **Fix**: confirm `PasswordAuthentication yes` in `/etc/ssh/sshd_config` on the remote, or use `ssh-copy-id` first and then connect (the skill still prompts for a password but you can press Enter — paramiko will fall back gracefully only if password auth is allowed; otherwise see "Key auth not supported in v1" below).

### `Could not reach host:port`

- **Cause**: host is unreachable / firewall blocking port 22.
- **Fix**: confirm `ssh user@host` works from the same workstation first.

### Sudo step hangs on remote

- **Cause**: remote sudo is prompting for a password but the skill thinks the user is passwordless.
- **Fix**: re-run; when the second prompt asks for the sudo password, enter it (don't press Enter).

### Remote disk full mid-install

- **Cause**: the remote host's `$HOME` doesn't have enough free space.
- **Fix**: pass `--isaaclab-dir /mnt/big/IsaacLab` (or similar) when planning. The plan resolves all paths in advance using the remote `$HOME` from preflight.

### Key auth not supported in v1

- **By design**: v1 of remote mode uses password auth only. If your remote requires key-only authentication, either set up `PasswordAuthentication yes` temporarily, or run the skill in local mode by SSH-ing to the host yourself first.

## Filing an issue

Before opening a GitHub issue, attach a redacted profile:

```bash
python3 skills/isaaclab-installer/scripts/profile_io.py redact -o profile-for-issue.yaml
```

Drag-drop `profile-for-issue.yaml` into the issue. It carries OS, driver, GLIBC, Python, install combo, and the failing step — without your username or home path.
