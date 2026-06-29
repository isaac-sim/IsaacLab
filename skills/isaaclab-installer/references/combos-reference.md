# Combos Reference

Verbose, command-by-command expansion of every install combination. The canonical machine-readable form lives in `resources/combos.py`. This file mirrors that data for humans and is the most useful single document for understanding what the skill will actually run.

> Notation: `{ISAACLAB_DIR}` is the chosen install location (default `$HOME/IsaacLab`). `{ENV_NAME}` is the chosen environment name (default `env_isaaclab`).

---

## 1. `pip-uv-source`  — Isaac Sim (pip) + Isaac Lab (source) + uv  *(recommended)*

Best for: RL research, manipulation, sim2real, contributing to Isaac Lab, exploring.

Prerequisites: GLIBC ≥ 2.35, driver ≥ 580.95.05, sudo for apt.

```bash
# 1. System build deps (sudo)
sudo apt-get update && sudo apt-get install -y cmake build-essential
# On aarch64 also: python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev

# 2. Clone Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git {ISAACLAB_DIR}

# 3. Create uv venv (Python 3.12)
cd {ISAACLAB_DIR}
uv venv --python 3.12 --seed {ENV_NAME}

# 4. Upgrade pip inside the env
{ISAACLAB_DIR}/{ENV_NAME}/bin/python -m pip install --upgrade pip

# 5. Install Isaac Sim from pypi.nvidia.com
uv pip install "isaacsim[all,extscache]==6.0.1.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match --prerelease=allow

# 6. Install CUDA-enabled PyTorch (x86_64 uses cu128; aarch64 uses cu130)
uv pip install -U torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 7. Install Isaac Lab packages (all submodules + RL frameworks)
./isaaclab.sh -i all

# 8. Verify (headless)
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

---

## 2. `pip-conda-source`  — Isaac Sim (pip) + Isaac Lab (source) + conda

Same as #1 but conda for env management.

Step 3 changes to:

```bash
conda create -y -n {ENV_NAME} python=3.12
conda activate {ENV_NAME}
```

Step 4 uses `pip` (since `uv pip` is not present by default in a conda env). The `{PIP}` placeholder in the combos is `{ENV_PYTHON} -m pip` for non-uv envs.

---

## 3. `pip-venv-source`  — Isaac Sim (pip) + Isaac Lab (source) + venv

Same as #1 but using stdlib venv. Requires Python 3.12 on the system:

```bash
python3.12 -m venv {ISAACLAB_DIR}/{ENV_NAME}
source {ISAACLAB_DIR}/{ENV_NAME}/bin/activate
```

---

## 4. `binary-uv-source`  — Isaac Sim (binary zip) + Isaac Lab (source) + uv

Use when GLIBC < 2.35 or you prefer a self-contained Isaac Sim install.

```bash
# Manual: download Isaac Sim binary zip from
# https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html
# Extract to e.g. $HOME/isaacsim
# Skill will pause until you confirm completion.

sudo apt-get install -y cmake build-essential
git clone https://github.com/isaac-sim/IsaacLab.git {ISAACLAB_DIR}
cd {ISAACLAB_DIR}
ln -sf $HOME/isaacsim _isaac_sim

uv venv --python 3.12 --seed {ENV_NAME}
{ISAACLAB_DIR}/{ENV_NAME}/bin/python -m pip install --upgrade pip
uv pip install -U torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
./isaaclab.sh -i all
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

---

## 5. `binary-conda-source`  — same as #4 but conda

Substitute the env-creation step with `conda create -y -n {ENV_NAME} python=3.12`.

---

## 6. `binary-venv-source`  — same as #4 but stdlib venv

Substitute the env-creation step with `python3.12 -m venv {ISAACLAB_DIR}/{ENV_NAME}`.

---

## 7. `source-uv-source`  — Build Isaac Sim from source + uv

Only for Isaac Sim contributors. Builds take 30-60 minutes.

```bash
sudo apt-get install -y cmake build-essential
git clone https://github.com/isaac-sim/IsaacSim.git
cd IsaacSim
./build.sh

# Then proceed like combo #4, with:
#   _isaac_sim -> $HOME/IsaacSim/_build/linux-x86_64/release
```

Requires Ubuntu 22.04 LTS or newer.

---

## 8. `source-conda-source`  — same as #7 but conda

---

## 9. `pip-only-uv`  — Isaac Lab + Isaac Sim as pip packages, uv env

For external extensions only. No training scripts ship with this layout.

```bash
uv venv --python 3.12 --seed {ENV_NAME}
{HOME}/{ENV_NAME}/bin/python -m pip install --upgrade pip
uv pip install "isaaclab[isaacsim,all]" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match --prerelease=allow
uv pip install -U torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Optional: rl_games is NOT bundled in the `[all]` extra (pypi disallows git
# links). Install it separately if your workflow needs it:
#   pip install "rl-games @ git+https://github.com/isaac-sim/rl_games.git@python3.11" gym standard-distutils

# Verify
{HOME}/{ENV_NAME}/bin/python -c "import isaaclab, isaacsim; print('OK', isaaclab.__version__)"
```

---

## 10. `pip-only-conda`  — same as #9 but conda

---

## 11. `pip-only-venv`  — same as #9 but stdlib venv

---

## 12. `kitless-uv`  — Isaac Lab + Newton physics only, uv env (no Isaac Sim)

Fastest possible path. PhysX / RTX rendering / ROS / URDF importer are unavailable.

```bash
sudo apt-get install -y cmake build-essential
git clone https://github.com/isaac-sim/IsaacLab.git {ISAACLAB_DIR}
cd {ISAACLAB_DIR}
uv venv --python 3.12 --seed {ENV_NAME}
{ISAACLAB_DIR}/{ENV_NAME}/bin/python -m pip install --upgrade pip
uv pip install -U torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
./isaaclab.sh -i 'newton,rl[rsl-rl]'

# Verify (trains a cartpole for 2 iterations)
./isaaclab.sh train --rl_library rsl_rl \
    --task=Isaac-Cartpole-Direct-v0 --num_envs=16 --max_iterations=2 \
    --headless physics=newton_mjwarp --visualizer newton
```

---

## 13. `kitless-conda`  — same as #12 but conda

---

## Selective install tokens (passed to `./isaaclab.sh -i ...`)

The `-i` argument accepts a comma-separated list of tokens. Quote on bash to protect brackets:

| Token             | Installs                                                          |
| ----------------- | ----------------------------------------------------------------- |
| `all`             | core + optional submodules + auto extras (default)                |
| `none`            | core packages only                                                |
| `mimic`           | `isaaclab_mimic` (imitation learning tools)                       |
| `teleop`          | `isaaclab_teleop` (Linux x86 only)                                |
| `newton`          | Newton physics library                                            |
| `rl[<framework>]` | RL framework — selectors: `rsl-rl`, `skrl`, `sb3`, `rl-games`     |
| `visualizer[<b>]` | Visualizer backend — selectors: `rerun`, `viser`, `newton`, `kit` |
| `ov[ovrtx]`       | OVRTX runtime (vision-only rendering, no Kit)                     |
| `ov[ovphysx]`     | OVPhysX runtime                                                   |
| `contrib[rlinf]`  | rlinf extras                                                      |

Example: `./isaaclab.sh -i 'newton,rl[rsl-rl],visualizer[newton]'`.
