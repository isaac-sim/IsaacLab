#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Install the two isolated environments from a transferred Isaac Lab checkout.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
model_project="$repo_root/../Isaac-GR00T"
model_path="$repo_root/../embodied-template/models/stack-cube-n1.7-sft/checkpoint"
backbone_path="$repo_root/../embodied-template/models/nvidia/Cosmos-Reason2-2B"
gr00t_revision=d2b7e75b937e3ec9aa5dbc798f08b89692c49734
gpu=${CUDA_VISIBLE_DEVICES:-0}
min_free_gb=100
skip_system_deps=false
verify_only=false
dry_run=false
smoke_test=false

usage() {
    cat <<'EOF'
Usage: bash scripts/reinforcement_learning/gr00t_skrl/deploy.sh [options]

Deploy from this Isaac Lab checkout; no activation or interactive prompts needed.
Requires Ubuntu 22.04/24.04 x86_64, a rendering-capable NVIDIA GPU with >=23,000
MiB VRAM, driver >=580.65.06, >=32 GB RAM and local micro-SFT/backbone assets.
Installs OS packages with root or passwordless sudo unless --skip_system_deps.

Options:
  --model_project PATH     GR00T checkout (default: ../Isaac-GR00T)
  --model_path PATH        Transferred micro-SFT checkpoint
  --backbone_path PATH     Transferred Cosmos backbone; retain nvidia/Cosmos-Reason2
  --gpu INDEX_OR_UUID      One physical NVIDIA GPU (default: CUDA_VISIBLE_DEVICES or 0)
  --min_free_gb N          Minimum free GiB on installation/cache volumes (default: 100)
  --skip_system_deps       Use OS packages already installed by the cloud image
  --verify_only            Verify existing environments without installing anything
  --dry_run                Print paths and installation commands; make no changes
  --smoke_test             Also run the real two-step PPO smoke (writes a ~9.1 GiB checkpoint)
  -h, --help               Show this help

Relative paths are relative to the caller's working directory. Models are never
downloaded or rewritten. Transfer backbone symlink targets as well (e.g. rsync -aL).
Logs: logs/gr00t_skrl/deploy/<timestamp>/. Re-run after an interrupted installation.
EOF
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
report_failure() {
    local status=$1 line=$2
    printf 'Deployment failed at line %s (exit %s). Log: %s\n' "$line" "$status" "${log_dir:-not created}" >&2
    exit "$status"
}
trap 'report_failure "$?" "$LINENO"' ERR

while (($#)); do
    case "$1" in
        --model_project|--model_path|--backbone_path|--gpu|--min_free_gb)
            (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || fail "Missing value for $1"
            case "$1" in
                --model_project) model_project=$2 ;;
                --model_path) model_path=$2 ;;
                --backbone_path) backbone_path=$2 ;;
                --gpu) gpu=$2 ;;
                --min_free_gb) min_free_gb=$2 ;;
            esac
            shift 2 ;;
        --skip_system_deps) skip_system_deps=true; shift ;;
        --verify_only) verify_only=true; shift ;;
        --dry_run) dry_run=true; shift ;;
        --smoke_test) smoke_test=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1 (see --help)" ;;
    esac
done

[[ "$min_free_gb" =~ ^[0-9]+$ && ${#min_free_gb} -le 4 ]] || fail "--min_free_gb must be an integer (0-9999)"
[[ "$gpu" =~ ^[0-9]+$ || "$gpu" =~ ^GPU-[[:xdigit:]-]+$ ]] || fail "Select one GPU index or GPU UUID with --gpu"
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || fail "Only Linux x86_64 is supported by this deployment"
[[ -r /etc/os-release ]] || fail "Missing /etc/os-release"
# shellcheck disable=SC1091
source /etc/os-release
[[ "$ID" == ubuntu && ( "$VERSION_ID" == 22.04 || "$VERSION_ID" == 24.04 ) ]] || fail "Use Ubuntu 22.04 or 24.04"
glibc_version=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
dpkg --compare-versions "$glibc_version" ge 2.35 || fail "Isaac Sim wheels require GLIBC >=2.35"

# Do not resolve model symlinks: upstream chooses the backbone class from the path.
model_project=$(realpath -ms -- "$model_project")
model_path=$(realpath -ms -- "$model_path")
backbone_path=$(realpath -ms -- "$backbone_path")
[[ "$model_project" != "$repo_root" ]] || fail "GR00T must have its own checkout and environment"
[[ "$backbone_path" == *nvidia/Cosmos-Reason2* ]] || fail "--backbone_path must contain nvidia/Cosmos-Reason2"
for filename in config.json processor_config.json statistics.json embodiment_id.json; do
    [[ -s "$model_path/$filename" ]] || fail "Missing $model_path/$filename; transfer the micro-SFT checkpoint first"
done
for filename in config.json tokenizer_config.json preprocessor_config.json; do
    [[ -s "$backbone_path/$filename" ]] || fail "Missing $backbone_path/$filename; transfer the backbone and symlink target first"
done

if [[ -e "$model_project" ]]; then
    [[ -d "$model_project/.git" ]] || fail "Existing --model_project is not a standalone Git checkout"
    existing_revision=$(git -C "$model_project" rev-parse --verify HEAD 2>/dev/null || true)
    if [[ -n "$existing_revision" ]]; then
        [[ "$existing_revision" == "$gr00t_revision" ]] || fail "GR00T revision differs from validated $gr00t_revision; use a separate --model_project"
        git -C "$model_project" diff --quiet HEAD -- || fail "GR00T has tracked modifications; use a clean checkout"
    fi
fi

printf 'Isaac Lab: %s\nGR00T: %s @ %s\nCheckpoint: %s\nBackbone: %s\nGPU: %s\n' \
    "$repo_root" "$model_project" "$gr00t_revision" "$model_path" "$backbone_path" "$gpu"
if "$dry_run"; then
    cat <<'EOF'
Plan (not executed):
  Check NVIDIA driver, VRAM, RAM and free disk space.
  Install OS packages (unless --skip_system_deps or --verify_only).
  Install uv 0.11.26 if uv is absent; obtain Python 3.12 with uv.
  Fetch and detach the fixed GR00T revision if its checkout is absent.
  Isaac Lab: uv sync --frozen --inexact --extra isaacsim
  GR00T (from its directory): uv sync --frozen --inexact --extra dev
  GR00T: uv --no-config pip freeze --exclude-editable -> constraints
  GR00T: uv --no-config pip install --constraint constraints -r requirements-model.txt
  Verify asset shards, both CUDA environments, offline processor and headless PhysX/Kit.
  With --smoke_test: run the two-step native skrl PPO launcher.
  With --verify_only: skip all installation steps.
EOF
    exit 0
fi

command -v nvidia-smi >/dev/null || fail "NVIDIA driver is missing; install the cloud provider's GPU driver first"
gpu_info=$(nvidia-smi -i "$gpu" --query-gpu=driver_version,memory.total,name --format=csv,noheader,nounits)
IFS=, read -r driver_version gpu_memory gpu_name <<< "$gpu_info"
driver_version=${driver_version//[[:space:]]/}
gpu_memory=${gpu_memory//[[:space:]]/}
dpkg --compare-versions "$driver_version" ge 580.65.06 || fail "Driver $driver_version is below CUDA 13.0 minimum 580.65.06"
if [[ ! "$gpu_memory" =~ ^[0-9]+$ ]] || ((gpu_memory < 23000)); then
    fail "This PPO configuration needs a GPU with >=23,000 MiB VRAM"
fi
ram_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
((ram_kib >= 30000000)) || fail "Use a cloud instance with at least 32 GB RAM"
printf 'GPU:%s, %s MiB; driver %s; RAM %s KiB\n' "$gpu_name" "$gpu_memory" "$driver_version" "$ram_kib"

if ! "$verify_only"; then
    for volume_path in "$repo_root" "$model_project" "${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"; do
        while [[ ! -d "$volume_path" ]]; do volume_path=$(dirname -- "$volume_path"); done
        free_kib=$(df -Pk -- "$volume_path" | awk 'END {print $4}')
        ((free_kib >= 10#$min_free_gb * 1024 * 1024)) || fail "Less than $min_free_gb GiB free at $volume_path; choose a larger disk/cache volume"
    done
fi

if "$verify_only"; then
    if ! command -v uv >/dev/null && [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    command -v uv >/dev/null || fail "uv is missing; run deployment without --verify_only"
    [[ -x "$repo_root/.venv/bin/python" && -x "$model_project/.venv/bin/python" ]] || fail "Both .venv environments must exist for --verify_only"
elif ! "$skip_system_deps"; then
    sudo_command=()
    if ((EUID != 0)); then
        if ! command -v sudo >/dev/null || ! sudo -n true; then
            fail "Need root/passwordless sudo for OS packages, or use --skip_system_deps"
        fi
        sudo_command=(sudo -n)
    fi
fi

log_dir="$repo_root/logs/gr00t_skrl/deploy/$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p -- "$log_dir"
exec > >(tee "$log_dir/install.log") 2>&1
printf 'Deployment log: %s/install.log\n' "$log_dir"

# Keep caller activation and uv project overrides out of both environments.
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH UV_PROJECT UV_PROJECT_ENVIRONMENT UV_WORKING_DIR UV_PYTHON UV_CONFIG_FILE UV_NO_CONFIG
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export OMNI_KIT_ACCEPT_EULA=YES NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
export UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5
cd -- "$repo_root"
if ! "$verify_only"; then
    if ! "$skip_system_deps"; then
        glib_package=libglib2.0-0
        if [[ "$VERSION_ID" == 24.04 ]]; then glib_package=libglib2.0-0t64; fi
        "${sudo_command[@]}" apt-get update
        "${sudo_command[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            ca-certificates curl git build-essential cmake pkg-config ffmpeg \
            libgl1 libegl1 "$glib_package" libsm6 libxext6 libxrender1 libxi6 libxrandr2 \
            libxcursor1 libxinerama1 libvulkan1 mesa-vulkan-drivers libgomp1
    fi
    for command_name in curl git ffmpeg; do
        command -v "$command_name" >/dev/null || fail "Missing $command_name; install OS packages first"
    done
    if ! command -v uv >/dev/null && [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    if ! command -v uv >/dev/null; then
        curl -LsSf --retry 3 \
            https://github.com/astral-sh/uv/releases/download/0.11.26/uv-installer.sh -o "$log_dir/install-uv.sh"
        UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh "$log_dir/install-uv.sh"
        export PATH="$HOME/.local/bin:$PATH"
    fi
    uv python install 3.12
    if [[ ! -d "$model_project/.git" ]]; then
        git init "$model_project"
        git -C "$model_project" remote add origin https://github.com/NVIDIA/Isaac-GR00T.git
    fi
    if ! git -C "$model_project" rev-parse --verify HEAD >/dev/null 2>&1; then
        git -C "$model_project" fetch --depth 1 origin "$gr00t_revision"
        git -C "$model_project" checkout --detach "$gr00t_revision"
    fi
    [[ $(git -C "$model_project" rev-parse HEAD) == "$gr00t_revision" ]] || fail "GR00T must be at validated revision $gr00t_revision"
    git -C "$model_project" diff --quiet HEAD -- || fail "GR00T has tracked modifications; use a clean checkout"
fi

uv --version
asset_python=3.12
if "$verify_only"; then asset_python="$repo_root/.venv/bin/python"; fi
uv run --no-project --python "$asset_python" python - "$model_path" "$backbone_path" <<'PY'
import json
import sys
from pathlib import Path

for directory in map(Path, sys.argv[1:]):
    index = directory / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    if not shards:
        raise RuntimeError(f"Empty weight index: {index}")
    for shard in shards:
        file = directory / shard
        if not file.is_file() or file.stat().st_size < 1024:
            raise RuntimeError(f"Missing weights or Git LFS pointer: {file}")
print("Local weight shards present")
PY
if ! "$verify_only"; then
    uv sync --frozen --inexact --extra isaacsim
    (
        cd -- "$model_project"
        uv sync --frozen --inexact --extra dev
    )
    uv --no-config pip freeze --python "$model_project/.venv/bin/python" --exclude-editable > "$log_dir/model-constraints.txt"
    uv --no-config pip install --python "$model_project/.venv/bin/python" \
        --constraint "$log_dir/model-constraints.txt" -r "$script_dir/requirements-model.txt"
fi

export HF_HUB_OFFLINE=1
uv run --project "$model_project" --no-sync python - "$model_path" "$backbone_path" <<'PY'
import sys
from importlib.metadata import version

import gr00t.model
import numpy as np
import torch
from transformers import AutoProcessor

for package, expected in {"torch": "2.9.0+cu128", "numpy": "1.26.4", "transformers": "4.57.3",
                          "skrl": "2.1.0", "tensorboard": "2.20.0"}.items():
    actual = version(package)
    if actual != expected:
        raise RuntimeError(f"Model {package}: expected {expected}, got {actual}")
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError("GR00T requires CUDA and BF16 support")
torch.ones(2, 2, device="cuda", dtype=torch.bfloat16).matmul(torch.ones(2, 2, device="cuda", dtype=torch.bfloat16))
torch.cuda.synchronize()
processor = AutoProcessor.from_pretrained(sys.argv[1], model_name=sys.argv[2], local_files_only=True)
processor.eval()
if processor.model_name != sys.argv[2]:
    raise RuntimeError("Processor did not select the transferred backbone")
print(f"Model environment and offline processor OK: NumPy {np.__version__}, GPU {torch.cuda.get_device_name(0)}")
PY

uv run --no-sync python - <<'PY'
from importlib.metadata import version

import numpy as np
import torch

from isaaclab.app import launch_simulation
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

for package, expected in {"torch": "2.12.0+cu130", "isaacsim": "6.1.0.0", "transformers": "5.10.4"}.items():
    if version(package) != expected:
        raise RuntimeError(f"Simulator {package}: expected {expected}, got {version(package)}")
if np.lib.NumpyVersion(np.__version__) < "2.0.0" or not torch.cuda.is_available():
    raise RuntimeError("Simulator requires NumPy >=2 and CUDA")
cfg = parse_env_cfg("IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor", device="cuda:0", num_envs=1,
                    overrides=["physics=isaacsim_physx"])
with launch_simulation(cfg, {"headless": True, "enable_cameras": True, "visualizer": None,
                             "visualizer_explicit": True}):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext

    simulation = SimulationContext(cfg.sim)
    sphere = RigidObject(RigidObjectCfg(
        prim_path="/World/DeploymentProbe",
        spawn=sim_utils.SphereCfg(radius=0.1, rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                                 mass_props=sim_utils.MassCfg(mass=1.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    ))
    simulation.reset()
    initial_height = sphere.data.root_pos_w.torch[:, 2].clone()
    for _ in range(8):
        simulation.step()
        sphere.update(simulation.get_physics_dt())
    if not torch.all(sphere.data.root_pos_w.torch[:, 2] < initial_height):
        raise RuntimeError("PhysX sphere did not fall under gravity")
    print("Simulator environment, headless Kit and falling PhysX sphere OK", flush=True)
PY

if "$smoke_test"; then
    uv run --no-sync python -m scripts.reinforcement_learning.gr00t_skrl.launch \
        --model_project "$model_project" --model_path "$model_path" --backbone_path "$backbone_path" \
        --run_dir "$log_dir/smoke" --episode_steps 2
fi
printf '\nDeployment verified. Log: %s/install.log\nRun from the Isaac Lab checkout:\n' "$log_dir"
uv_bin_dir=$(dirname -- "$(command -v uv)")
printf "PATH=%q:\"\$PATH\" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=%q uv run --no-sync python -m scripts.reinforcement_learning.gr00t_skrl.launch --model_project %q --model_path %q --backbone_path %q --episode_steps 2\n" \
    "$uv_bin_dir" "$gpu" "$model_project" "$model_path" "$backbone_path"
