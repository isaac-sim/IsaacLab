#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Diagnostic only (do not merge). Runs INSIDE the ARM CI image on the DGX Spark runner.
#
# Question: does NCCL communicator creation fail next to an RTX renderer on this GPU and driver
# (NVBug 6890323), and which NCCL build avoids it? Torch stays as built (2.11 + cu130 on PR 8372);
# only the NCCL library changes between arms:
#   A  nvidia-nccl-cu13 2.28.9  (as locked by PR 8372)
#   B  nvidia-nccl-cu13 2.29.7  (develop, and PR 8357's ARM leg)
#   C  nvidia-nccl-cu12 2.29.7  (PR 8280)
# RL-Games creates a communicator even at one rank (rsl_rl skips distributed mode below two), and
# communicator creation is where NCCL sets the kernels' shared-memory attribute. A case that never
# logs an NCCL init is reported as invalid, not passed. ONLY=<regex> limits the cases run.

set -u
out=/reports
mkdir -p "$out"
summary="$out/summary.md"
: > "$summary"
py="${VIRTUAL_ENV}/bin/python"
site=$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cd /workspace/isaaclab || exit 1
rm -f _isaac_sim
if [ -x /isaac-sim/python.sh ]; then ln -s /isaac-sim _isaac_sim; fi

if [ -n "${EXTRA_PIP:-}" ]; then
  # shellcheck disable=SC2086
  uv pip install --python "$py" ${EXTRA_PIP} >"$out/extra-pip.log" 2>&1 || echo "extra pip install failed" | tee -a "$summary"
fi

nccl_version() {
  "$py" -c 'import torch; print(".".join(map(str, torch.cuda.nccl.version())))' 2>/dev/null
}

{
  echo "## Environment"
  echo '```'
  echo "arch: $(uname -m)"
  nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total --format=csv
  "$py" - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} (CUDA {torch.version.cuda})")
print(f"device {p.name} sm_{p.major}{p.minor} smem_per_block_optin={getattr(p, 'shared_memory_per_block_optin', '?')}")
PY
  echo '```'
} 2>&1 | tee -a "$summary"

# ---- Standalone driver probe: CUDA driver API + Vulkan only (no NCCL, RTX or PyTorch) ----
probe() {
  local hdr=/tmp/vulkan-headers cap nvrtc_dir
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
  nvrtc_dir="$site/nvidia/cu13/lib"
  mkdir -p "$hdr"
  curl -fsSL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.3.275.tar.gz \
    | tar xz -C "$hdr" --strip-components=2 Vulkan-Headers-1.3.275/include || return 1
  gcc -O1 -I"$hdr" -I"$site/nvidia/cu13/include" -o /tmp/vk_legal_probe /diag/vk_legal_probe.c \
    -l:libcuda.so.1 -l:libvulkan.so.1 -ldl || return 1
  local exts=VK_KHR_ray_tracing_pipeline,VK_KHR_acceleration_structure,VK_KHR_deferred_host_operations
  exts=$exts,VK_EXT_descriptor_indexing,VK_KHR_buffer_device_address,VK_KHR_spirv_1_4,VK_KHR_shader_float_controls
  local mode
  for mode in none before; do
    LD_LIBRARY_PATH="$nvrtc_dir:${LD_LIBRARY_PATH:-}" DISPLAY= NVRTC_LIB="$nvrtc_dir/libnvrtc.so.13" \
      NVRTC_ARCH="sm_$cap" FEATURES=1 BSEARCH=1 VK_EXTS="$exts" /tmp/vk_legal_probe "$mode" kernel 0 nvrtc
  done
}
echo "## Driver probe (Vulkan ray-tracing device, then cuKernelSetAttribute at the documented maximum)" | tee -a "$summary"
echo '```' >>"$summary"
probe >"$out/probe.log" 2>&1
grep -q "RESULT vk=before" "$out/probe.log" || echo "probe did not complete; see probe.log" >>"$summary"
grep -E "BSEARCH|RESULT|vulkan:|VALIDATION_SUMMARY" "$out/probe.log" | tee -a "$summary"
echo '```' >>"$summary"

# ---- One-rank training: NCCL communicator creation next to each renderer ----
train() {  # <label> <task> <presets>
  local label=$1 task=$2 presets=$3 log rc res nccl err
  if [ -n "${ONLY:-}" ] && ! [[ $label =~ $ONLY ]]; then return; fi
  log="$out/$label.log"
  timeout -k 30 1200 env NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT CUDA_LOG_FILE=stderr \
    uv run --no-sync isaaclab -p scripts/reinforcement_learning/train_multigpu.py \
    --num_gpus 1 --log_all_ranks --rdzv_backend c10d --rdzv_endpoint localhost:0 --rdzv_id "$label" \
    --rl_library rl_games --task "$task" "presets=$presets" \
    --num_envs 32 --max_iterations 3 --seed 7 >"$log" 2>&1
  rc=$?
  pkill -9 -f "isaaclab_rl|train_multigpu|torch.distributed.run" >/dev/null 2>&1 || true
  nccl=$(grep -m1 -oE "NCCL version [0-9.]+\+cuda[0-9.]+" "$log" | sed 's/NCCL version //')
  if [ "$rc" -eq 0 ] && ! grep -q "Traceback (most recent call last):" "$log"; then
    if [ -n "$nccl" ]; then res="✅ pass"; else res="⚠️ invalid: no NCCL init logged"; fi
  else
    res="❌ fail (rc=$rc)"
  fi
  err=$(grep -m1 -E "larger than limit|invalid argument|unhandled cuda error|undefined symbol" "$log" | cut -c1-160 | tr '|' '/')
  echo "| $label | \`$presets\` | ${nccl:-not logged} | $res | ${err:-} |" | tee -a "$summary"
}

camera=Isaac-Cartpole-Camera-Direct
{
  echo
  echo "## One-rank training (Torch $("$py" -c 'import torch; print(torch.__version__)'))"
  echo
  echo "| Case | Presets | NCCL loaded | Result | First error line |"
  echo "|---|---|---|---|---|"
} | tee -a "$summary"

uv pip install --python "$py" --no-deps --reinstall nvidia-nccl-cu13==2.28.9 >"$out/swap-A.log" 2>&1
echo "[diag] arm A: NCCL $(nccl_version)"
train A-kit_rtx "$camera" isaacsim_physx,isaacsim_rtx
train A-ovrtx "$camera" newton_mjwarp,ovrtx

uv pip install --python "$py" --no-deps --reinstall nvidia-nccl-cu13==2.29.7 >"$out/swap-B.log" 2>&1
echo "[diag] arm B: NCCL $(nccl_version)"
train B-no_rtx "$camera" newton_mjwarp,newton_renderer
train B-kit_rtx "$camera" isaacsim_physx,isaacsim_rtx
train B-ovrtx "$camera" newton_mjwarp,ovrtx

uv pip uninstall --python "$py" nvidia-nccl-cu13 >"$out/swap-C.log" 2>&1
uv pip install --python "$py" --no-deps --reinstall nvidia-nccl-cu12==2.29.7 >>"$out/swap-C.log" 2>&1
echo "[diag] arm C: NCCL $(nccl_version)"
train C-kit_rtx "$camera" isaacsim_physx,isaacsim_rtx
train C-ovrtx "$camera" newton_mjwarp,ovrtx

chmod -R a+rwX "$out" 2>/dev/null || true
exit 0
