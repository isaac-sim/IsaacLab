#!/usr/bin/env bash
# Runs INSIDE the 1-docker container hosting the multi-GPU lane's pytest
# shards. Mounted by both the CI workflow (.github/workflows/test-multi-gpu-
# pytest.yaml) and the local probe (data/mgpu-1docker-probe/run.sh) so the
# logic lives in exactly one place and changes are version-controlled +
# shellcheck-able.
#
# Container expectations (set by the host launcher):
#   --gpus all
#   --user $host_uid:$host_gid
#   -v <workspace>:/workspace/isaaclab:rw
#   -v <queue_root>:/mgpu:rw           (subdirs queue/, inflight/, done/)
#   -v <logs_dir>:/shard-logs:rw       (per-shard tee target)
#   -e HOME=/tmp/mgpu-base-home        (writable tmpfs for pip --user)
#   -e PYTHONUSERBASE=/tmp/mgpu-pyuserbase   (shared site-packages across shards)
#   -e ISAACLAB_TEST_QUEUE=/mgpu       (conftest queue root)
#   -e TEST_INCLUDE_FILES="<comma-sep basenames>"
#   -e ISAACLAB_PIN_KIT_GPU=1          (Kit overrides for the multi-GPU race)
#   -e CUDA_VISIBLE_DEVICES="<MIG-UUID,...>"  (MIG hosts only; discrete = --gpus all)
#
# Behavior:
#   1. Materializes HOME + PYTHONUSERBASE dirs (tmpfs, world-writable)
#   2. Installs pytest deps (junitparser et al.) into the shared PYTHONUSERBASE
#   3. Derives shard count from nvidia-smi -L (authoritative; torch.cuda.device_count
#      under-counts MIG-on-same-parent unless CUDA_VISIBLE_DEVICES enumerates each)
#   4. Cross-checks torch against the nvidia-smi count and caps shards to what torch
#      can address (guards against CUDA_VISIBLE_DEVICES misconfig on a MIG host)
#   5. Fans out 1 pytest subshell per non-default cuda:N with per-shard HOME +
#      ISAACLAB_SIM_DEVICE + ISAACLAB_TEST_DEVICES; each shard tees its stdout to
#      /shard-logs/cuda-N.log for the host's grouped re-print after the run
#   6. Waits on every shard before aggregating exit codes — a fast failure doesn't
#      tear down still-running siblings

set +e
cd /workspace/isaaclab
unset HUB__ARGS__DETECT_ONLY DISPLAY

# Container-level HOME + PYTHONUSERBASE for pip --user installs. The image
# runs as --user $host_uid:$host_gid with no matching /etc/passwd entry, so
# HOME defaults to /root which the user cannot write. /tmp/* is on tmpfs
# (1777, world-writable).
#
# PYTHONUSERBASE is the key for the 1-docker shape: pip --user writes to
# ${PYTHONUSERBASE}/lib/python3.12/site-packages, and every Python invocation
# that sees the same env var imports from there. Per-shard subshells below
# override HOME (so .cache / .nvidia-omniverse are isolated) but inherit
# PYTHONUSERBASE so junitparser et al. resolve everywhere.
mkdir -p /tmp/mgpu-base-home /tmp/mgpu-pyuserbase

# Pytest deps (same as run-tests action). junitparser is imported at
# tools/conftest.py load time, so it must be present first.
./isaaclab.sh -p -m pip install pytest pytest-mock junitparser flatdict flaky py-spy "coverage>=7.6.1"

# Shard count from nvidia-smi -L (truth; torch under-counts MIG).
MIG_COUNT=$(nvidia-smi -L | grep -c "^  MIG ")
GPU_COUNT=$(nvidia-smi -L | grep -c "^GPU ")
if [ "$MIG_COUNT" -gt 0 ]; then
  DEV_COUNT=$MIG_COUNT
  echo "::notice::container: MIG mode, $MIG_COUNT slices"
else
  DEV_COUNT=$GPU_COUNT
  echo "::notice::container: discrete mode, $GPU_COUNT GPUs"
fi
if [ "$DEV_COUNT" -lt 2 ]; then
  echo "::error::Need at least 2 visible devices; found $DEV_COUNT"
  exit 1
fi

# Cross-check with torch and cap shard count to what torch can actually
# address. Guards against a CUDA_VISIBLE_DEVICES misconfig silently fanning
# out shards that crash on device access.
TORCH_COUNT=$(/isaac-sim/python.sh -c "import torch; print(torch.cuda.device_count())")
echo "container: torch sees $TORCH_COUNT cuda devices (cross-check vs $DEV_COUNT)"
if [ "$TORCH_COUNT" -lt "$DEV_COUNT" ]; then
  echo "::warning::torch sees fewer devices than nvidia-smi — capping shards to $TORCH_COUNT"
  DEV_COUNT=$TORCH_COUNT
fi

# Fan out 1 pytest subshell per non-default cuda:N. Each gets its own HOME
# (per-shard isolation for .cache, .local/share, etc.) and per-shard
# ISAACLAB_SIM_DEVICE / ISAACLAB_TEST_DEVICES.
declare -A pids
for ((cuda = 1; cuda < DEV_COUNT; cuda++)); do
  zeros=""
  for ((i = 0; i <= cuda; i++)); do zeros+="0"; done
  runtime_devices="${zeros}1"

  shard_home="/tmp/isaaclab-ci-home-${cuda}"
  mkdir -p "${shard_home}/.cache" "${shard_home}/.local/share" \
           "${shard_home}/.nvidia-omniverse/config" \
           "${shard_home}/.nvidia-omniverse/logs"

  shard_log="/shard-logs/cuda-${cuda}.log"

  (
    export HOME="$shard_home"
    export XDG_CACHE_HOME="${HOME}/.cache"
    export XDG_DATA_HOME="${HOME}/.local/share"
    export ISAACLAB_TEST_DEVICES="$runtime_devices"
    export ISAACLAB_SIM_DEVICE="cuda:${cuda}"

    ./isaaclab.sh -p -m pytest \
      --ignore=tools/conftest.py \
      --ignore=source/isaaclab/test/install_ci \
      tools -v 2>&1 \
      | tee "$shard_log" \
      | stdbuf -oL sed "s/^/[cuda:${cuda}] /"

    exit "${PIPESTATUS[0]}"
  ) &
  pids[$cuda]=$!
  echo "::notice::launched shard cuda:${cuda} (pid ${pids[$cuda]}, runtime_devices=$runtime_devices)"
done

# Wait for every shard before aggregating exits — a fast failure must not
# tear down still-running siblings.
declare -A results
for cuda in "${!pids[@]}"; do
  wait "${pids[$cuda]}"
  results[$cuda]=$?
done

fail=0
for cuda in "${!results[@]}"; do
  rc="${results[$cuda]}"
  echo "shard cuda:${cuda} exited $rc"
  [ "$rc" -eq 0 ] || fail=1
done
exit $fail
