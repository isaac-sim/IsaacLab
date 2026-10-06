#!/usr/bin/env bash
# Runs INSIDE the 1-docker container hosting the multi-GPU lane's pytest
# shards. Mounted by the CI workflow (.github/workflows/test-multi-gpu-
# pytest.yaml) so the logic lives in exactly one place and changes are
# version-controlled + shellcheck-able. Lives under ``.github/actions/multi-gpu/``
# so it sits next to the workflow that consumes it rather than alongside ``tools/``.
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
#   -e ISAACLAB_FABRIC_USE_GPU_INTEROP=0  (temporary Kit/PhysX CI workaround)
#   -e CUDA_VISIBLE_DEVICES="<MIG-UUID,...>"  (MIG hosts only; discrete = --gpus all)
#
# Behavior:
#   1. Materializes HOME + PYTHONUSERBASE dirs (tmpfs, world-writable)
#   2. Derives shard count from nvidia-smi -L (authoritative; torch.cuda.device_count
#      under-counts MIG-on-same-parent unless CUDA_VISIBLE_DEVICES enumerates each)
#   3. Cross-checks torch against the nvidia-smi count and caps shards to what torch
#      can address (guards against CUDA_VISIBLE_DEVICES misconfig on a MIG host)
#   4. Fans out 1 pytest subshell per non-default cuda:N with per-shard HOME +
#      ISAACLAB_TEST_DEVICES; each shard tees its stdout to
#      /shard-logs/cuda-N.log for the host's grouped re-print after the run
#   5. Waits on every shard before aggregating exit codes — a fast failure doesn't
#      tear down still-running siblings
#
# The pytest deps (pytest, junitparser et al.) are baked into the image by
# .github/actions/docker-build, so this script no longer installs them.

set +e  # keep going on errors; per-shard exit codes are aggregated at the end
cd /workspace/isaaclab
unset DISPLAY  # clear the var that would force Kit into headed (X11) mode

# Container-level HOME + PYTHONUSERBASE. The image runs as
# --user $host_uid:$host_gid with no matching /etc/passwd entry, so HOME
# defaults to /root which the user cannot write. /tmp/* is on tmpfs
# (1777, world-writable).
#
# Both must exist before any shard starts: per-shard subshells below override
# HOME (so .cache / .nvidia-omniverse are isolated) but inherit
# PYTHONUSERBASE, so anything writing to the user site shares one directory.
mkdir -p /tmp/mgpu-base-home /tmp/mgpu-pyuserbase

# Shard count from nvidia-smi -L (truth; torch under-counts MIG).
MIG_COUNT=$(nvidia-smi -L | grep -c "^  MIG ")  # grep -c = count of matching lines (MIG slices)
GPU_COUNT=$(nvidia-smi -L | grep -c "^GPU ")    # count of whole GPUs
if [ "$MIG_COUNT" -gt 0 ]; then
  DEV_COUNT=$MIG_COUNT
  echo "::notice::container: MIG mode, $MIG_COUNT slices"
else
  DEV_COUNT=$GPU_COUNT
  echo "::notice::container: discrete mode, $GPU_COUNT GPUs"
fi

# Cross-check with torch and cap shard count to what torch can actually
# address. Guards against a CUDA_VISIBLE_DEVICES misconfig silently fanning
# out shards that crash on device access.
# torch lives in the image's venv, not Kit's bundled Python, so resolve it the way the shards do.
TORCH_COUNT=$(uv run --no-sync isaaclab -p -c "import torch; print(torch.cuda.device_count())" | tail -n1)
if ! [[ "$TORCH_COUNT" =~ ^[0-9]+$ ]]; then
  echo "::error::Could not read torch.cuda.device_count(); got '$TORCH_COUNT'"
  exit 1
fi
echo "container: torch sees $TORCH_COUNT cuda devices (cross-check vs $DEV_COUNT)"
if [ "$TORCH_COUNT" -lt "$DEV_COUNT" ]; then
  echo "::warning::torch sees fewer devices than nvidia-smi — capping shards to $TORCH_COUNT"
  DEV_COUNT=$TORCH_COUNT
fi
if [ "$DEV_COUNT" -lt 2 ]; then
  echo "::error::Need at least 2 visible devices; found $DEV_COUNT"
  exit 1
fi

# Fan out 1 pytest subshell per shard. Each gets its own HOME (per-shard
# isolation for .cache, .local/share, etc.) and per-shard ISAACLAB_TEST_DEVICES.
#   launch_shard <name> <cuda> [<CUDA_VISIBLE_DEVICES>]
declare -A pids  # associative array: shard name -> background PID
launch_shard() {
  local name="$1" cuda="$2" cvd="${3:-}"
  local zeros=""
  for ((i = 0; i <= cuda; i++)); do zeros+="0"; done  # build the leading zeros of the device mask
  local runtime_devices="${zeros}1"  # e.g. cuda:2 -> "0001": only this GPU active in the ISAACLAB_TEST_DEVICES mask

  local shard_home="/tmp/isaaclab-ci-home-${name}"
  mkdir -p "${shard_home}/.cache" "${shard_home}/.local/share" \
           "${shard_home}/.nvidia-omniverse/config" \
           "${shard_home}/.nvidia-omniverse/logs"

  local shard_log="/shard-logs/cuda-${name}.log"

  # each shard runs in its own subshell, backgrounded with '&' below, so all shards run in parallel
  (
    export HOME="$shard_home"
    export XDG_CACHE_HOME="${HOME}/.cache"
    export XDG_DATA_HOME="${HOME}/.local/share"
    export ISAACLAB_TEST_DEVICES="$runtime_devices"
    export ISAACLAB_TEST_QUEUE_SHARD="cuda-${name}"  # done/<shard>/ in the queue, so the summary splits shards
    [ -z "$cvd" ] || export CUDA_VISIBLE_DEVICES="$cvd"

    # Full pytest output captures to $shard_log; live stdout is filtered
    # down to high-signal lines (test boundaries, failures, summary stats,
    # tracebacks). Kit init chatter, plugin registration, omni.usd Transfer
    # logs, etc. only end up in the shard logfile, NOT on the live workflow
    # log. The workflow's grouped re-print at end of run still shows the
    # full $shard_log under a collapsible ``::group::shard cuda:N log``.
    # (tee = full output to the log file; stdbuf -oL = flush per line so the
    # filtered grep/sed stream appears live, not in delayed chunks.)
    ./isaaclab.sh -p -m pytest \
      --ignore=tools/conftest.py \
      --ignore=source/isaaclab/test/install_ci \
      tools -v 2>&1 \
      | tee "$shard_log" \
      | stdbuf -oL grep -aE \
          '🚀|^source/.*::.* (PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)|^(Total|Passing|Failing|Crashed|Startup Hang|Timeout|Total Wall Time|Total Test Time|Passing Percentage):|^~~~~|^=+ |^E +|^ +File |Traceback|^FAILED|^ERROR ' \
      | stdbuf -oL sed "s/^/[cuda:${name}] /"

    exit "${PIPESTATUS[0]}"  # exit with pytest's code, not tee/grep/sed's (PIPESTATUS[0] = first pipe stage)
  ) &
  pids[$name]=$!  # $! = PID of the subshell just backgrounded
  echo "::notice::launched shard cuda:${name} (pid ${pids[$name]}, runtime_devices=$runtime_devices${cvd:+, CUDA_VISIBLE_DEVICES=$cvd})"
}

for ((cuda = 1; cuda < DEV_COUNT; cuda++)); do  # start at 1: a shard tests a non-default device index
  launch_shard "$cuda" "$cuda"
done

# Physical GPU 0 would idle, since single-GPU CI covers cuda:0. Swapping the first two visible
# devices makes it this shard's cuda:1, so it still tests a non-default index.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra visible <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t visible < <(seq 0 $((DEV_COUNT - 1)))
fi
swapped=("${visible[1]}" "${visible[0]}" "${visible[@]:2}")
launch_shard "1-gpu0" 1 "$(IFS=,; echo "${swapped[*]}")"

# Wait for every shard before aggregating exits — a fast failure must not
# tear down still-running siblings.
declare -A results  # associative array: shard name -> exit code
for cuda in "${!pids[@]}"; do  # "${!pids[@]}" = the array's keys (the shard indices)
  wait "${pids[$cuda]}"  # block until that shard's PID exits
  results[$cuda]=$?      # $? = the waited shard's exit code
done

fail=0
for cuda in "${!results[@]}"; do
  rc="${results[$cuda]}"
  echo "shard cuda:${cuda} exited $rc"
  [ "$rc" -eq 0 ] || fail=1  # any non-zero shard makes the whole run fail
done
exit $fail
