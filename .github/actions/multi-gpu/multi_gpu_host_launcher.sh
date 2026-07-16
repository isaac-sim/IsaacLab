#!/usr/bin/env bash
# Runs ON THE HOST runner — the counterpart to multi_gpu_shard_runner.sh, which
# runs INSIDE the container. Launches ONE container that hosts all N pytest
# shards as parallel subshells, then reconciles the shared work queue.
#
# Invoked by .github/workflows/test-multi-gpu-pytest.yaml's "Run shards in
# parallel" step. Reads from the environment (set by that step):
#   IMAGE_TAG      — per-commit CI image to ``docker run``
#   INCLUDE_FILES  — comma-sep test basenames (conftest include filter)
#   PATHS          — comma-sep relative test paths (seed the work queue)
# plus the runner built-ins GITHUB_RUN_ID / GITHUB_RUN_ATTEMPT (container name)
# and GITHUB_ENV (exports MGPU_RUNTIME_DIR for the downstream summary step).
#
# 1-docker N-shard rationale: ONE container hosts all shards (each pinned to its
# own non-default cuda:N via ISAACLAB_TEST_DEVICES, pulling from the shared
# queue). Collapsing the previous N-container layout into one
# removes cross-container races on the workspace mount (``_isaac_sim`` symlink,
# ``/mgpu`` queue, ``/dev/dri`` cap-add) and ~30s of docker init per shard, at
# the cost of sharing /isaac-sim writable subtrees; per-shard HOME (under /tmp,
# set in the shard runner) keeps user-level caches isolated.
#
# Scaling: shard count is derived inside the container from ``nvidia-smi -L``
# (truth — torch.cuda.device_count() under-counts MIG slices on the same parent
# GPU), so this works unchanged on any GPU count/topology. CUDA_VISIBLE_DEVICES
# is set to the comma-separated MIG-UUID list only on MIG-mode hosts; discrete
# runners rely on ``--gpus all``.

set +e  # keep going on errors so the reconciler below always runs and reports
host_uid="$(id -u)"
host_gid="$(id -g)"
host_user="$(id -un)"
# ``/isaac-sim`` is the Kit install root inside the container; pytest
# invocations under ``./isaaclab.sh -p`` resolve the bundled Python as
# ``./_isaac_sim/python.sh``. Mounting the workspace at ``/workspace/isaaclab``
# brings the host's CWD into the container, so we plant a host-side symlink
# whose target resolves only inside the container. Single creator (this step)
# — no concurrent creators.
rm -f _isaac_sim && ln -s /isaac-sim _isaac_sim

# unique per-run temp dir (RUNNER_TEMP if set, else /tmp; XXXXXX -> random suffix)
runtime_dir="$(mktemp -d "${RUNNER_TEMP:-/tmp}/mgpu-runtime.XXXXXX")"
mkdir -p "$runtime_dir"

# Directory-based work queue (atomic os.rename pattern; conftest claims entries
# via os.rename(queue/X, inflight/<shard>/X) — POSIX rename is atomic on the
# same filesystem, so two shards racing on the same entry are kernel-serialized:
# one wins, the other gets ENOENT and moves on). Each pending test path is
# materialized as an empty file in queue/ with '/' slugged to '__' so it's a
# flat filename.
#
# Subdirs:
#   queue/<slug>             — pending claims
#   inflight/<shard>/<slug>  — claimed by this shard, test in flight
#   done/<shard>/<slug>      — test exited cleanly
#
# At job-end the reconciler below asserts queue/ + inflight/ are empty — if
# either is non-empty the job FAILS (silent-drop signal: never claimed OR
# claimed but crashed mid-test).
queue_root="$runtime_dir/queue"
mkdir -p "$queue_root/queue" "$queue_root/inflight" "$queue_root/done"
IFS=',' read -ra _paths <<< "$PATHS"  # split the comma-separated PATHS into the _paths array
for path in "${_paths[@]}"; do
  [ -n "$path" ] || continue          # skip empty entries (PATHS has a trailing comma)
  slug="${path//\//__}"               # flatten to a filename: replace every '/' with '__'
  : > "$queue_root/queue/$slug"       # create an empty marker file (':' = no-op, '>' creates it)
done
echo "::notice::seeded work queue with $(ls -1 "$queue_root/queue" | wc -l) entries"

# Per-shard log dir mounted from the container so we can grouped-print the
# per-shard stream after the run completes.
logs_dir="$runtime_dir/logs"
mkdir -p "$logs_dir"

# MIG detection: only override CUDA_VISIBLE_DEVICES on MIG hosts. nvidia-smi -L
# is authoritative — torch.cuda.device_count() under-counts MIG slices on the
# same parent GPU.
cvd_args=()
if nvidia-smi -L | grep -q "^  MIG "; then
  # read each MIG slice's UUID into the MIGS array (one element per line)
  mapfile -t MIGS < <(nvidia-smi -L | awk -F'UUID: ' '/^  MIG /{print $2}' | tr -d ')')
  MIG_LIST=$(IFS=,; echo "${MIGS[*]}")               # join the array into a comma-separated string
  cvd_args=(-e "CUDA_VISIBLE_DEVICES=${MIG_LIST}")    # stays empty on discrete hosts -> no flag added
  echo "::notice::MIG mode: ${#MIGS[@]} slices — overriding CUDA_VISIBLE_DEVICES"
else
  echo "::notice::discrete GPU mode — relying on --gpus all"
fi

# "${cvd_args[@]}" expands to the CUDA_VISIBLE_DEVICES flag on MIG hosts, nothing otherwise.
docker run --rm --gpus all --network=host \
  --entrypoint bash \
  --user "${host_uid}:${host_gid}" \
  --name "isaac-lab-mgpu-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" \
  -v "$PWD:/workspace/isaaclab:rw" \
  -v "$queue_root:/mgpu:rw" \
  -v "$logs_dir:/shard-logs:rw" \
  -e USER="${host_user}" \
  -e LOGNAME="${host_user}" \
  -e OMNI_KIT_ACCEPT_EULA=yes \
  -e ACCEPT_EULA=Y \
  -e OMNI_KIT_DISABLE_CUP=1 \
  -e ISAAC_SIM_HEADLESS=1 \
  -e ISAAC_SIM_LOW_MEMORY=1 \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONIOENCODING=utf-8 \
  -e ISAACLAB_TEST_QUEUE=/mgpu \
  -e TEST_INCLUDE_FILES="$INCLUDE_FILES" \
  -e ISAACLAB_FABRIC_USE_GPU_INTEROP=0 \
  -e HOME=/tmp/mgpu-base-home \
  -e PYTHONUSERBASE=/tmp/mgpu-pyuserbase \
  "${cvd_args[@]}" \
  -v "$PWD/.github/actions/multi-gpu/multi_gpu_shard_runner.sh:/multi_gpu_shard_runner.sh:ro" \
  "$IMAGE_TAG" \
  /multi_gpu_shard_runner.sh
docker_rc=$?  # $? = exit code of the docker run above

# Grouped re-print of per-shard logs (preserved across the host).
for log in "$logs_dir"/cuda-*.log; do
  cuda=$(basename "$log" .log | sed s/cuda-//)  # extract N from "cuda-N.log"
  echo "::group::shard cuda:${cuda} log"
  cat "$log"
  echo "::endgroup::"
done

# Reconciler — silent-drop guard. Anything left in queue/ was never claimed by
# any shard (work-queue starvation). Anything still in inflight/<shard>/ was
# claimed but never moved to done/ (shard crashed mid-test). Either is a
# silent-drop signal that the docker exit alone can't surface. Fail the job and
# name the orphans.
echo "::group::work-queue reconciler"
unclaimed=$(ls -1 "$queue_root/queue" 2>/dev/null | wc -l)  # files still pending (never claimed)
if [ "$unclaimed" -gt 0 ]; then
  echo "::error::${unclaimed} test(s) never claimed by any shard:"
  ls -1 "$queue_root/queue" | sed 's|__|/|g; s/^/  /'  # decode '__' back to '/', indent for the log
fi
orphans=0
for shard_dir in "$queue_root/inflight"/*/; do  # each inflight/<shard>/ dir (the /*/ matches dirs only)
  [ -d "$shard_dir" ] || continue               # skip if the glob matched nothing
  count=$(ls -1 "$shard_dir" 2>/dev/null | wc -l)
  if [ "$count" -gt 0 ]; then
    echo "::error::$(basename "$shard_dir") claimed but never finished ${count} test(s):"
    ls -1 "$shard_dir" | sed 's|__|/|g; s/^/  /'  # decode '__' back to '/', indent
    orphans=$((orphans + count))                  # $(( )) = integer arithmetic
  fi
done
done_total=$(find "$queue_root/done" -type f 2>/dev/null | wc -l)
echo "::notice::reconciler: done=${done_total} unclaimed=${unclaimed} orphans=${orphans}"
echo "::endgroup::"

# Treat reconciler signals as fatal even if all shards exited 0.
if [ "$unclaimed" -gt 0 ] || [ "$orphans" -gt 0 ]; then
  echo "::error::silent-drop detected — see reconciler group above"
  [ "$docker_rc" -eq 0 ] && docker_rc=2  # clean docker exit but drops found -> force a non-zero code
fi

# Export the runtime directory so the downstream "Aggregated test summary" step
# can locate the done/<shard>/<slug> work distribution and the per-shard logs.
echo "MGPU_RUNTIME_DIR=$runtime_dir" >> "$GITHUB_ENV"  # export to later workflow steps via GitHub's env file

exit "$docker_rc"
