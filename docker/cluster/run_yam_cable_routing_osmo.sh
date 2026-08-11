#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Run checksum-gated YAM cable-routing preflight and training in an official Isaac Lab image.

set -euo pipefail

image_root=/workspace/isaaclab
image_python="${OSMO_IMAGE_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
source_sync="${OSMO_SOURCE_SYNC:-/osmo/run/workspace/source-sync}"
source_acceptance=/osmo/run/workspace/yam-source-accepted.sha256
source_root=/osmo/run/workspace/IsaacLab
live_results=/osmo/run/workspace/results
final_results="${OSMO_OUTPUT_DIR}"
training_artifacts="${live_results}/training-artifacts"
training_runs="${training_artifacts}/runs"
console_logs="${training_artifacts}/console"
preflight_results="${training_artifacts}/preflight"
provenance="${training_artifacts}/run-info"
cache_seed=/osmo/run/workspace/cache-seed
runtime_deps=/osmo/run/workspace/runtime-deps
gpu_monitor_pid=""

if [[ ! -x "${image_python}" ]]; then
    echo "Isaac Sim image Python is unavailable: ${image_python}" >&2
    exit 2
fi

# OSMO reserves some conventional top-level output names. Keep durable
# checkpoints, console streams, preflight data, and provenance below the
# confirmed-retained training-artifacts tree.
mkdir -p \
    "${training_runs}" \
    "${console_logs}" \
    "${preflight_results}" \
    "${provenance}" \
    "${live_results}/wandb" \
    "${cache_seed}/warp" \
    "${cache_seed}/omniclient" \
    "${runtime_deps}" \
    "${final_results}"
# Legacy workflows consume a prepared input or receive source through their
# authenticated rsync tunnel. Multi-node workers fetch from the source task in
# their own synchronized group, avoiding the external output store at startup.
if [[ -z "${OSMO_SOURCE_SYNC:-}" ]]; then
    install -d -m 0777 "${source_sync}"
elif [[ ! -d "${source_sync}" ]]; then
    echo "Prepared source input is unavailable: ${source_sync}" >&2
    exit 3
fi

publish_results() {
    local run_status="$?"
    trap - EXIT
    if [[ -n "${gpu_monitor_pid}" ]]; then
        kill "${gpu_monitor_pid}" 2>/dev/null || true
        wait "${gpu_monitor_pid}" 2>/dev/null || true
    fi
    # OSMO output mounts accept file data but may reject timestamp/ownership
    # preservation on the mount root. Copy recursively without metadata.
    if ! cp -R --no-preserve=all "${live_results}/." "${final_results}/"; then
        echo "Failed to publish training artifacts to ${final_results}." >&2
        if (( run_status == 0 )); then
            run_status=12
        fi
    fi
    manifest_required=1
    if [[ "${OSMO_RUN_MODE}" == multinode && "${OSMO_NODE_RANK:-0}" != 0 ]]; then
        manifest_required=0
    fi
    if (( run_status == 0 && manifest_required != 0 )) \
        && ! (
            cd "${final_results}"
            sha256sum --check "training-artifacts/run-info/checkpoint-manifest.sha256"
        ); then
        echo "Published training checkpoints failed checksum verification in ${final_results}." >&2
        run_status=13
    fi
    exit "${run_status}"
}
trap publish_results EXIT
exec > >(tee -a "${console_logs}/osmo-run.log") 2>&1

if [[ -e /usr/share/vulkan && -e /etc/vulkan && -w /usr/share ]]; then
    mv /usr/share/vulkan /usr/share/vulkan_hidden
fi

if [[ ! "${OSMO_SOURCE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "OSMO_SOURCE_SHA256 is not a lowercase SHA-256 digest." >&2
    exit 2
fi

if [[ -n "${OSMO_SOURCE_URL:-}" ]]; then
    if [[ -n "${OSMO_SOURCE_SYNC:-}" ]]; then
        echo "OSMO_SOURCE_URL and OSMO_SOURCE_SYNC are mutually exclusive." >&2
        exit 2
    fi
    source_fetcher=/tmp/yam-cable-routing-source-fetch.py
    if [[ ! -f "${source_fetcher}" ]]; then
        echo "Injected source fetcher is unavailable: ${source_fetcher}" >&2
        exit 3
    fi
    "${image_python}" "${source_fetcher}" \
        --source-url "${OSMO_SOURCE_URL}" \
        --destination "${source_sync}" \
        --expected-sha256 "${OSMO_SOURCE_SHA256}" \
        --wait-seconds "${OSMO_SOURCE_FETCH_WAIT_SECONDS:-2700}"
fi

# OSMO's rsync daemon starts with the workflow. Wait until the local packager's
# readiness marker and complete archive agree with the checksum rendered into YAML.
source_ready=0
for _ in $(seq 1 150); do
    if [[ -f "${source_sync}/source.sha256" \
        && -f "${source_sync}/source.tar.gz" \
        && -f "${source_sync}/source.metadata" \
        && -f "${source_sync}/git-status.txt" ]]; then
        synced_sha="$(tr -d '[:space:]' < "${source_sync}/source.sha256")"
        if [[ "${synced_sha}" == "${OSMO_SOURCE_SHA256}" ]]; then
            actual_sha="$(sha256sum "${source_sync}/source.tar.gz" | awk '{print $1}')"
            status_sha="$(sed -n 's/^git_status_sha256=\([0-9a-f]\{64\}\)$/\1/p' "${source_sync}/source.metadata")"
            actual_status_sha="$(sha256sum "${source_sync}/git-status.txt" | awk '{print $1}')"
            if [[ "${actual_sha}" == "${OSMO_SOURCE_SHA256}" \
                && "${status_sha}" == "${actual_status_sha}" \
                && "$(grep -Fxc "source_sha256=${OSMO_SOURCE_SHA256}" "${source_sync}/source.metadata")" == 1 \
                && "$(grep -Fxc "commit=${OSMO_VCS_REF}" "${source_sync}/source.metadata")" == 1 ]]; then
                source_ready=1
                break
            fi
        fi
    fi
    sleep 2
done
if (( source_ready == 0 )); then
    echo "Timed out waiting for checksum-complete source rsync ${OSMO_SOURCE_SHA256}." >&2
    exit 3
fi

# A root-level acknowledgement avoids the OSMO 6.3 client's nested-download
# path bug. The submit helper only returns after receiving this digest, which
# proves that the task independently validated every source payload file.
source_acceptance_tmp="${source_acceptance}.tmp"
printf '%s\n' "${OSMO_SOURCE_SHA256}" > "${source_acceptance_tmp}"
mv "${source_acceptance_tmp}" "${source_acceptance}"

if [[ -e "${source_root}" ]]; then
    echo "Refusing to overwrite an existing source extraction at ${source_root}." >&2
    exit 4
fi
mkdir -p "${source_root}"
tar --extract --gzip --file="${source_sync}/source.tar.gz" --directory="${source_root}"
test -f "${source_root}/pyproject.toml"
test -f "${source_root}/source/isaaclab_tasks/isaaclab_tasks/contrib/cable_routing/__init__.py"
test -x "${image_root}/_isaac_sim/python.sh"
runtime_files=(
    run_yam_cable_routing_osmo.sh
    yam_cable_routing_rank_entrypoint.py
    yam_cable_routing_preflight.py
)
if [[ "${OSMO_RUN_MODE}" == multinode ]]; then
    runtime_files+=(
        run_yam_cable_routing_multinode_node.sh
        yam_cable_routing_ddp_barrier.py
        yam_cable_routing_source_fetch.py
    )
fi
for runtime_file in "${runtime_files[@]}"; do
    case "${runtime_file}" in
        run_yam_cable_routing_osmo.sh)
            injected=/tmp/run-yam-cable-routing-osmo.sh
            ;;
        run_yam_cable_routing_multinode_node.sh)
            injected=/tmp/run-yam-cable-routing-multinode-node.sh
            ;;
        yam_cable_routing_ddp_barrier.py)
            injected=/tmp/yam-cable-routing-ddp-barrier.py
            ;;
        yam_cable_routing_rank_entrypoint.py)
            injected=/tmp/yam-cable-routing-rank-entrypoint.py
            ;;
        yam_cable_routing_preflight.py)
            injected=/tmp/yam-cable-routing-preflight.py
            ;;
        yam_cable_routing_source_fetch.py)
            injected=/tmp/yam-cable-routing-source-fetch.py
            ;;
    esac
    if ! cmp -s "${injected}" "${source_root}/docker/cluster/${runtime_file}"; then
        echo "Injected runtime file does not match checksum-gated source: ${runtime_file}" >&2
        exit 4
    fi
done
ln -s "${image_root}/_isaac_sim" "${source_root}/_isaac_sim"

grep -Fx "source_sha256=${OSMO_SOURCE_SHA256}" "${source_sync}/source.metadata" >/dev/null
grep -Fx "commit=${OSMO_VCS_REF}" "${source_sync}/source.metadata" >/dev/null
install -m 0644 "${source_sync}/source.metadata" "${provenance}/source.metadata"
install -m 0644 "${source_sync}/git-status.txt" "${provenance}/git-status.txt"
printf '%s  %s\n' "${OSMO_SOURCE_SHA256}" source.tar.gz > "${provenance}/source.sha256"

python="${source_root}/_isaac_sim/python.sh"
# The release image contains an older Newton stack than current Isaac Lab
# develop. Install the exact source-compatible core packages into an isolated
# overlay instead of mutating the image environment or accepting an unpinned
# latest build.
install_newton_runtime() {
    "${python}" -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --no-deps \
        --retries 10 \
        --timeout 120 \
        --target "${runtime_deps}" \
        --extra-index-url https://pypi.nvidia.com \
        "warp-lang==${OSMO_WARP_VERSION}" \
        "mujoco==${OSMO_MUJOCO_VERSION}" \
        "mujoco-warp==${OSMO_MUJOCO_VERSION}" \
        "newton-usd-schemas==${OSMO_NEWTON_USD_SCHEMAS_VERSION}" \
        "newton[sim,importers] @ git+https://github.com/newton-physics/newton.git@${OSMO_NEWTON_REVISION}"
}

runtime_install_attempt=1
until install_newton_runtime; do
    if (( runtime_install_attempt >= 3 )); then
        echo "Pinned Newton runtime installation failed after ${runtime_install_attempt} attempts." >&2
        exit 5
    fi
    runtime_install_attempt="$(( runtime_install_attempt + 1 ))"
    echo \
        "Pinned Newton runtime installation failed; retrying attempt ${runtime_install_attempt} of 3." \
        >&2
    sleep 5
done
printf 'runtime_install_attempts=%s\n' "${runtime_install_attempt}" > "${provenance}/runtime-install-attempts.txt"
source_pythonpath="$(
    find "${source_root}/source" -mindepth 1 -maxdepth 1 -type d -print \
        | sort \
        | paste -sd:
)"
export ISAACLAB_PATH="${source_root}"
export PYTHONPATH="${runtime_deps}:${source_pythonpath}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export ACCEPT_EULA=Y
# Training uses the bundled YAM and public Isaac asset fallback. Never allow a
# headless worker to enter interactive authentication for the private server.
export NO_NUCLEUS=1
export OMP_NUM_THREADS="$(( OSMO_NUM_CPU / OSMO_NUM_GPU ))"
if (( OMP_NUM_THREADS < 1 )); then
    export OMP_NUM_THREADS=1
fi
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export WANDB_DIR="${live_results}/wandb"
export WANDB_ENTITY="${OSMO_WANDB_ENTITY}"
export WANDB_USERNAME="${OSMO_WANDB_ENTITY}"
export WANDB_PROJECT="${OSMO_WANDB_PROJECT}"
export WANDB_INIT_TIMEOUT=120
export WANDB_MODE=online
export WANDB_RESUME=never
export WANDB_RUN_GROUP="${OSMO_WORKFLOW_ID}"
export WANDB_SILENT=true

"${python}" - <<'PY' | tee "${provenance}/newton-runtime.txt"
from importlib.metadata import version
import os

import newton
import warp

expected = {
    "warp-lang": os.environ["OSMO_WARP_VERSION"],
    "mujoco": os.environ["OSMO_MUJOCO_VERSION"],
    "mujoco-warp": os.environ["OSMO_MUJOCO_VERSION"],
    "newton-usd-schemas": os.environ["OSMO_NEWTON_USD_SCHEMAS_VERSION"],
}
for distribution, expected_version in expected.items():
    actual_version = version(distribution)
    if actual_version != expected_version:
        raise SystemExit(f"{distribution} version {actual_version} != {expected_version}")
    print(f"{distribution}={actual_version}")
if not hasattr(newton, "ModelFlags"):
    raise SystemExit("Pinned Newton runtime does not expose ModelFlags")
print(f"newton={version('newton')}")
print(f"newton_revision={os.environ['OSMO_NEWTON_REVISION']}")
print(f"warp_module={warp.__file__}")
print("newton_model_flags=available")
PY

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "OSMO credential 'wandb' did not inject WANDB_API_KEY." >&2
    exit 5
fi
WANDB_API_KEY="$(printf '%s' "${WANDB_API_KEY}" | tr -d '[:space:]')"
export WANDB_API_KEY
if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "OSMO credential 'wandb' injected an empty WANDB_API_KEY." >&2
    exit 5
fi

printf \
    'workflow_id=%s\nimage=%s\nsource_sha256=%s\nvcs_ref=%s\nrun_mode=%s\nworker_index=%s\ncampaign_size=%s\nnode_rank=%s\nnum_nodes=%s\ntotal_gpu=%s\nnum_gpu=%s\nnum_cpu=%s\nnum_envs_per_process=%s\nglobal_num_envs=%s\nmax_iterations=%s\n' \
    "${OSMO_WORKFLOW_ID}" \
    "${OSMO_IMAGE_REF}" \
    "${OSMO_SOURCE_SHA256}" \
    "${OSMO_VCS_REF}" \
    "${OSMO_RUN_MODE}" \
    "${OSMO_CAMPAIGN_INDEX}" \
    "${OSMO_CAMPAIGN_SIZE}" \
    "${OSMO_NODE_RANK:-0}" \
    "${OSMO_NUM_NODES:-1}" \
    "${OSMO_TOTAL_GPU}" \
    "${OSMO_NUM_GPU}" \
    "${OSMO_NUM_CPU}" \
    "${OSMO_NUM_ENVS}" \
    "$(( OSMO_NUM_ENVS * OSMO_TOTAL_GPU ))" \
    "${OSMO_MAX_ITERATIONS}" \
    > "${provenance}/run.env"

nvidia-smi -L | tee "${provenance}/nvidia-smi-list.txt"
nvidia-smi topo -m | tee "${provenance}/nvidia-smi-topology.txt"
"${python}" -m pip freeze > "${provenance}/pip-freeze.txt"

run_wandb_preflight=1
if [[ "${OSMO_RUN_MODE}" == multinode && "${OSMO_NODE_RANK:-0}" != 0 ]]; then
    run_wandb_preflight=0
fi
if (( run_wandb_preflight != 0 )) && ! timeout 120 "${python}" - <<'PY' > "${provenance}/wandb-preflight.txt"
import os
import wandb

if not wandb.login(key=os.environ["WANDB_API_KEY"], verify=True, relogin=True):
    raise SystemExit("W&B rejected the configured OSMO credential")
api = wandb.Api(timeout=30)
project = api.project(os.environ["OSMO_WANDB_PROJECT"], entity=os.environ["OSMO_WANDB_ENTITY"])
if project.entity != os.environ["OSMO_WANDB_ENTITY"]:
    raise SystemExit("W&B returned a project under an unexpected entity")
print(f"wandb_sdk={wandb.__version__}")
print("wandb_auth=verified")
print("wandb_project_access=verified")
PY
then
    echo "W&B authentication preflight failed; refusing an unlogged training run." >&2
    exit 6
fi
if (( run_wandb_preflight == 0 )); then
    printf 'wandb_auth=verified_by_global_rank_zero\n' > "${provenance}/wandb-preflight.txt"
fi
cat "${provenance}/wandb-preflight.txt"

monitor_gpus() {
    printf 'timestamp_utc,index,name,uuid,gpu_util_percent,memory_util_percent,memory_used_mib,memory_total_mib,power_w\n'
    while true; do
        timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        while IFS= read -r sample; do
            printf '%s,%s\n' "${timestamp}" "${sample}"
        done < <(
            nvidia-smi \
                --query-gpu=index,name,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
                --format=csv,noheader,nounits
        )
        sleep 30
    done
}
monitor_gpus >> "${provenance}/gpu-utilization.csv" &
gpu_monitor_pid="$!"

visible_devices="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "${visible_devices}" ]]; then
    IFS=',' read -r -a device_tokens <<< "${visible_devices}"
else
    mapfile -t device_tokens < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
fi
device_count="$(printf '%s\n' "${device_tokens[@]}" | wc -l)"
if (( device_count != OSMO_NUM_GPU )); then
    echo "Expected ${OSMO_NUM_GPU} scheduler-visible GPUs, found ${device_count}." >&2
    exit 7
fi

# GPU 0 constructs and steps the complete environment before any training process starts.
# Its compiled Warp and downloaded OmniClient caches then seed rank-private copies.
export WARP_CACHE_PATH="${cache_seed}/warp"
export OMNICLIENT_HUB_CACHE_DIR="${cache_seed}/omniclient"
preflight_device="${device_tokens[0]//[[:space:]]/}"
preflight_seed="${OSMO_SEED}"
if [[ "${OSMO_RUN_MODE}" == campaign ]]; then
    preflight_seed="$(( OSMO_SEED + OSMO_CAMPAIGN_INDEX ))"
elif [[ "${OSMO_RUN_MODE}" == multinode ]]; then
    preflight_seed="$(( OSMO_SEED + OSMO_NODE_RANK * OSMO_NUM_GPU ))"
fi
run_newton_preflight() {
    CUDA_VISIBLE_DEVICES="${preflight_device}" \
        "${python}" /tmp/yam-cable-routing-preflight.py \
        --task "${OSMO_PREFLIGHT_TASK}" \
        --num_envs "${OSMO_PREFLIGHT_NUM_ENVS}" \
        --reset_rounds "${OSMO_PREFLIGHT_RESET_ROUNDS}" \
        --warmup_steps "${OSMO_PREFLIGHT_WARMUP_STEPS}" \
        --benchmark_steps "${OSMO_PREFLIGHT_BENCHMARK_STEPS}" \
        --min_env_steps_per_second "${OSMO_MIN_ENV_STEPS_PER_SECOND}" \
        --seed "${preflight_seed}" \
        --output "${preflight_results}/report.json"
}

# Newton's isolated IK prototype importer can very rarely abort inside native
# USD parsing before simulation begins. A fresh process is safe to retry once;
# every Python assertion and all other exit codes remain fail-closed.
preflight_attempt=1
while true; do
    set +e
    run_newton_preflight
    preflight_status="$?"
    set -e
    if (( preflight_status == 0 )); then
        break
    fi
    if (( preflight_attempt == 1 && (preflight_status == 134 || preflight_status == 139) )); then
        echo \
            "Newton preflight exited ${preflight_status} during native import; retrying once in a fresh process." \
            >&2
        rm -f "${preflight_results}/report.json"
        preflight_attempt=2
        continue
    fi
    exit "${preflight_status}"
done
printf 'preflight_attempts=%s\n' "${preflight_attempt}" > "${provenance}/preflight-attempts.txt"

# All eight OSMO tasks perform source/runtime validation and the Newton preflight
# independently. Release them together only after the slowest task is ready so
# static torchrun never consumes its rendezvous timeout on setup skew.
if [[ "${OSMO_RUN_MODE}" == multinode ]]; then
    "${python}" /tmp/yam-cable-routing-ddp-barrier.py report \
        --host "${OSMO_MASTER_ADDR}" \
        --port "${OSMO_COMPLETION_PORT}" \
        --node-rank "${OSMO_NODE_RANK}" \
        --phase ready \
        --workflow-id "${OSMO_WORKFLOW_ID}" \
        --timeout-seconds "${OSMO_BARRIER_STARTUP_TIMEOUT_SECONDS}"
fi

train_script="${source_root}/scripts/reinforcement_learning/train.py"
rank_entrypoint=/tmp/yam-cable-routing-rank-entrypoint.py
test -f "${train_script}"
test -f "${rank_entrypoint}"
cd "${source_root}"

run_single() {
    local device_token="$1"
    local task="$2"
    local seed="$3"
    local run_suffix="$4"
    local slot_index="$5"
    local slot_label="$6"
    local rank_root="/tmp/${OSMO_WORKFLOW_ID}-independent-${slot_label}"
    local slot_training_root="${training_runs}/${slot_label}"
    local slot_warp="${rank_root}/warp-cache"
    local slot_omni="${rank_root}/omniclient-cache"
    mkdir -p "${rank_root}" "${slot_training_root}" "${live_results}/wandb/${slot_label}"
    ln -s "${slot_training_root}" "${rank_root}/logs"
    cp -a "${cache_seed}/warp" "${slot_warp}"
    cp -a "${cache_seed}/omniclient" "${slot_omni}"
    (
        cd "${rank_root}"
        export CUDA_VISIBLE_DEVICES="${device_token}"
        export LOCAL_RANK=0
        export RANK=0
        export WORLD_SIZE=1
        export TMPDIR="${rank_root}"
        export TMP="${rank_root}"
        export TEMP="${rank_root}"
        export WARP_CACHE_PATH="${slot_warp}"
        export OMNICLIENT_HUB_CACHE_DIR="${slot_omni}"
        export TORCH_EXTENSIONS_DIR="${rank_root}/torch-extensions"
        export XDG_CACHE_HOME="${rank_root}/xdg-cache"
        export WANDB_DIR="${live_results}/wandb/${slot_label}"
        export WANDB_RUN_ID="${OSMO_WORKFLOW_ID}-${slot_label}"
        sleep "$(( slot_index * OSMO_RANK_STARTUP_STAGGER_SECONDS ))"
        "${python}" "${train_script}" \
            --rl_library rsl_rl \
            --task "${task}" \
            --device cuda:0 \
            --num_envs "${OSMO_NUM_ENVS}" \
            --max_iterations "${OSMO_MAX_ITERATIONS}" \
            --seed "${seed}" \
            --logger wandb \
            --log_project_name "${OSMO_WANDB_PROJECT}" \
            --run_name "${OSMO_RUN_NAME}-${run_suffix}" \
            --viz none \
            "env.commands.route.reset_replay.seed=${seed}" \
            2>&1 | sed -u "s/^/[${slot_label}] /" | tee -a "${console_logs}/${slot_label}.log"
    )
}

case "${OSMO_RUN_MODE}" in
    smoke)
        if (( OSMO_NUM_GPU != 1 )); then
            echo "Smoke mode requires num_gpu=1." >&2
            exit 9
        fi
        printf 'slot\tgpu\ttask\tseed\nsmoke\t0\t%s\t%s\n' \
            "${OSMO_TASK}" "${OSMO_SEED}" > "${provenance}/training-plan.tsv"
        run_single "${preflight_device}" "${OSMO_TASK}" "${OSMO_SEED}" smoke 0 smoke
        ;;
    campaign)
        if (( OSMO_NUM_GPU != 1 )); then
            echo "Each independent curriculum worker requires exactly one GPU." >&2
            exit 9
        fi
        if (( OSMO_CAMPAIGN_SIZE < 1 || OSMO_CAMPAIGN_SIZE > 32 )); then
            echo "The independent campaign requires one through 32 workers." >&2
            exit 9
        fi
        if (( OSMO_CAMPAIGN_INDEX < 0 || OSMO_CAMPAIGN_INDEX >= OSMO_CAMPAIGN_SIZE )); then
            echo "OSMO_CAMPAIGN_INDEX is outside the rendered campaign." >&2
            exit 9
        fi
        if (( OSMO_CAMPAIGN_SIZE != OSMO_TOTAL_GPU )); then
            echo "The one-GPU worker count must equal OSMO_TOTAL_GPU." >&2
            exit 9
        fi
        slot="${OSMO_CAMPAIGN_INDEX}"
        campaign_seed="$(( OSMO_SEED + slot ))"
        case "${OSMO_CAMPAIGN_TASK_MODE}" in
            round_robin)
                campaign_tasks=(
                    IsaacContrib-CableRouting-YAM-Peg0-CCW
                    IsaacContrib-CableRouting-YAM-Peg1-CW
                    IsaacContrib-CableRouting-YAM-Tier1-Pegs
                    IsaacContrib-CableRouting-YAM
                )
                campaign_goal_labels=(peg0-ccw peg1-cw tier1 multigoal)
                goal_index="$(( slot % 4 ))"
                campaign_task="${campaign_tasks[goal_index]}"
                campaign_goal_label="${campaign_goal_labels[goal_index]}"
                ;;
            focused)
                campaign_task="${OSMO_TASK}"
                case "${campaign_task}" in
                    IsaacContrib-CableRouting-YAM-Peg0-CCW)
                        campaign_goal_label=peg0-ccw
                        ;;
                    IsaacContrib-CableRouting-YAM-Peg1-CW)
                        campaign_goal_label=peg1-cw
                        ;;
                    IsaacContrib-CableRouting-YAM-Tier1-Pegs)
                        campaign_goal_label=tier1
                        ;;
                    IsaacContrib-CableRouting-YAM-SevenGoals)
                        campaign_goal_label=seven-goals
                        ;;
                    IsaacContrib-CableRouting-YAM)
                        campaign_goal_label=multigoal
                        ;;
                    *)
                        echo "Unsupported focused cable-routing task: ${campaign_task}" >&2
                        exit 9
                        ;;
                esac
                ;;
            *)
                echo \
                    "Unsupported OSMO_CAMPAIGN_TASK_MODE=${OSMO_CAMPAIGN_TASK_MODE}; " \
                    "expected round_robin or focused." >&2
                exit 9
                ;;
        esac
        campaign_slot_label="${campaign_goal_label}-s${campaign_seed}"
        printf 'slot\tworker\tgpu\ttask\tseed\n' > "${provenance}/training-plan.tsv"
        printf 'slot\texit_code\n' > "${provenance}/campaign-status.tsv"
        device_token="${device_tokens[0]//[[:space:]]/}"
        printf '%s\t%s\t0\t%s\t%s\n' \
            "${campaign_slot_label}" \
            "${OSMO_CAMPAIGN_INDEX}" \
            "${campaign_task}" \
            "${campaign_seed}" \
            >> "${provenance}/training-plan.tsv"
        if run_single \
            "${device_token}" \
            "${campaign_task}" \
            "${campaign_seed}" \
            "${campaign_slot_label}" \
            "${OSMO_CAMPAIGN_INDEX}" \
            "${campaign_slot_label}"; then
            campaign_status=0
        else
            campaign_status="$?"
            echo "Campaign process ${campaign_slot_label} failed." >&2
        fi
        printf '%s\t%s\n' "${campaign_slot_label}" "${campaign_status}" \
            >> "${provenance}/campaign-status.tsv"
        if (( campaign_status != 0 )); then
            exit 10
        fi
        ;;
    ddp)
        printf 'slot\tgpu\ttask\tseed\nddp\tall\t%s\t%s\n' \
            "${OSMO_TASK}" "${OSMO_SEED}" > "${provenance}/training-plan.tsv"
        ddp_root="/tmp/${OSMO_WORKFLOW_ID}-ddp"
        mkdir -p "${ddp_root}" "${training_runs}/ddp"
        ln -s "${training_runs}/ddp" "${ddp_root}/logs"
        export OSMO_RANK_RUNTIME_ROOT="/tmp/${OSMO_WORKFLOW_ID}"
        export OSMO_WARP_CACHE_SEED="${cache_seed}/warp"
        export OSMO_OMNICLIENT_CACHE_SEED="${cache_seed}/omniclient"
        export OSMO_RANK_STARTUP_STAGGER_S="${OSMO_RANK_STARTUP_STAGGER_SECONDS}"
        export WANDB_RUN_ID="${OSMO_WORKFLOW_ID}-ddp"
        (
            cd "${ddp_root}"
            "${python}" -m torch.distributed.run \
                --nproc_per_node "${OSMO_NUM_GPU}" \
                --master_port "${OSMO_MASTER_PORT}" \
                --local_ranks_filter 0 \
                --log_dir "${console_logs}/torchrun-ranks" \
                --tee 3 \
                "${rank_entrypoint}" \
                "${train_script}" \
                --rl_library rsl_rl \
                --task "${OSMO_TASK}" \
                --num_envs "${OSMO_NUM_ENVS}" \
                --max_iterations "${OSMO_MAX_ITERATIONS}" \
                --seed "${OSMO_SEED}" \
                --logger wandb \
                --log_project_name "${OSMO_WANDB_PROJECT}" \
                --run_name "${OSMO_RUN_NAME}-ddp" \
                --viz none \
                "env.commands.route.reset_replay.seed=${OSMO_SEED}" \
                --distributed
        ) 2>&1 | tee -a "${console_logs}/ddp.log"
        ;;
    multinode)
        if (( OSMO_NUM_NODES < 2 )); then
            echo "Multi-node DDP requires at least two nodes." >&2
            exit 9
        fi
        if (( OSMO_NODE_RANK < 0 || OSMO_NODE_RANK >= OSMO_NUM_NODES )); then
            echo "OSMO_NODE_RANK is outside the configured node world." >&2
            exit 9
        fi
        if (( OSMO_NUM_NODES * OSMO_NUM_GPU != OSMO_TOTAL_GPU )); then
            echo "The multi-node topology does not match OSMO_TOTAL_GPU." >&2
            exit 9
        fi
        rollout_samples_per_rank="$(( OSMO_NUM_ENVS * 32 ))"
        if (( OSMO_NUM_MINI_BATCHES < 1 || rollout_samples_per_rank % OSMO_NUM_MINI_BATCHES != 0 )); then
            echo "OSMO_NUM_MINI_BATCHES must evenly divide each rank's 32-step rollout." >&2
            exit 9
        fi
        if [[ -z "${OSMO_MASTER_ADDR}" ]]; then
            echo "OSMO_MASTER_ADDR is required for multi-node DDP." >&2
            exit 9
        fi
        if [[ "${OSMO_TASK}" == IsaacContrib-CableRouting-YAM-Tier1-Pegs ]]; then
            multinode_goal_label=tier1
        elif [[ "${OSMO_TASK}" == IsaacContrib-CableRouting-YAM-SevenGoals ]]; then
            multinode_goal_label=seven-goals
        else
            echo "Unsupported multi-node cable-routing task: ${OSMO_TASK}" >&2
            exit 9
        fi
        printf 'slot\tnode_rank\tgpus\tworld_size\ttask\tseed\nddp\t%s\t%s\t%s\t%s\t%s\n' \
            "${OSMO_NODE_RANK}" \
            "${OSMO_NUM_GPU}" \
            "${OSMO_TOTAL_GPU}" \
            "${OSMO_TASK}" \
            "${OSMO_SEED}" \
            > "${provenance}/training-plan.tsv"
        ddp_root="/tmp/${OSMO_WORKFLOW_ID}-ddp"
        mkdir -p "${ddp_root}" "${training_runs}/ddp" "${console_logs}/torchrun-ranks"
        ln -s "${training_runs}/ddp" "${ddp_root}/logs"
        export OSMO_RANK_RUNTIME_ROOT="/tmp/${OSMO_WORKFLOW_ID}"
        export OSMO_WARP_CACHE_SEED="${cache_seed}/warp"
        export OSMO_OMNICLIENT_CACHE_SEED="${cache_seed}/omniclient"
        export OSMO_RANK_STARTUP_STAGGER_S="${OSMO_RANK_STARTUP_STAGGER_SECONDS}"
        export OSMO_RESET_REPLAY_SEED_PER_RANK=1
        export WANDB_RUN_ID="${OSMO_WORKFLOW_ID}-${multinode_goal_label}-s${OSMO_SEED}-ddp${OSMO_TOTAL_GPU}"
        (
            cd "${ddp_root}"
            "${python}" -m torch.distributed.run \
                --nnodes "${OSMO_NUM_NODES}" \
                --nproc_per_node "${OSMO_NUM_GPU}" \
                --node_rank "${OSMO_NODE_RANK}" \
                --master_addr "${OSMO_MASTER_ADDR}" \
                --master_port "${OSMO_MASTER_PORT}" \
                --local_ranks_filter 0 \
                --log_dir "${console_logs}/torchrun-ranks" \
                --tee 3 \
                "${rank_entrypoint}" \
                "${train_script}" \
                --rl_library rsl_rl \
                --task "${OSMO_TASK}" \
                --num_envs "${OSMO_NUM_ENVS}" \
                --max_iterations "${OSMO_MAX_ITERATIONS}" \
                --seed "${OSMO_SEED}" \
                --logger wandb \
                --log_project_name "${OSMO_WANDB_PROJECT}" \
                --run_name "${OSMO_RUN_NAME}-${multinode_goal_label}-s${OSMO_SEED}-ddp${OSMO_TOTAL_GPU}" \
                --viz none \
                "env.commands.route.reset_replay.seed=${OSMO_SEED}" \
                "agent.algorithm.num_mini_batches=${OSMO_NUM_MINI_BATCHES}" \
                --distributed
        ) 2>&1 | tee -a "${console_logs}/ddp.log"
        ;;
    *)
        echo "Unsupported OSMO_RUN_MODE=${OSMO_RUN_MODE}; expected smoke, campaign, ddp, or multinode." >&2
        exit 11
        ;;
esac

# Contact-list overflow drops constraints and invalidates the resulting policy,
# even if the trainer itself exits successfully. Fail closed so a noisy run can
# never be published as a valid checkpoint campaign.
if grep -R -E -q \
    'Per-body (rigid|particle) contact buffer overflowed|Maximum logging rate exceeded' \
    "${console_logs}"; then
    echo "Newton contact overflow or rate-limited warnings occurred; checkpoints are invalid." >&2
    exit 15
fi

# BEGIN verify_training_checkpoints
verify_training_checkpoints() {
    local expected_final_checkpoint
    local inventory_tmp="${provenance}/checkpoint-inventory.tsv.tmp"
    local manifest_tmp="${provenance}/checkpoint-manifest.sha256.tmp"
    local checkpoint_count=0
    local checkpoint
    local checkpoint_bytes
    local checkpoint_kind
    local checkpoint_sha
    local expected_slot_count
    local relative_checkpoint
    local slot
    local -a expected_slots
    local -a slot_checkpoints
    local -a slot_final_checkpoints

    if (( OSMO_MAX_ITERATIONS < 1 )); then
        echo "OSMO_MAX_ITERATIONS must be positive before checkpoint validation." >&2
        return 14
    fi
    expected_final_checkpoint="model_$(( OSMO_MAX_ITERATIONS - 1 )).pt"
    case "${OSMO_RUN_MODE}" in
        smoke)
            expected_slots=(smoke)
            expected_slot_count=1
            ;;
        campaign)
            if [[ -z "${campaign_slot_label:-}" ]]; then
                echo "Campaign slot label is unavailable during checkpoint validation." >&2
                return 14
            fi
            expected_slots=("${campaign_slot_label}")
            expected_slot_count=1
            ;;
        ddp|multinode)
            expected_slots=(ddp)
            expected_slot_count=1
            ;;
        *)
            echo "Cannot validate checkpoints for unsupported mode ${OSMO_RUN_MODE}." >&2
            return 14
            ;;
    esac

    printf 'slot\tkind\tbytes\tsha256\tpath\n' > "${inventory_tmp}"
    : > "${manifest_tmp}"
    for slot in "${expected_slots[@]}"; do
        mapfile -d '' -t slot_checkpoints < <(
            find "${training_runs}/${slot}" -type f -name 'model_*.pt' -print0 | sort -z
        )
        mapfile -d '' -t slot_final_checkpoints < <(
            find "${training_runs}/${slot}" -type f -name "${expected_final_checkpoint}" -print0 | sort -z
        )
        if [[ ! -v 'slot_final_checkpoints[0]' ]]; then
            echo "Missing final ${expected_final_checkpoint} checkpoint for training slot ${slot}." >&2
            rm -f "${inventory_tmp}" "${manifest_tmp}"
            return 14
        fi
        for checkpoint in "${slot_checkpoints[@]}"; do
            if [[ ! -s "${checkpoint}" ]]; then
                echo "Training checkpoint is empty: ${checkpoint}" >&2
                rm -f "${inventory_tmp}" "${manifest_tmp}"
                return 14
            fi
            relative_checkpoint="${checkpoint#"${live_results}/"}"
            checkpoint_sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
            checkpoint_bytes="$(stat --format=%s "${checkpoint}")"
            checkpoint_kind=periodic
            if [[ "${checkpoint##*/}" == "${expected_final_checkpoint}" ]]; then
                checkpoint_kind=final
            fi
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "${slot}" \
                "${checkpoint_kind}" \
                "${checkpoint_bytes}" \
                "${checkpoint_sha}" \
                "${relative_checkpoint}" \
                >> "${inventory_tmp}"
            printf '%s  %s\n' "${checkpoint_sha}" "${relative_checkpoint}" >> "${manifest_tmp}"
            checkpoint_count="$(( checkpoint_count + 1 ))"
        done
    done
    if (( checkpoint_count < expected_slot_count )); then
        echo "Found ${checkpoint_count} checkpoints for ${expected_slot_count} training slots." >&2
        rm -f "${inventory_tmp}" "${manifest_tmp}"
        return 14
    fi

    mv "${inventory_tmp}" "${provenance}/checkpoint-inventory.tsv"
    mv "${manifest_tmp}" "${provenance}/checkpoint-manifest.sha256"
    if ! (
        cd "${live_results}"
        sha256sum --check "training-artifacts/run-info/checkpoint-manifest.sha256"
    ); then
        echo "Training checkpoint manifest failed immediate verification." >&2
        return 14
    fi
    printf 'run_mode=%s\nexpected_slots=%s\ncheckpoint_count=%s\nexpected_final_checkpoint=%s\n' \
        "${OSMO_RUN_MODE}" \
        "${expected_slot_count}" \
        "${checkpoint_count}" \
        "${expected_final_checkpoint}" \
        > "${provenance}/checkpoint-validation.txt"
}
# END verify_training_checkpoints

if [[ "${OSMO_RUN_MODE}" != multinode || "${OSMO_NODE_RANK:-0}" == 0 ]]; then
    verify_training_checkpoints
    checkpoint_manifest_sha="$(sha256sum "${provenance}/checkpoint-manifest.sha256" | awk '{print $1}')"
    printf 'training_complete_utc=%s\ncheckpoint_manifest_sha256=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "${checkpoint_manifest_sha}" \
        > "${provenance}/training-complete.txt"
else
    printf 'training_complete_utc=%s\nnode_rank=%s\ncheckpoint_owner_node_rank=0\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "${OSMO_NODE_RANK}" \
        > "${provenance}/training-complete.txt"
fi
