#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Package, validate, render, and (only with --submit) submit YAM cable-routing training.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
isaaclab_root="$(git -C "${script_dir}" rev-parse --show-toplevel)"
workflow_file="${script_dir}/yam_cable_routing_osmo_workflow.yaml"
mode="${1:-validate}"
if (( $# > 0 )); then
    shift
fi

pool=isaac-lab-l40s-03
priority=NORMAL
submit=0
task=IsaacContrib-CableRouting-YAM
campaign_task_mode=round_robin
wandb_project=yam-cable-routing-newton
wandb_entity=nvidia-isaac
min_env_steps_per_second=0
num_mini_batches=4
sync_root="$(dirname "${isaaclab_root}")/.osmo-yam-cable-routing-sync"
source_upload_wait_seconds=86400
source_upload_retry_seconds=900

case "${mode}" in
    validate|smoke)
        run_mode=smoke
        num_gpu=1
        # Use the same per-GPU share as the campaign so this diagnostic fits
        # both H100 and L40 nodes without fragmenting a mostly occupied node.
        num_cpu=4
        memory_gi=24
        storage_gi=32
        num_envs=16
        max_iterations=2
        seed=17
        preflight_num_envs=8
        preflight_warmup_steps=16
        preflight_benchmark_steps=64
        ;;
    campaign)
        run_mode=campaign
        num_gpu=24
        num_cpu=4
        memory_gi=24
        storage_gi=32
        num_envs=256
        max_iterations=3000
        seed=41
        preflight_num_envs=256
        preflight_warmup_steps=32
        preflight_benchmark_steps=128
        ;;
    ddp)
        run_mode=ddp
        num_gpu=8
        num_cpu=64
        memory_gi=256
        storage_gi=512
        num_envs=256
        max_iterations=3000
        seed=41
        preflight_num_envs=256
        preflight_warmup_steps=32
        preflight_benchmark_steps=128
        ;;
    multinode)
        run_mode=multinode
        workflow_file="${script_dir}/yam_cable_routing_multinode_osmo_workflow.yaml"
        num_gpu=32
        gpus_per_node=4
        num_cpu=16
        memory_gi=96
        storage_gi=128
        task=IsaacContrib-CableRouting-YAM-SevenGoals
        campaign_task_mode=focused
        num_envs=256
        # Preserve the winning Tier-1 PPO optimizer cadence. The synchronized
        # ranks weak-scale its mini-batch to prioritize cross-node throughput.
        num_mini_batches=4
        max_iterations=3000
        seed=47
        preflight_num_envs=256
        preflight_warmup_steps=32
        preflight_benchmark_steps=128
        ;;
    *)
        echo "Usage: $0 {validate|smoke|campaign|ddp|multinode} [options]" >&2
        exit 2
        ;;
esac

while (( $# > 0 )); do
    case "$1" in
        --submit)
            submit=1
            shift
            ;;
        --pool)
            pool="$2"
            shift 2
            ;;
        --priority)
            priority="$2"
            shift 2
            ;;
        --num-gpu)
            num_gpu="$2"
            shift 2
            ;;
        --gpus-per-node)
            gpus_per_node="$2"
            shift 2
            ;;
        --num-envs)
            num_envs="$2"
            shift 2
            ;;
        --iterations)
            max_iterations="$2"
            shift 2
            ;;
        --num-mini-batches)
            num_mini_batches="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --task)
            task="$2"
            campaign_task_mode=focused
            shift 2
            ;;
        --wandb-project)
            wandb_project="$2"
            shift 2
            ;;
        --wandb-entity)
            wandb_entity="$2"
            shift 2
            ;;
        --min-env-steps-per-second)
            min_env_steps_per_second="$2"
            shift 2
            ;;
        --sync-root)
            sync_root="$2"
            shift 2
            ;;
        --source-upload-wait-seconds)
            source_upload_wait_seconds="$2"
            shift 2
            ;;
        --source-upload-retry-seconds)
            source_upload_retry_seconds="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if ! [[ "${num_gpu}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-gpu must be a positive integer." >&2
    exit 2
fi
if [[ "${run_mode}" == smoke && "${num_gpu}" != 1 ]]; then
    echo "Smoke mode requires --num-gpu 1." >&2
    exit 2
fi
case "${run_mode}" in
    campaign)
        if (( num_gpu > 32 )); then
            echo "The independent campaign supports at most 32 GPUs." >&2
            exit 2
        fi
        # One trainer per GPU allows OSMO to use partially occupied eight-GPU nodes.
        num_workers="${num_gpu}"
        gpus_per_worker=1
        ;;
    ddp)
        if (( num_gpu > 8 )); then
            echo "Single-node DDP supports at most eight GPUs." >&2
            exit 2
        fi
        num_workers=1
        gpus_per_worker="${num_gpu}"
        ;;
    multinode)
        if ! [[ "${gpus_per_node}" =~ ^[1-9][0-9]*$ ]]; then
            echo "--gpus-per-node must be a positive integer." >&2
            exit 2
        fi
        if (( num_gpu > 32 )); then
            echo "Multi-node DDP supports at most 32 GPUs." >&2
            exit 2
        fi
        if (( num_gpu % gpus_per_node != 0 )); then
            echo "--num-gpu must be divisible by --gpus-per-node." >&2
            exit 2
        fi
        num_nodes="$(( num_gpu / gpus_per_node ))"
        if (( num_nodes < 2 )); then
            echo "Multi-node DDP requires at least two nodes." >&2
            exit 2
        fi
        num_workers="${num_nodes}"
        gpus_per_worker="${gpus_per_node}"
        ;;
    *)
        num_workers=1
        gpus_per_worker=1
        ;;
esac
if [[ "${run_mode}" == campaign && "${campaign_task_mode}" == focused ]]; then
    case "${task}" in
        IsaacContrib-CableRouting-YAM-Peg0-CCW | \
            IsaacContrib-CableRouting-YAM-Peg1-CW | \
            IsaacContrib-CableRouting-YAM-Tier1-Pegs | \
            IsaacContrib-CableRouting-YAM-SevenGoals | \
            IsaacContrib-CableRouting-YAM)
            ;;
        *)
            echo "Unsupported focused cable-routing task: ${task}" >&2
            exit 2
            ;;
    esac
fi
if [[ "${run_mode}" == multinode ]]; then
    case "${task}" in
        IsaacContrib-CableRouting-YAM-Tier1-Pegs | IsaacContrib-CableRouting-YAM-SevenGoals)
            ;;
        *)
            echo "Multi-node training supports the Tier-1 and seven-goal cable-routing tasks." >&2
            exit 2
            ;;
    esac
fi
if [[ "${priority}" != HIGH && "${priority}" != NORMAL && "${priority}" != LOW ]]; then
    echo "--priority must be HIGH, NORMAL, or LOW." >&2
    exit 2
fi
if ! [[ "${num_envs}" =~ ^[1-9][0-9]*$ \
    && "${max_iterations}" =~ ^[1-9][0-9]*$ \
    && "${num_mini_batches}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-envs, --iterations, and --num-mini-batches must be positive integers." >&2
    exit 2
fi
if ! [[ "${source_upload_wait_seconds}" =~ ^[1-9][0-9]*$ \
    && "${source_upload_retry_seconds}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--source-upload-wait-seconds and --source-upload-retry-seconds must be positive integers." >&2
    exit 2
fi

osmo version
osmo profile list >/dev/null
credential_listing=""
credential_query_succeeded=0
for credential_attempt in 1 2 3 4 5; do
    if credential_listing="$(osmo credential list)"; then
        credential_query_succeeded=1
        break
    fi
    echo "OSMO credential query attempt ${credential_attempt} failed; retrying." >&2
    sleep 2
done
if (( credential_query_succeeded == 0 )); then
    echo "Unable to query OSMO credentials after five attempts." >&2
    exit 3
fi
if ! awk '$1 == "wandb" && $2 == "GENERIC" { found = 1 } END { exit !found }' <<< "${credential_listing}"; then
    echo "Required OSMO GENERIC credential 'wandb' is unavailable." >&2
    exit 3
fi

pool_json="$(osmo pool list --pool "${pool}" --mode free --format-type json)"
read -r pool_status quota_free total_free pool_platform < <(
    python3 -c '
import json, sys
data = json.load(sys.stdin)
pools = [pool for node_set in data["node_sets"] for pool in node_set["pools"]]
if len(pools) != 1:
    raise SystemExit(f"Expected exactly one pool, found {len(pools)}")
pool = pools[0]
usage = pool["resource_usage"]
print(pool["status"], usage["quota_free"], usage["total_free"], pool["default_platform"])
' <<< "${pool_json}"
)
if [[ "${pool_status}" != ONLINE ]]; then
    echo "Pool ${pool} is ${pool_status}, not ONLINE." >&2
    exit 4
fi
effective_free="$(( quota_free < total_free ? quota_free : total_free ))"
if (( effective_free < 0 )); then
    effective_free=0
fi
capacity_free="${effective_free}"
capacity_kind=effective
if [[ "${priority}" == LOW ]]; then
    capacity_free="${total_free}"
    capacity_kind=physical
    if (( capacity_free < 0 )); then
        capacity_free=0
    fi
fi
if (( capacity_free < num_gpu )); then
    if (( submit != 0 )); then
        echo "Pool ${pool} has ${capacity_free} ${capacity_kind} GPUs free; ${num_gpu} requested." >&2
        exit 4
    fi
    echo \
        "Warning: pool ${pool} currently has ${capacity_free} ${capacity_kind} GPUs free; " \
        "validating ${num_gpu} anyway." \
        >&2
fi
if [[ "${priority}" == LOW ]] && (( quota_free < num_gpu && total_free >= num_gpu )); then
    echo \
        "LOW priority bypasses quota: pool ${pool} has ${quota_free} quota GPUs free and " \
        "${total_free} physical GPUs free; the workflow may be preempted." \
        >&2
fi
if [[ ! "${pool_platform}" =~ ^(ovx-l40s|ovx-l40|dgx-h100)$ ]]; then
    echo "Pool ${pool} uses unsupported GPU platform ${pool_platform}." >&2
    exit 4
fi
printf 'pool=%s priority=%s quota_free=%s total_free=%s effective_free=%s platform=%s\n' \
    "${pool}" "${priority}" "${quota_free}" "${total_free}" "${effective_free}" "${pool_platform}"

sync_root="$(bash "${script_dir}/package_yam_cable_routing_source.sh" "${sync_root}")"
source_sha256="$(tr -d '[:space:]' < "${sync_root}/source.sha256")"
for source_file in source.tar.gz source.metadata git-status.txt source.sha256; do
    if [[ ! -s "${sync_root}/${source_file}" ]]; then
        echo "Packaged source file is missing or empty: ${sync_root}/${source_file}" >&2
        exit 5
    fi
    if git -C "${isaaclab_root}" check-ignore -q "${sync_root}/${source_file}" 2>/dev/null; then
        echo \
            "OSMO filters Git-ignored rsync inputs; choose a --sync-root outside the repository: ${sync_root}" \
            >&2
        exit 5
    fi
done
vcs_ref="$(git -C "${isaaclab_root}" rev-parse HEAD)"
timestamp="$(date -u +%Y%m%d-%H%M%S)"
workflow_name="yam-cable-${run_mode}-${num_gpu}g-bounded-${timestamp}"
run_name="${workflow_name}"

numeric_overrides=(
    "num_gpu=${num_gpu}"
    "num_cpu=${num_cpu}"
    "memory_gi=${memory_gi}"
    "storage_gi=${storage_gi}"
    "num_envs=${num_envs}"
    "max_iterations=${max_iterations}"
    "seed=${seed}"
    "min_env_steps_per_second=${min_env_steps_per_second}"
    "preflight_num_envs=${preflight_num_envs}"
    "preflight_warmup_steps=${preflight_warmup_steps}"
    "preflight_benchmark_steps=${preflight_benchmark_steps}"
)
string_overrides=(
    "workflow_name=${workflow_name}"
    "platform=${pool_platform}"
    "source_sha256=${source_sha256}"
    "vcs_ref=${vcs_ref}"
    "run_mode=${run_mode}"
    "task=${task}"
    "preflight_task=${task}"
    "run_name=${run_name}"
    "wandb_project=${wandb_project}"
    "wandb_entity=${wandb_entity}"
)
if [[ "${run_mode}" == multinode ]]; then
    rollout_samples_per_rank="$(( num_envs * 32 ))"
    if (( rollout_samples_per_rank % num_mini_batches != 0 )); then
        echo "Per-rank rollout samples must be divisible by --num-mini-batches." >&2
        exit 2
    fi
    numeric_overrides+=(
        "num_nodes=${num_nodes}"
        "gpus_per_node=${gpus_per_node}"
        "num_mini_batches=${num_mini_batches}"
    )
else
    numeric_overrides+=(
        "num_workers=${num_workers}"
        "gpus_per_worker=${gpus_per_worker}"
    )
    string_overrides+=("campaign_task_mode=${campaign_task_mode}")
fi

(
    cd "${script_dir}"
    osmo workflow validate "$(basename "${workflow_file}")" \
        --pool "${pool}" \
        --set "${numeric_overrides[@]}" \
        --set-string "${string_overrides[@]}"
    osmo workflow submit "$(basename "${workflow_file}")" \
        --pool "${pool}" \
        --priority "${priority}" \
        --dry-run \
        --set "${numeric_overrides[@]}" \
        --set-string "${string_overrides[@]}" \
        > "${sync_root}/rendered-${run_mode}.yaml"
)

printf 'validated=%s\nrendered=%s\nsource_sha256=%s\n' \
    "${workflow_file}" "${sync_root}/rendered-${run_mode}.yaml" "${source_sha256}"

if (( submit == 0 )); then
    if [[ "${mode}" != validate ]]; then
        echo "Dry-run only. Add --submit to create the OSMO workflow."
    fi
    exit 0
fi
if [[ "${mode}" == validate ]]; then
    echo "The validate mode never submits; choose smoke, campaign, ddp, or multinode with --submit." >&2
    exit 2
fi

submission_json="$(
    cd "${script_dir}"
    osmo workflow submit "$(basename "${workflow_file}")" \
        --pool "${pool}" \
        --priority "${priority}" \
        --format-type json \
        --set "${numeric_overrides[@]}" \
        --set-string "${string_overrides[@]}"
)"
printf '%s\n' "${submission_json}"

workflow_id="$(
    python3 -c '
import json
import sys

submission = json.load(sys.stdin)
for key in ("name", "workflow_id", "id", "uuid"):
    value = submission.get(key)
    if isinstance(value, str) and value:
        print(value)
        break
else:
    raise SystemExit("OSMO submission response did not contain a workflow identifier")
' <<< "${submission_json}"
)"

task_status() {
    local query_json="$1"
    local task_name="$2"
    python3 -c '
import json
import sys

query = json.load(sys.stdin)
task_name = sys.argv[1]
for task in query.get("tasks", []):
    if task.get("name") == task_name:
        print(task.get("status", "UNKNOWN"))
        raise SystemExit
for group in query.get("groups", []):
    for task in group.get("tasks", []):
        if task.get("name") == task_name:
            print(task.get("status", "UNKNOWN"))
            raise SystemExit
raise SystemExit(f"{task_name} task missing from OSMO workflow query")
' "${task_name}" <<< "${query_json}"
}

query_workflow() {
    local workflow="$1"
    local attempt query_output
    for attempt in 1 2 3 4 5; do
        if query_output="$(osmo workflow query "${workflow}" --format-type json)"; then
            printf '%s\n' "${query_output}"
            return 0
        fi
        echo "OSMO workflow query attempt ${attempt} failed; retrying." >&2
        sleep 2
    done
    echo "Unable to query OSMO workflow ${workflow} after five attempts." >&2
    return 1
}

source_task=source-prep
if [[ "${run_mode}" == multinode ]]; then
    # OSMO exposes the authenticated task rsync module on a group's lead.
    # trainer-node-0 validates the upload and serves it to the remaining nodes.
    source_task=trainer-node-0
fi
wait_deadline="$(( SECONDS + source_upload_wait_seconds ))"
printf 'source_upload=waiting workflow=%s task=%s\n' "${workflow_id}" "${source_task}"
while true; do
    query_json="$(query_workflow "${workflow_id}")"
    status="$(task_status "${query_json}" "${source_task}")"
    case "${status}" in
        RUNNING)
            break
            ;;
        FAILED*|COMPLETED|CANCELED)
            echo "Cannot upload source: ${source_task} reached terminal status ${status}." >&2
            exit 13
            ;;
    esac
    if (( SECONDS >= wait_deadline )); then
        echo \
            "Timed out after ${source_upload_wait_seconds}s waiting for ${source_task} to accept the source upload." \
            >&2
        exit 13
    fi
    sleep 2
done

# OSMO 6.3's bundled Go rsync client can report success after transferring an
# empty file list, and a local-file:remote-file operand creates an empty remote
# directory. Use its authenticated daemon only to hold the task tunnel open,
# then transfer and verify regular files with the system rsync implementation.
system_rsync="${OSMO_SYSTEM_RSYNC_BIN:-/usr/bin/rsync}"
if [[ ! -x "${system_rsync}" ]]; then
    echo "System rsync is not executable: ${system_rsync}" >&2
    exit 13
fi

hold_dir="$(mktemp -d /tmp/yam-osmo-rsync-hold.XXXXXX)"
verify_dir="$(mktemp -d /tmp/yam-osmo-marker-verify.XXXXXX)"
hold_file="${hold_dir}/keepalive"
release_file="${hold_dir}/yam-source-release.sha256"
verify_sha256="${verify_dir}/yam-source-accepted.sha256"
touch "${hold_file}"
rsync_daemon_started=0
cleanup_source_transfer() {
    local run_status="$?"
    trap - EXIT
    if (( rsync_daemon_started != 0 )); then
        osmo workflow rsync stop "${workflow_id}" --task "${source_task}" >/dev/null 2>&1 || true
    fi
    rm -f "${verify_sha256}" "${hold_file}" "${release_file}"
    rmdir "${verify_dir}" "${hold_dir}" 2>/dev/null || true
    exit "${run_status}"
}
trap cleanup_source_transfer EXIT

osmo workflow rsync upload \
    "${workflow_id}" \
    "${source_task}" \
    "${hold_dir}:/osmo/run/workspace/.yam-rsync-tunnel" \
    --daemon \
    --timeout 30 \
    --poll-interval 20 \
    --debounce-delay 20 \
    --reconcile-interval 300 \
    --verbose
rsync_daemon_started=1

upload_deadline="$(( SECONDS + source_upload_retry_seconds ))"
rsync_port="${OSMO_RSYNC_PORT_OVERRIDE:-}"
while true; do
    if [[ -z "${rsync_port}" ]]; then
        rsync_daemon_pid="$(
            osmo workflow rsync status \
                | awk -v workflow="${workflow_id}" -v task="${source_task}" \
                    '$1 == workflow && $2 == task && $4 == "RUNNING" { print $3; exit }'
        )"
        if [[ "${rsync_daemon_pid}" =~ ^[0-9]+$ ]]; then
            rsync_port="$(
                ss -ltnpH \
                    | awk -v process="pid=${rsync_daemon_pid}," \
                        '$0 ~ process && $4 ~ /^127[.]0[.]0[.]1:/ { count = split($4, fields, ":"); print fields[count]; exit }'
            )"
        fi
    fi
    if [[ "${rsync_port}" =~ ^[0-9]+$ ]] \
        && timeout 3 "${system_rsync}" --list-only "rsync://127.0.0.1:${rsync_port}/" 2>/dev/null \
            | awk '{ print $1 }' \
            | grep -Fxq osmo; then
        break
    fi
    rsync_port="${OSMO_RSYNC_PORT_OVERRIDE:-}"
    query_json="$(query_workflow "${workflow_id}")"
    status="$(task_status "${query_json}" "${source_task}")"
    if [[ "${status}" != RUNNING ]]; then
        echo "Source tunnel stopped because ${source_task} is ${status}." >&2
        exit 13
    fi
    if (( SECONDS >= upload_deadline )); then
        echo "Timed out after ${source_upload_retry_seconds}s opening the authenticated rsync tunnel." >&2
        exit 13
    fi
    sleep 1
done

remote_source="rsync://127.0.0.1:${rsync_port}/osmo/source-sync/"
rsync_transfer_options=(--archive)
# gokr-rsync rejects protocol-level --timeout options. Bound the local client
# process instead, which avoids modifying the server-side argument list.
timeout 300 "${system_rsync}" \
    "${rsync_transfer_options[@]}" \
    --info=stats2 \
    "${sync_root}/source.tar.gz" \
    "${sync_root}/source.metadata" \
    "${sync_root}/git-status.txt" \
    "${remote_source}"

# Publish readiness only after the payload transfer completes. The task checks
# both payload digests and emits a root-level acknowledgement after acceptance.
timeout 60 "${system_rsync}" \
    "${rsync_transfer_options[@]}" \
    --info=stats2 \
    "${sync_root}/source.sha256" \
    "${remote_source}"
remote_acceptance="rsync://127.0.0.1:${rsync_port}/osmo/yam-source-accepted.sha256"
acknowledgement_deadline="$(( SECONDS + source_upload_retry_seconds ))"
while ! timeout 10 "${system_rsync}" \
    "${rsync_transfer_options[@]}" \
    "${remote_acceptance}" \
    "${verify_dir}/" \
    2>/dev/null; do
    query_json="$(query_workflow "${workflow_id}")"
    status="$(task_status "${query_json}" "${source_task}")"
    if [[ "${status}" != RUNNING ]]; then
        echo "Source validation stopped because ${source_task} is ${status}." >&2
        exit 13
    fi
    if (( SECONDS >= acknowledgement_deadline )); then
        echo "Timed out waiting for the task's checksum-validated source acknowledgement." >&2
        exit 13
    fi
    sleep 1
done
if [[ ! -f "${verify_sha256}" ]] || ! cmp -s "${sync_root}/source.sha256" "${verify_sha256}"; then
    echo "Task source acknowledgement is not a regular, byte-identical file." >&2
    exit 13
fi

cp "${sync_root}/source.sha256" "${release_file}"
timeout 60 "${system_rsync}" "${rsync_transfer_options[@]}" --info=stats2 "${release_file}" \
    "rsync://127.0.0.1:${rsync_port}/osmo/"

osmo workflow rsync stop "${workflow_id}" --task "${source_task}" >/dev/null 2>&1 || true
rsync_daemon_started=0
rm -f "${verify_sha256}" "${hold_file}" "${release_file}"
rmdir "${verify_dir}" "${hold_dir}"
trap - EXIT
printf 'source_upload=complete workflow=%s sha256=%s\n' "${workflow_id}" "${source_sha256}"
