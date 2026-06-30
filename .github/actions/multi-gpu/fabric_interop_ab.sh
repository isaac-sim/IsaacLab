#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Diagnostic-only A/B driver for /physics/fabricUseGPUInterop. It invokes the
# production multi-GPU host launcher for every trial, preserving the actual CI
# container, shard, pytest, and work-queue setup. The negative arm (argument
# omitted) runs first so a regression is surfaced before the control arm.

set +e

readonly trials_per_arm="${TRIALS_PER_ARM:-5}"
readonly original_runner_temp="${RUNNER_TEMP:-/tmp}"
readonly original_run_id="${GITHUB_RUN_ID:-local}"
readonly original_run_attempt="${GITHUB_RUN_ATTEMPT:-1}"
readonly results_dir="${original_runner_temp}/fabric-interop-ab"
readonly launcher=".github/actions/multi-gpu/multi_gpu_host_launcher.sh"
readonly aggregator=".github/actions/multi-gpu/aggregate_test_summary.py"

if ! [[ "$trials_per_arm" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::TRIALS_PER_ARM must be a positive integer, got: $trials_per_arm"
  exit 2
fi

rm -rf "$results_dir"
mkdir -p "$results_dir"
printf 'arm\ttrial\tsetting\trc\telapsed_s\tdone\tunclaimed\torphans\tmarkers\tmismatches\ttimeouts\tstage_attached\n' \
  > "$results_dir/summary.tsv"

candidate_count=0
IFS=',' read -ra candidate_paths <<< "${PATHS:-}"
for candidate_path in "${candidate_paths[@]}"; do
  [ -n "$candidate_path" ] && candidate_count=$((candidate_count + 1))
done
echo "::notice::Fabric interop A/B: $candidate_count files per trial, $trials_per_arm trials per arm"

overall_rc=0
for mode in omitted false; do
  if [ "$mode" = "omitted" ]; then
    arm="negative"
    omit_fabric_gpu_interop=1
    expected_resolved=true
  else
    arm="control"
    omit_fabric_gpu_interop=0
    expected_resolved=false
  fi

  for ((trial = 1; trial <= trials_per_arm; trial++)); do
    trial_name="${arm}-${trial}"
    trial_dir="$results_dir/$trial_name"
    trial_temp="$trial_dir/temp"
    trial_env="$trial_dir/github-env"
    trial_log="$trial_dir/launcher.log"
    mkdir -p "$trial_temp" "$trial_dir/junit"
    : > "$trial_env"
    rm -f tests/test-reports-*.xml

    echo "::group::Fabric interop $trial_name (mode=$mode)"
    if [ "$mode" = "omitted" ]; then
      echo "Starting $trial_name without a /physics/fabricUseGPUInterop argument"
    else
      echo "Starting $trial_name with /physics/fabricUseGPUInterop=false"
    fi
    start_seconds=$SECONDS
    ISAACLAB_DIAGNOSTIC_OMIT_FABRIC_GPU_INTEROP="$omit_fabric_gpu_interop" \
      RUNNER_TEMP="$trial_temp" \
      GITHUB_ENV="$trial_env" \
      GITHUB_RUN_ID="${original_run_id}-${trial_name}" \
      GITHUB_RUN_ATTEMPT="$original_run_attempt" \
      bash "$launcher" 2>&1 | tee "$trial_log"
    launcher_rc=${PIPESTATUS[0]}
    elapsed_seconds=$((SECONDS - start_seconds))

    runtime_dir=$(sed -n 's/^MGPU_RUNTIME_DIR=//p' "$trial_env" | tail -n 1)
    aggregate_rc=0
    if [ -n "$runtime_dir" ] && [ -d "$runtime_dir" ]; then
      RUNTIME_DIR="$runtime_dir" python3 "$aggregator" 2>&1 | tee "$trial_dir/aggregate.txt"
      aggregate_rc=${PIPESTATUS[0]}
      done_count=$(find "$runtime_dir/queue/done" -type f 2>/dev/null | wc -l)
      unclaimed_count=$(find "$runtime_dir/queue/queue" -type f 2>/dev/null | wc -l)
      orphan_count=$(find "$runtime_dir/queue/inflight" -type f 2>/dev/null | wc -l)
    else
      echo "::error::$trial_name did not export a valid MGPU_RUNTIME_DIR"
      : > "$trial_dir/aggregate.txt"
      done_count=0
      unclaimed_count="$candidate_count"
      orphan_count=0
      aggregate_rc=2
    fi
    cp tests/test-reports-*.xml "$trial_dir/junit/" 2>/dev/null

    expected_marker="FABRIC_INTEROP_DIAGNOSTIC mode=${mode} resolved=${expected_resolved}"
    marker_count=$(grep -aFc "$expected_marker" "$trial_log" || true)
    all_marker_count=$(grep -aFc 'FABRIC_INTEROP_DIAGNOSTIC mode=' "$trial_log" || true)
    mismatch_count=$((all_marker_count - marker_count))
    timeout_count=$(grep -aEic 'Startup Hang|Timeout:|timed out|timeout after' "$trial_log" || true)
    stage_attached_count=$(grep -aFic 'already attached' "$trial_log" || true)

    trial_rc="$launcher_rc"
    if [ "$aggregate_rc" -ne 0 ] || [ "$done_count" -ne "$candidate_count" ] || \
       [ "$unclaimed_count" -ne 0 ] || [ "$orphan_count" -ne 0 ] || \
       [ "$marker_count" -eq 0 ] || [ "$mismatch_count" -ne 0 ]; then
      [ "$trial_rc" -eq 0 ] && trial_rc=2
    fi
    [ "$trial_rc" -eq 0 ] || overall_rc=1

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$arm" "$trial" "$mode" "$trial_rc" "$elapsed_seconds" "$done_count" \
      "$unclaimed_count" "$orphan_count" "$marker_count" "$mismatch_count" \
      "$timeout_count" "$stage_attached_count" >> "$results_dir/summary.tsv"
    echo "Result $trial_name: rc=$trial_rc elapsed=${elapsed_seconds}s done=$done_count/$candidate_count " \
      "unclaimed=$unclaimed_count orphans=$orphan_count setting_markers=$marker_count mismatches=$mismatch_count " \
      "timeouts=$timeout_count stage_already_attached=$stage_attached_count"
    echo "::endgroup::"
  done
done

{
  echo "## Fabric GPU interop diagnostic"
  echo ""
  echo "All candidate multi-GPU test files ran in every trial. Negative trials omitted"
  echo '\`/physics/fabricUseGPUInterop\`; controls explicitly set it to \`false\`.'
  echo ""
  echo '| Arm | Trial | Setting | RC | Elapsed (s) | Done | Unclaimed | Orphans | Setting markers | Mismatches | Timeouts | Stage attached |'
  echo '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
  tail -n +2 "$results_dir/summary.tsv" | while IFS=$'\t' read -r arm trial setting rc elapsed done unclaimed orphans markers mismatches timeouts attached; do
    echo "| $arm | $trial | \`$setting\` | $rc | $elapsed | $done/$candidate_count | $unclaimed | $orphans | $markers | $mismatches | $timeouts | $attached |"
  done
  echo ""
  echo "Raw logs, per-trial aggregates, and JUnit XML are in the uploaded \`fabric-interop-ab\` artifact."
} >> "${GITHUB_STEP_SUMMARY:-/dev/null}"

column -t -s $'\t' "$results_dir/summary.tsv" || cat "$results_dir/summary.tsv"
exit "$overall_rc"
