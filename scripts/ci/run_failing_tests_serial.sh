#!/usr/bin/env bash
# Run CI-failing test modules one at a time; continue on failure; per-test log files.
#
# Usage (from repo root):
#   ./scripts/ci/run_failing_tests_serial.sh
#   ./scripts/ci/run_failing_tests_serial.sh -v --tb=short    # extra pytest args
#
# Detached run after SSH disconnect (pick one):
#   nohup ./scripts/ci/run_failing_tests_serial.sh > run_serial_master.log 2>&1 & disown
#   # or: screen -S ci / tmux new -s ci — then run the script inside the session.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/logs/ci_serial_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# Paths relative to repo root (match isaaclab CI layout under workspace/isaaclab).
TESTS=(
  "source/isaaclab/test/utils/test_configclass.py"
  "source/isaaclab_mimic/test/test_generate_dataset_gr1t2_pickplace.py"
  "source/isaaclab_physx/test/assets/test_deformable_object.py"
)

SUMMARY="${LOG_DIR}/SUMMARY.txt"
{
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "Started: $(date -Is)"
  echo "Extra pytest args: $*"
  echo "----------"
} | tee "${SUMMARY}"

failed=0
passed=0
for rel in "${TESTS[@]}"; do
  slug="$(basename "${rel}" .py)"
  log="${LOG_DIR}/${slug}.log"
  echo "" | tee -a "${SUMMARY}"
  echo "=== $(date -Is) :: pytest ${rel} ===" | tee -a "${SUMMARY}"
  # Continue on failure. Avoid `pytest | tee` for exit code — preceding pipelines can
  # leave PIPESTATUS stale in some bash builds; write to file then cat.
  set +e
  pytest "${rel}" "$@" >"${log}" 2>&1
  ec=$?
  set -e
  cat "${log}"
  if [[ "${ec}" -eq 0 ]]; then
    echo "RESULT: PASS (exit ${ec})" | tee -a "${SUMMARY}"
    passed=$((passed + 1))
  else
    echo "RESULT: FAIL (exit ${ec})" | tee -a "${SUMMARY}"
    failed=$((failed + 1))
  fi
done

{
  echo "----------"
  echo "Finished: $(date -Is)"
  echo "Passed modules: ${passed}  Failed modules: ${failed}"
  echo "Logs: ${LOG_DIR}"
} | tee -a "${SUMMARY}"

exit $(( failed > 0 ? 1 : 0 ))
