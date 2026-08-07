#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Record the runner's GPU interconnect topology and warn when it cannot exercise
# a multi-GPU smoke case.
#
# Invoked by .github/workflows/test-multi-gpu-pytest.yaml's "Multi-GPU topology
# report" step, immediately before the smoke tests run.
#
# Why this is a separate step: the smoke tests themselves pick their GPU pair per
# interconnect class, and skip when the host offers no qualifying pair. That is a
# correct skip, but a silent one -- skips among many read as a normal green run,
# so the step would look like it covered the defect when it did not. This makes
# the gap visible in the run summary.

set -euo pipefail

echo "::group::GPU topology on this runner"
TOPO="$(nvidia-smi topo -m 2>/dev/null || true)"
echo "${TOPO:-topology unavailable}"
echo "::endgroup::"

# Match the MATRIX ROWS only (lines starting GPU<n>). ``topo -m`` ends with a
# legend that spells out every class name, so grepping the whole output finds
# "SYS" and "PIX" on a host that has neither and the guard never fires.
TOPO_ROWS="$(grep -E '^GPU[0-9]+' <<<"$TOPO" || true)"

if [ -z "$TOPO_ROWS" ]; then
  echo "::warning::GPU topology unavailable -- multi-GPU smoke cases will skip; this run does not cover NVBUG#6565122"
  exit 0
fi

if ! grep -qw SYS <<<"$TOPO_ROWS"; then
  echo "::warning::No cross-socket (SYS) GPU pair on this runner -- the NVBUG#6565122 cases will skip; this run does not cover the cross-socket regression"
fi
if ! grep -qwE "PIX|NV[0-9]+" <<<"$TOPO_ROWS"; then
  echo "::warning::No same-switch (PIX/NVLink) GPU pair on this runner -- the strict camera regression guard will skip"
fi
