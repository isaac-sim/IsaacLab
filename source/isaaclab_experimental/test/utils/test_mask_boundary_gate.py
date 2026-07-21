# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static gate for the mask-first contract.

Host-side mask-to-ID compaction is a synchronization point and must only happen
at explicitly reviewed boundaries (recorders, legacy term/actuator APIs,
init-time tooling, diagnostics). Every such call site carries a same-line
``# mask-boundary: <reason>`` marker; this gate fails when a new unmarked site
appears in the scanned production packages.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCAN_ROOTS = (
    "source/isaaclab_experimental/isaaclab_experimental",
    "source/isaaclab_newton/isaaclab_newton",
    "source/isaaclab/isaaclab/scene",
    "source/isaaclab/isaaclab/terrains",
)
_COMPACTION_PATTERN = "nonzero("
_BOUNDARY_MARKER = "mask-boundary:"


def test_host_compaction_sites_carry_mask_boundary_markers():
    """Every production mask-to-ID compaction must be a marked, reviewed boundary."""
    violations = []
    for scan_root in _SCAN_ROOTS:
        for path in sorted((_REPO_ROOT / scan_root).rglob("*.py")):
            for line_number, line in enumerate(path.read_text().splitlines(), start=1):
                if _COMPACTION_PATTERN in line and _BOUNDARY_MARKER not in line:
                    violations.append(f"{path.relative_to(_REPO_ROOT)}:{line_number}: {line.strip()}")
    assert not violations, (
        "Host mask-to-ID compaction found outside marked boundaries. Make the call mask-native, or"
        " append '# mask-boundary: <reason>' after review:\n" + "\n".join(violations)
    )
