# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free unit tests for backend identity normalization."""

from __future__ import annotations

import sys
from pathlib import Path

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from backend_identity import identity_from_presets  # noqa: E402


def test_identity_from_presets_treats_newton_as_newton() -> None:
    """The 6.0 prerelease task presets report ``newton`` instead of ``newton_mjwarp``."""
    identity = identity_from_presets("newton")

    assert identity is not None
    assert identity.backend_key == "newton"


def test_identity_from_presets_keeps_newton_mjwarp_compatibility() -> None:
    """The newer preset spelling remains supported."""
    identity = identity_from_presets("cube,newton_mjwarp")

    assert identity is not None
    assert identity.backend_key == "newton"
