# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporary NVBug 7071342 probe: repeat the Valiant Isaac RTX color cases in one Kit process. Do not commit."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

import dataclasses  # noqa: E402
import os  # noqa: E402

import pytest  # noqa: E402
from rendering_cases import KIT_RENDERING_CASES  # noqa: E402
from rendering_runner import run_rendering_case  # noqa: E402

from isaaclab.renderers import RenderBufferKind  # noqa: E402

_ITERATIONS = int(os.environ.get("TEAL_PROBE_ITERATIONS", "30"))
_COLOR_CASES = [
    dataclasses.replace(case, marks=())
    for case in KIT_RENDERING_CASES
    if case.scene == "gaussian_splats_valiant_1env"
    and case.renderer == "isaac_rtx"
    and case.aovs in ((RenderBufferKind.RGB,), (RenderBufferKind.RGB_HDR,))
]


@pytest.mark.parametrize("case", _COLOR_CASES, ids=[case.id for case in _COLOR_CASES])
@pytest.mark.parametrize("iteration", range(_ITERATIONS))
def test_valiant_teal_probe(case, iteration, request):
    run_rendering_case(case, request, artifact_namespace=f"iter{iteration:03d}")
