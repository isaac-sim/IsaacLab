# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render tests for the foveated camera on the Newton Warp and Isaac RTX renderers.

The cases live in :mod:`foveated_render_cases`; OVRTX runs them without Kit in ``test_foveated_camera_render_ovrtx``.
"""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

import foveated_render_cases as cases  # noqa: E402
import pytest  # noqa: E402
from isaaclab_fovi.sensors import FoveatedNewtonCamera, FoveatedRTXCamera  # noqa: E402
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # noqa: E402
from isaaclab_newton.renderers import NewtonWarpRendererCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from isaaclab_physx.renderers import IsaacRtxRendererCfg  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.isaacsim_ci]

# Each backend pairs a physics engine with the renderer under test and the implementation it selects.
BACKENDS = {
    "newton_warp": (NewtonCfg(solver_cfg=MJWarpSolverCfg()), NewtonWarpRendererCfg, FoveatedNewtonCamera),
    "isaac_rtx": (PhysxCfg(), IsaacRtxRendererCfg, FoveatedRTXCamera),
}


@pytest.mark.parametrize("source", cases.CAMERA_SOURCES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_foveated_camera_follows_gaze(backend, source):
    cases.foveated_camera_follows_gaze(BACKENDS[backend], source)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gaze_change_reaches_the_next_frame(backend):
    cases.gaze_change_reaches_the_next_frame(BACKENDS[backend])


@pytest.mark.parametrize("backend", BACKENDS)
def test_gaze_stays_relative_to_a_moved_mount(backend):
    cases.gaze_stays_relative_to_a_moved_mount(BACKENDS[backend])


@pytest.mark.parametrize("backend", BACKENDS)
def test_foveated_camera_samples_densest_at_the_fixation(backend):
    cases.foveated_camera_samples_densest_at_the_fixation(BACKENDS[backend])
