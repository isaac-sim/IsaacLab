# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render tests for the foveated camera on the Newton (Warp and OptiX) and Isaac RTX renderers.

The cases live in :mod:`foveated_render_cases`; OVRTX runs them without Kit in ``test_foveated_camera_render_ovrtx``.
"""

import os

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

import foveated_render_cases as cases  # noqa: E402
import pytest  # noqa: E402
from isaaclab_fovi.sensors import FoveatedNewtonCamera, FoveatedRTXCamera  # noqa: E402
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # noqa: E402
from isaaclab_newton.renderers import NewtonOptixRendererCfg, NewtonWarpRendererCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from isaaclab_physx.renderers import IsaacRtxRendererCfg  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.isaacsim_ci]

# Each backend pairs a physics engine with the renderer under test and the implementation it selects.
BACKENDS = {
    "newton_warp": (NewtonCfg(solver_cfg=MJWarpSolverCfg()), NewtonWarpRendererCfg, FoveatedNewtonCamera),
    "newton_optix": (NewtonCfg(solver_cfg=MJWarpSolverCfg()), NewtonOptixRendererCfg, FoveatedNewtonCamera),
    "isaac_rtx": (PhysxCfg(), IsaacRtxRendererCfg, FoveatedRTXCamera),
}


def backend_cfg(name: str):
    """The backend, after checking that its renderer can run here.

    OptiX needs the ``optix`` extra and a driver that supports its ABI; without them its cases skip, unless
    ``ISAACLAB_TEST_REQUIRE_OPTIX`` is set, as in Isaac Lab's own OptiX rendering tests.
    """
    if name == "newton_optix":
        if os.environ.get("ISAACLAB_TEST_REQUIRE_OPTIX"):
            import optix  # noqa: F401
        else:
            pytest.importorskip("optix", exc_type=ImportError)
    return BACKENDS[name]


@pytest.mark.parametrize("backend", BACKENDS)
def test_foveated_camera_follows_gaze(backend):
    cases.foveated_camera_follows_gaze(backend_cfg(backend))


@pytest.mark.parametrize("backend", BACKENDS)
def test_gaze_stays_relative_to_moved_mounts(backend):
    cases.gaze_stays_relative_to_moved_mounts(backend_cfg(backend))


@pytest.mark.parametrize("backend", BACKENDS)
def test_foveated_camera_samples_densest_at_the_fixation(backend):
    cases.foveated_camera_samples_densest_at_the_fixation(backend_cfg(backend))


def test_hybrid_gaze_turns_the_head_onto_the_fixation():
    # Gaze that turns a rig needs a tracked mount, so a renderer that turns the camera's pose by the gaze.
    cases.hybrid_gaze_turns_the_head_onto_the_fixation(backend_cfg("newton_warp"))
