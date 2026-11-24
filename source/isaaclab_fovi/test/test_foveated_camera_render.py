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
from isaaclab_isaacsim.physics import PhysxCfg  # noqa: E402
from isaaclab_isaacsim.renderers import IsaacRtxRendererCfg  # noqa: E402
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # noqa: E402
from isaaclab_newton.renderers import NewtonOptixRendererCfg, NewtonWarpRendererCfg  # noqa: E402

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


@pytest.fixture(scope="module", params=list(BACKENDS))
def ground(request):
    """One ground scene per backend, which every case below renders in turn."""
    with cases.ground_scene(backend_cfg(request.param)) as scene:
        yield scene


def test_foveated_camera_follows_gaze(ground):
    cases.foveated_camera_follows_gaze(ground)


def test_gaze_stays_relative_to_moved_mounts(ground):
    cases.gaze_stays_relative_to_moved_mounts(ground)


def test_foveated_camera_samples_densest_at_the_fixation(ground):
    cases.foveated_camera_samples_densest_at_the_fixation(ground)


# Only Newton traces rays a camera can replace; test_camera_gaze checks that the RTX camera refuses.
@pytest.mark.parametrize("ground", ["newton_warp", "newton_optix"], indirect=True)
def test_cmf_a_switches_the_retina_at_runtime(ground):
    cases.cmf_a_switches_the_retina_at_runtime(ground)
