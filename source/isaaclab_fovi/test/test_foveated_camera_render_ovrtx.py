# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render tests for the foveated camera on the OVRTX renderer, with Newton physics.

Runs without Kit: ``ovrtx`` and Isaac Sim Kit ship conflicting RTX libraries that cannot load in one process. The
cases live in :mod:`foveated_render_cases`.
"""

import importlib.util

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.ov]

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx", "isaaclab_newton")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]
pytestmark.append(
    pytest.mark.skipif(bool(_MISSING_MODULES), reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}")
)

if not _MISSING_MODULES:
    import foveated_render_cases as cases
    from isaaclab_fovi.sensors import FoveatedRTXCamera
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_ov.renderers import OVRTXRendererCfg

    BACKEND = (NewtonCfg(solver_cfg=MJWarpSolverCfg()), OVRTXRendererCfg, FoveatedRTXCamera)


@pytest.fixture(scope="module")
def ground():
    """One ground scene, which every case below renders in turn."""
    with cases.ground_scene(BACKEND) as scene:
        yield scene


def test_foveated_camera_follows_gaze(ground):
    cases.foveated_camera_follows_gaze(ground)


def test_gaze_stays_relative_to_moved_mounts(ground):
    cases.gaze_stays_relative_to_moved_mounts(ground)


def test_foveated_camera_samples_densest_at_the_fixation(ground):
    cases.foveated_camera_samples_densest_at_the_fixation(ground)
