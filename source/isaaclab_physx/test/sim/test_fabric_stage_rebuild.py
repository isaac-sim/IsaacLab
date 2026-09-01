# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import pytest
import torch

import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom

from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.utils.stage import get_current_stage_id

from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip

pytestmark = pytest.mark.isaacsim_ci

TRANSFORM_TOLERANCE = 1e-4
IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _flatten(value) -> tuple[float, ...]:
    if hasattr(value, "__len__"):
        flat: list[float] = []
        for item in value:
            flat.extend(_flatten(item))
        return tuple(flat)
    return (float(value),)


def _prims_left_at_identity_in_fabric() -> list[str]:
    """Prims that USD places away from the origin but Fabric still reports at identity."""
    import usdrt

    usd_stage = omni.usd.get_context().get_stage()
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    usd_transforms: dict[str, tuple[float, ...]] = {}
    for prim in usd_stage.Traverse():
        if prim.IsA(UsdGeom.Xformable):
            matrix = xform_cache.GetLocalToWorldTransform(prim)
            usd_transforms[prim.GetPath().pathString] = tuple(v for row in matrix for v in row)

    fabric_stage = usdrt.Usd.Stage.Attach(get_current_stage_id())
    stale: list[str] = []
    for prim in fabric_stage.Traverse():
        usd_value = usd_transforms.get(str(prim.GetPath()))
        if usd_value is None:
            continue
        attribute = prim.GetAttribute("omni:fabric:worldMatrix")
        if not attribute or not attribute.IsValid():
            continue
        value = attribute.Get()
        if value is None:
            continue
        fabric_value = _flatten(value)
        if len(fabric_value) != 16:
            continue
        # Only prims USD places off the origin can be distinguished from identity.
        if max(abs(v) for v in usd_value[12:15]) <= TRANSFORM_TOLERANCE:
            continue
        if max(abs(a - b) for a, b in zip(fabric_value, IDENTITY)) <= TRANSFORM_TOLERANCE:
            stale.append(str(prim.GetPath()))
    return stale


def _build_step_and_collect(steps: int = 5) -> list[str]:
    """Build a one-articulation stage, step it, and return prims Fabric left at identity."""
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cuda:0", use_fabric=True))
    robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, 0.5)
    robot = Articulation(robot_cfg)

    sim.reset()
    for _ in range(steps):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

    stale = _prims_left_at_identity_in_fabric()

    SimulationContext.clear_instance()
    omni.timeline.get_timeline_interface().stop()
    omni.usd.get_context().new_stage()
    return stale


def test_fabric_world_matrices_populated_after_stage_rebuild():
    """Static child prims keep their Fabric world matrices when a stage is rebuilt in-process.

    Regression test for https://github.com/isaac-sim/IsaacLab/issues/7472: the render delegate
    reads ``omni:fabric:worldMatrix``, so children left at identity are drawn at the world origin.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    first_build = _build_step_and_collect()
    assert not first_build, f"first build already left prims at identity: {first_build}"

    second_build = _build_step_and_collect()
    assert not second_build, (
        f"{len(second_build)} prim(s) lost their Fabric world matrix after a stage rebuild: {second_build[:8]}"
    )
