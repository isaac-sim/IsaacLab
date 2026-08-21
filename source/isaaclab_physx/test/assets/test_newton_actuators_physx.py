# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal real-PhysX integration proof for Lab and Newton actuator dispatch."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from pathlib import Path
from typing import TypedDict

import pytest
import torch
from isaaclab_physx.assets import Articulation
from isaaclab_physx.physics import PhysxCfg

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, build_simulation_context

pytestmark = pytest.mark.integration

_FIXTURE = Path(__file__).parent / "data" / "articulation_ordering_branching.usda"


class _ActuatorResult(TypedDict):
    """Observations from one real actuator path."""

    initial_position: torch.Tensor
    final_position: torch.Tensor
    applied_effort: torch.Tensor
    native_path: bool
    has_wrapper: bool
    has_nonidentity_ordering: bool


def _run_actuator_path(use_newton_actuators: bool) -> _ActuatorResult:
    """Run four steps through one actuator path on the local branching fixture."""
    with build_simulation_context(
        sim_cfg=SimulationCfg(
            device="cuda:0",
            dt=1.0 / 120.0,
            gravity=(0.0, 0.0, 0.0),
            physics=PhysxCfg(),
            use_newton_actuators=use_newton_actuators,
        )
    ) as sim:
        articulation = Articulation(
            ArticulationCfg(
                prim_path="/World/Robot",
                spawn=sim_utils.UsdFileCfg(usd_path=str(_FIXTURE)),
                actuators={
                    "joints": IdealPDActuatorCfg(
                        joint_names_expr=[".*"],
                        stiffness=20.0,
                        damping=2.0,
                        actuator_effort_limit=50.0,
                    )
                },
                joint_ordering="mjwarp",
            )
        )
        UsdPhysics.FixedJoint.Define(sim_utils.get_current_stage(), "/World/Robot/fixed_root").GetBody1Rel().SetTargets(
            ["/World/Robot/base"]
        )
        sim.reset()
        initial_position = articulation.data.joint_pos.torch.clone()
        articulation.actuators.target_command.set_position_index(value=initial_position + 0.05)
        articulation.actuators.target_command.set_velocity_index(value=torch.zeros_like(initial_position))
        applied = []
        for _ in range(4):
            articulation.write_data_to_sim()
            applied.append(articulation.actuators.applied_effort.torch.clone())
            sim.step()
            articulation.update(sim.cfg.dt)
        result = {
            "initial_position": initial_position,
            "final_position": articulation.data.joint_pos.torch.clone(),
            "applied_effort": torch.stack(applied),
            "native_path": articulation._has_newton_actuators,
            "has_wrapper": articulation._physx_actuator_wrapper is not None,
            "has_nonidentity_ordering": tuple(articulation.joint_names) != tuple(articulation.backend_joint_names),
        }
    return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_lab_and_newton_actuator_paths_dispatch_equivalent_initial_command_and_move() -> None:
    """Distinguish both dispatch paths and require the same first torque plus live motion."""
    lab = _run_actuator_path(use_newton_actuators=False)
    native = _run_actuator_path(use_newton_actuators=True)

    assert not lab["native_path"]
    assert not lab["has_wrapper"]
    assert native["native_path"]
    assert native["has_wrapper"]
    assert lab["has_nonidentity_ordering"]
    assert native["has_nonidentity_ordering"]
    assert torch.any(lab["applied_effort"] != 0.0)
    assert torch.any(native["applied_effort"] != 0.0)
    assert torch.any(lab["final_position"] != lab["initial_position"])
    assert torch.any(native["final_position"] != native["initial_position"])
    torch.testing.assert_close(native["applied_effort"][0], lab["applied_effort"][0], atol=1e-6, rtol=0)
    torch.testing.assert_close(lab["applied_effort"][0], torch.ones_like(lab["applied_effort"][0]))
