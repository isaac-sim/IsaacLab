# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none


"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils
import pytest
from isaacsim.core.version import get_version

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    RigidObject,
    RigidObjectCfg,
    SurfaceGripper,
    SurfaceGripperCfg,
)
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# from isaacsim.robot.surface_gripper import GripperView


def generate_surface_gripper_cfgs(
    kinematic_enabled: bool = False,
    max_grip_distance: float = 0.1,
    coaxial_force_limit: float = 100.0,
    shear_force_limit: float = 100.0,
    retry_interval: float = 0.1,
    reset_xform_op_properties: bool = False,
) -> tuple[SurfaceGripperCfg, ArticulationCfg]:
    """Generate a surface gripper cfg and an articulation cfg.

    Args:
        max_grip_distance: The maximum grip distance of the surface gripper.
        coaxial_force_limit: The coaxial force limit of the surface gripper.
        shear_force_limit: The shear force limit of the surface gripper.
        retry_interval: The retry interval of the surface gripper.
        reset_xform_op_properties: Whether to reset the xform op properties of the surface gripper.

    Returns:
        A tuple containing the surface gripper cfg and the articulation cfg.
    """
    articulation_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Tests/SurfaceGripper/test_gripper.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic_enabled),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                ".*": 0.0,
            },
        ),
        actuators={
            "dummy": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    surface_gripper_cfg = SurfaceGripperCfg(
        max_grip_distance=max_grip_distance,
        coaxial_force_limit=coaxial_force_limit,
        shear_force_limit=shear_force_limit,
        retry_interval=retry_interval,
    )

    return surface_gripper_cfg, articulation_cfg


def generate_surface_gripper(
    surface_gripper_cfg: SurfaceGripperCfg,
    articulation_cfg: ArticulationCfg,
    num_surface_grippers: int,
    device: str,
) -> tuple[SurfaceGripper, Articulation, torch.Tensor]:
    """Generate a surface gripper and an articulation.

    Args:
        surface_gripper_cfg: The surface gripper cfg.
        articulation_cfg: The articulation cfg.
        num_surface_grippers: The number of surface grippers to generate.
        device: The device to run the test on.

    Returns:
        A tuple containing the surface gripper, the articulation, and the translations of the surface grippers.
    """
    # Generate translations of 2.5 m in x for each articulation
    translations = torch.zeros(num_surface_grippers, 3, device=device)
    translations[:, 0] = torch.arange(num_surface_grippers) * 2.5

    # Create Top-level Xforms, one for each articulation
    for i in range(num_surface_grippers):
        prim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=translations[i][:3])
    articulation = Articulation(articulation_cfg.replace(prim_path="/World/Env_.*/Robot"))
    surface_gripper_cfg = surface_gripper_cfg.replace(prim_expr="/World/Env_.*/Robot/Gripper/SurfaceGripper")
    surface_gripper = SurfaceGripper(surface_gripper_cfg)

    return surface_gripper, articulation, translations


def generate_grippable_object(sim, num_grippable_objects: int):
    object_cfg = RigidObjectCfg(
        prim_path="/World/Env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    grippable_object = RigidObject(object_cfg)

    return grippable_object


@pytest.fixture
def sim(request):
    """Create simulation context with the specified device."""
    device = request.getfixturevalue("device")
    if "gravity_enabled" in request.fixturenames:
        gravity_enabled = request.getfixturevalue("gravity_enabled")
    else:
        gravity_enabled = True  # default to gravity enabled
    if "add_ground_plane" in request.fixturenames:
        add_ground_plane = request.getfixturevalue("add_ground_plane")
    else:
        add_ground_plane = False  # default to no ground plane
    with build_simulation_context(
        device=device, auto_add_lighting=True, gravity_enabled=gravity_enabled, add_ground_plane=add_ground_plane
    ) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.isaacsim_ci
def test_initialization(sim, num_articulations, device, add_ground_plane) -> None:
    """Test initialization for articulation with a surface gripper.

    This test verifies that:
    1. The surface gripper is initialized correctly.
    2. The command and state buffers have the correct shapes.
    3. The command and state are initialized to the correct values.

    Args:
        num_articulations: The number of articulations to initialize.
        device: The device to run the test on.
        add_ground_plane: Whether to add a ground plane to the simulation.
    """
    isaac_sim_version = get_version()
    if int(isaac_sim_version[2]) < 5:
        return
    surface_gripper_cfg, articulation_cfg = generate_surface_gripper_cfgs(kinematic_enabled=False)
    surface_gripper, articulation, _ = generate_surface_gripper(
        surface_gripper_cfg, articulation_cfg, num_articulations, device
    )

    sim.reset()

    assert articulation.is_initialized
    assert surface_gripper.is_initialized

    # Check that the command and state buffers have the correct shapes
    assert surface_gripper.command.shape == (num_articulations,)
    assert surface_gripper.state.shape == (num_articulations,)

    # Check that the command and state are initialized to the correct values
    assert surface_gripper.command == 0.0  # Idle command after a reset
    assert surface_gripper.state == -1.0  # Open state after a reset

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)
        surface_gripper.update(sim.cfg.dt)


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.isaacsim_ci
def test_raise_error_if_not_cpu(sim, device, add_ground_plane) -> None:
    """Test that the SurfaceGripper raises an error if the device is not CPU."""
    isaac_sim_version = get_version()
    if int(isaac_sim_version[2]) < 5:
        return
    num_articulations = 1
    surface_gripper_cfg, articulation_cfg = generate_surface_gripper_cfgs(kinematic_enabled=False)
    surface_gripper, articulation, translations = generate_surface_gripper(
        surface_gripper_cfg, articulation_cfg, num_articulations, device
    )

    with pytest.raises(Exception):
        sim.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
