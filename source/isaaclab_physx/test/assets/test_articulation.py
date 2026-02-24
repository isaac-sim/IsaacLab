# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

HEADLESS = True

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import ctypes

import pytest
import torch
import warp as wp
from isaaclab_physx.assets import Articulation

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import ActuatorBase, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs.mdp.terminations import joint_effort_out_of_limit
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.version import get_isaac_sim_version, has_kit

##
# Pre-defined configs
##
from isaaclab_assets import ANYMAL_C_CFG, FRANKA_PANDA_CFG, SHADOW_HAND_CFG  # isort:skip


def generate_articulation_cfg(
    articulation_type: str,
    stiffness: float | None = 10.0,
    damping: float | None = 2.0,
    velocity_limit: float | None = None,
    effort_limit: float | None = None,
    velocity_limit_sim: float | None = None,
    effort_limit_sim: float | None = None,
) -> ArticulationCfg:
    """Generate an articulation configuration.

    Args:
        articulation_type: Type of articulation to generate.
            It should be one of: "humanoid", "panda", "anymal", "shadow_hand", "single_joint_implicit",
            "single_joint_explicit".
        stiffness: Stiffness value for the articulation's actuators. Only currently used for "humanoid".
            Defaults to 10.0.
        damping: Damping value for the articulation's actuators. Only currently used for "humanoid".
            Defaults to 2.0.
        velocity_limit: Velocity limit for the actuators. Only currently used for "single_joint_implicit"
            and "single_joint_explicit".
        effort_limit: Effort limit for the actuators. Only currently used for "single_joint_implicit"
            and "single_joint_explicit".
        velocity_limit_sim: Velocity limit for the actuators (set into the simulation).
            Only currently used for "single_joint_implicit" and "single_joint_explicit".
        effort_limit_sim: Effort limit for the actuators (set into the simulation).
            Only currently used for "single_joint_implicit" and "single_joint_explicit".

    Returns:
        The articulation configuration for the requested articulation type.

    """
    if articulation_type == "humanoid":
        articulation_cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/Humanoid/humanoid_instanceable.usd"
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.34)),
            actuators={"body": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=stiffness, damping=damping)},
        )
    elif articulation_type == "panda":
        articulation_cfg = FRANKA_PANDA_CFG
    elif articulation_type == "anymal":
        articulation_cfg = ANYMAL_C_CFG
    elif articulation_type == "shadow_hand":
        articulation_cfg = SHADOW_HAND_CFG
    elif articulation_type == "single_joint_implicit":
        articulation_cfg = ArticulationCfg(
            # we set 80.0 default for max force because default in USD is 10e10 which makes testing annoying.
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
                joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_effort=80.0, max_velocity=5.0),
            ),
            actuators={
                "joint": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit_sim=effort_limit_sim,
                    velocity_limit_sim=velocity_limit_sim,
                    effort_limit=effort_limit,
                    velocity_limit=velocity_limit,
                    stiffness=2000.0,
                    damping=100.0,
                ),
            },
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos=({"RevoluteJoint": 1.5708}),
                rot=(0.7071081, 0, 0, 0.7071055),
            ),
        )
    elif articulation_type == "single_joint_explicit":
        # we set 80.0 default for max force because default in USD is 10e10 which makes testing annoying.
        articulation_cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
                joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_effort=80.0, max_velocity=5.0),
            ),
            actuators={
                "joint": IdealPDActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit_sim=effort_limit_sim,
                    velocity_limit_sim=velocity_limit_sim,
                    effort_limit=effort_limit,
                    velocity_limit=velocity_limit,
                    stiffness=0.0,
                    damping=10.0,
                ),
            },
        )
    elif articulation_type == "spatial_tendon_test_asset":
        # we set 80.0 default for max force because default in USD is 10e10 which makes testing annoying.
        articulation_cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/IsaacLab/Tests/spatial_tendons.usd",
            ),
            actuators={
                "joint": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=2000.0,
                    damping=100.0,
                ),
            },
        )
    else:
        raise ValueError(
            f"Invalid articulation type: {articulation_type}, valid options are 'humanoid', 'panda', 'anymal',"
            " 'shadow_hand', 'single_joint_implicit', 'single_joint_explicit' or 'spatial_tendon_test_asset'."
        )

    return articulation_cfg


def generate_articulation(
    articulation_cfg: ArticulationCfg, num_articulations: int, device: str
) -> tuple[Articulation, torch.tensor]:
    """Generate an articulation from a configuration.

    Handles the creation of the articulation, the environment prims and the articulation's environment
    translations

    Args:
        articulation_cfg: Articulation configuration.
        num_articulations: Number of articulations to generate.
        device: Device to use for the tensors.

    Returns:
        The articulation and environment translations.

    """
    # Generate translations of 2.5 m in x for each articulation
    translations = torch.zeros(num_articulations, 3, device=device)
    translations[:, 0] = torch.arange(num_articulations) * 2.5

    # Create Top-level Xforms, one for each articulation
    for i in range(num_articulations):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=translations[i][:3])
    articulation = Articulation(articulation_cfg.replace(prim_path="/World/Env_.*/Robot"))

    return articulation, translations


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


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.isaacsim_ci
def test_initialization_floating_base_non_root(sim, num_articulations, device, add_ground_plane):
    """Test initialization for a floating-base with articulation root on a rigid body.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is not fixed base
    3. All buffers have correct shapes
    4. The articulation can be simulated

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid", stiffness=0.0, damping=0.0)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # Check that boundedness of articulation is correct
    assert ctypes.c_long.from_address(id(articulation)).value == 1

    # Play sim
    sim.reset()

    # Check if articulation is initialized
    assert articulation.is_initialized
    # Check that is fixed base
    assert not articulation.is_fixed_base
    # Check buffers that exists and have correct shapes
    assert wp.to_torch(articulation.data.root_pos_w).shape == (num_articulations, 3)
    assert wp.to_torch(articulation.data.root_quat_w).shape == (num_articulations, 4)
    assert wp.to_torch(articulation.data.joint_pos).shape == (num_articulations, 21)

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.root_view.max_dofs == articulation.root_view.shared_metatype.dof_count
    # -- link related
    assert articulation.root_view.max_links == articulation.root_view.shared_metatype.link_count
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names
    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert actuator.is_implicit_model == is_implicit_model_cfg

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.isaacsim_ci
def test_initialization_floating_base(sim, num_articulations, device, add_ground_plane):
    """Test initialization for a floating-base with articulation root on provided prim path.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is not fixed base
    3. All buffers have correct shapes
    4. The articulation can be simulated

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal", stiffness=0.0, damping=0.0)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)

    # Check that boundedness of articulation is correct
    assert ctypes.c_long.from_address(id(articulation)).value == 1

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized
    # Check that floating base
    assert not articulation.is_fixed_base
    # Check buffers that exists and have correct shapes
    assert wp.to_torch(articulation.data.root_pos_w).shape == (num_articulations, 3)
    assert wp.to_torch(articulation.data.root_quat_w).shape == (num_articulations, 4)
    assert wp.to_torch(articulation.data.joint_pos).shape == (num_articulations, 12)
    assert wp.to_torch(articulation.data.body_mass).shape == (num_articulations, articulation.num_bodies)
    assert wp.to_torch(articulation.data.body_inertia).shape == (num_articulations, articulation.num_bodies, 9)

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.root_view.max_dofs == articulation.root_view.shared_metatype.dof_count
    # -- link related
    assert articulation.root_view.max_links == articulation.root_view.shared_metatype.link_count
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names
    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert actuator.is_implicit_model == is_implicit_model_cfg

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization_fixed_base(sim, num_articulations, device):
    """Test initialization for fixed base.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is fixed base
    3. All buffers have correct shapes
    4. The articulation maintains its default state

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="panda")
    articulation, translations = generate_articulation(articulation_cfg, num_articulations, device=device)

    # Check that boundedness of articulation is correct
    assert ctypes.c_long.from_address(id(articulation)).value == 1

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized
    # Check that fixed base
    assert articulation.is_fixed_base
    # Check buffers that exists and have correct shapes
    assert wp.to_torch(articulation.data.root_pos_w).shape == (num_articulations, 3)
    assert wp.to_torch(articulation.data.root_quat_w).shape == (num_articulations, 4)
    assert wp.to_torch(articulation.data.joint_pos).shape == (num_articulations, 9)
    assert wp.to_torch(articulation.data.body_mass).shape == (num_articulations, articulation.num_bodies)
    assert wp.to_torch(articulation.data.body_inertia).shape == (num_articulations, articulation.num_bodies, 9)

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.root_view.max_dofs == articulation.root_view.shared_metatype.dof_count
    # -- link related
    assert articulation.root_view.max_links == articulation.root_view.shared_metatype.link_count
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names
    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert actuator.is_implicit_model == is_implicit_model_cfg

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)

        # check that the root is at the correct state - its default state as it is fixed base
        default_root_pose = wp.to_torch(articulation.data.default_root_pose).clone()
        default_root_vel = wp.to_torch(articulation.data.default_root_vel).clone()
        default_root_pose[:, :3] = default_root_pose[:, :3] + translations

        torch.testing.assert_close(wp.to_torch(articulation.data.root_link_pose_w), default_root_pose)
        torch.testing.assert_close(wp.to_torch(articulation.data.root_com_vel_w), default_root_vel)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.isaacsim_ci
def test_initialization_fixed_base_single_joint(sim, num_articulations, device, add_ground_plane):
    """Test initialization for fixed base articulation with a single joint.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is fixed base
    3. All buffers have correct shapes
    4. The articulation maintains its default state

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="single_joint_implicit")
    articulation, translations = generate_articulation(articulation_cfg, num_articulations, device=device)

    # Check that boundedness of articulation is correct
    assert ctypes.c_long.from_address(id(articulation)).value == 1

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized
    # Check that fixed base
    assert articulation.is_fixed_base
    # Check buffers that exists and have correct shapes
    assert wp.to_torch(articulation.data.root_pos_w).shape == (num_articulations, 3)
    assert wp.to_torch(articulation.data.root_quat_w).shape == (num_articulations, 4)
    assert wp.to_torch(articulation.data.joint_pos).shape == (num_articulations, 1)
    assert wp.to_torch(articulation.data.body_mass).shape == (num_articulations, articulation.num_bodies)
    assert wp.to_torch(articulation.data.body_inertia).shape == (num_articulations, articulation.num_bodies, 9)

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.root_view.max_dofs == articulation.root_view.shared_metatype.dof_count
    # -- link related
    assert articulation.root_view.max_links == articulation.root_view.shared_metatype.link_count
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names
    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert actuator.is_implicit_model == is_implicit_model_cfg

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)

        # check that the root is at the correct state - its default state as it is fixed base
        default_root_pose = wp.to_torch(articulation.data.default_root_pose).clone()
        default_root_vel = wp.to_torch(articulation.data.default_root_vel).clone()
        default_root_pose[:, :3] = default_root_pose[:, :3] + translations

        torch.testing.assert_close(wp.to_torch(articulation.data.root_link_pose_w), default_root_pose)
        torch.testing.assert_close(wp.to_torch(articulation.data.root_com_vel_w), default_root_vel)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization_hand_with_tendons(sim, num_articulations, device):
    """Test initialization for fixed base articulated hand with tendons.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is fixed base
    3. All buffers have correct shapes
    4. The articulation can be simulated

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="shadow_hand")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)

    # Check that boundedness of articulation is correct
    assert ctypes.c_long.from_address(id(articulation)).value == 1

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized
    # Check that fixed base
    assert articulation.is_fixed_base
    # Check buffers that exists and have correct shapes
    assert wp.to_torch(articulation.data.root_pos_w).shape == (num_articulations, 3)
    assert wp.to_torch(articulation.data.root_quat_w).shape == (num_articulations, 4)
    assert wp.to_torch(articulation.data.joint_pos).shape == (num_articulations, 24)
    assert wp.to_torch(articulation.data.body_mass).shape == (num_articulations, articulation.num_bodies)
    assert wp.to_torch(articulation.data.body_inertia).shape == (num_articulations, articulation.num_bodies, 9)

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.root_view.max_dofs == articulation.root_view.shared_metatype.dof_count
    # -- link related
    assert articulation.root_view.max_links == articulation.root_view.shared_metatype.link_count
    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert actuator.is_implicit_model == is_implicit_model_cfg

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.isaacsim_ci
def test_initialization_floating_base_made_fixed_base(sim, num_articulations, device, add_ground_plane):
    """Test initialization for a floating-base articulation made fixed-base using schema properties.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is fixed base after modification
    3. All buffers have correct shapes
    4. The articulation maintains its default state

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal").copy()
    # Fix root link by making it kinematic
    articulation_cfg.spawn.articulation_props.fix_root_link = True
    articulation, translations = generate_articulation(articulation_cfg, num_articulations, device=device)

    # Check that boundedness of articulation is correct
    assert ctypes.c_long.from_address(id(articulation)).value == 1

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized
    # Check that is fixed base
    assert articulation.is_fixed_base
    # Check buffers that exists and have correct shapes
    assert wp.to_torch(articulation.data.root_pos_w).shape == (num_articulations, 3)
    assert wp.to_torch(articulation.data.root_quat_w).shape == (num_articulations, 4)
    assert wp.to_torch(articulation.data.joint_pos).shape == (num_articulations, 12)

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.root_view.max_dofs == articulation.root_view.shared_metatype.dof_count
    # -- link related
    assert articulation.root_view.max_links == articulation.root_view.shared_metatype.link_count
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)

        # check that the root is at the correct state - its default state as it is fixed base
        default_root_pose = wp.to_torch(articulation.data.default_root_pose).clone()
        default_root_vel = wp.to_torch(articulation.data.default_root_vel).clone()
        default_root_pose[:, :3] = default_root_pose[:, :3] + translations

        torch.testing.assert_close(wp.to_torch(articulation.data.root_link_pose_w), default_root_pose)
        torch.testing.assert_close(wp.to_torch(articulation.data.root_com_vel_w), default_root_vel)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.isaacsim_ci
def test_initialization_fixed_base_made_floating_base(sim, num_articulations, device, add_ground_plane):
    """Test initialization for fixed base made floating-base using schema properties.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is floating base after modification
    3. All buffers have correct shapes
    4. The articulation can be simulated

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="panda")
    # Unfix root link by making it non-kinematic
    articulation_cfg.spawn.articulation_props.fix_root_link = False
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # Check that boundedness of articulation is correct
    assert ctypes.c_long.from_address(id(articulation)).value == 1

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized
    # Check that is floating base
    assert not articulation.is_fixed_base
    # Check buffers that exists and have correct shapes
    assert wp.to_torch(articulation.data.root_pos_w).shape == (num_articulations, 3)
    assert wp.to_torch(articulation.data.root_quat_w).shape == (num_articulations, 4)
    assert wp.to_torch(articulation.data.joint_pos).shape == (num_articulations, 9)

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.root_view.max_dofs == articulation.root_view.shared_metatype.dof_count
    # -- link related
    assert articulation.root_view.max_links == articulation.root_view.shared_metatype.link_count
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.isaacsim_ci
def test_out_of_range_default_joint_pos(sim, num_articulations, device, add_ground_plane):
    """Test that the default joint position from configuration is out of range.

    This test verifies that:
    1. The articulation fails to initialize when joint positions are out of range
    2. The error is properly handled

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    # Create articulation
    articulation_cfg = generate_articulation_cfg(articulation_type="panda").copy()
    articulation_cfg.init_state.joint_pos = {
        "panda_joint1": 10.0,
        "panda_joint[2, 4]": -20.0,
    }

    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)

    # Check that boundedness of articulation is correct
    assert ctypes.c_long.from_address(id(articulation)).value == 1

    # Play sim
    with pytest.raises(ValueError):
        sim.reset()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_out_of_range_default_joint_vel(sim, device):
    """Test that the default joint velocity from configuration is out of range.

    This test verifies that:
    1. The articulation fails to initialize when joint velocities are out of range
    2. The error is properly handled
    """
    articulation_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Robot")
    articulation_cfg.init_state.joint_vel = {
        "panda_joint1": 100.0,
        "panda_joint[2, 4]": -60.0,
    }
    articulation = Articulation(articulation_cfg)

    # Check that boundedness of articulation is correct
    assert ctypes.c_long.from_address(id(articulation)).value == 1

    # Play sim
    with pytest.raises(ValueError):
        sim.reset()


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.isaacsim_ci
def test_joint_pos_limits(sim, num_articulations, device, add_ground_plane):
    """Test write_joint_limits_to_sim API and when default pos falls outside of the new limits.

    This test verifies that:
    1. Joint limits can be set correctly
    2. Default positions are preserved when setting new limits
    3. Joint limits can be set with indexing
    4. Invalid joint positions are properly handled

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    # Create articulation
    articulation_cfg = generate_articulation_cfg(articulation_type="panda")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized

    # Get current default joint pos
    default_joint_pos = wp.to_torch(articulation._data.default_joint_pos).clone()

    # Set new joint limits
    limits = torch.zeros(num_articulations, articulation.num_joints, 2, device=device)
    limits[..., 0] = (torch.rand(num_articulations, articulation.num_joints, device=device) + 5.0) * -1.0
    limits[..., 1] = torch.rand(num_articulations, articulation.num_joints, device=device) + 5.0
    articulation.write_joint_position_limit_to_sim_index(limits=limits)

    # Check new limits are in place
    torch.testing.assert_close(wp.to_torch(articulation._data.joint_pos_limits), limits)
    torch.testing.assert_close(wp.to_torch(articulation._data.default_joint_pos), default_joint_pos)

    # Set new joint limits with indexing
    env_ids = torch.arange(1, device=device, dtype=torch.int32)
    joint_ids = torch.arange(2, device=device, dtype=torch.int32)
    limits = torch.zeros(env_ids.shape[0], joint_ids.shape[0], 2, device=device)
    limits[..., 0] = (torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) + 5.0) * -1.0
    limits[..., 1] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) + 5.0
    articulation.write_joint_position_limit_to_sim_index(limits=limits, env_ids=env_ids, joint_ids=joint_ids)

    # Check new limits are in place
    torch.testing.assert_close(wp.to_torch(articulation._data.joint_pos_limits)[env_ids][:, joint_ids], limits)
    torch.testing.assert_close(wp.to_torch(articulation._data.default_joint_pos), default_joint_pos)

    # Set new joint limits that invalidate default joint pos
    limits = torch.zeros(num_articulations, articulation.num_joints, 2, device=device)
    limits[..., 0] = torch.rand(num_articulations, articulation.num_joints, device=device) * -0.1
    limits[..., 1] = torch.rand(num_articulations, articulation.num_joints, device=device) * 0.1
    articulation.write_joint_position_limit_to_sim_index(limits=limits)

    # Check if all values are within the bounds
    default_joint_pos_torch = wp.to_torch(articulation._data.default_joint_pos)
    within_bounds = (default_joint_pos_torch >= limits[..., 0]) & (default_joint_pos_torch <= limits[..., 1])
    assert torch.all(within_bounds)

    # Set new joint limits that invalidate default joint pos with indexing
    limits = torch.zeros(env_ids.shape[0], joint_ids.shape[0], 2, device=device)
    limits[..., 0] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) * -0.1
    limits[..., 1] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) * 0.1
    articulation.write_joint_position_limit_to_sim_index(limits=limits, env_ids=env_ids, joint_ids=joint_ids)

    # Check if all values are within the bounds
    default_joint_pos_torch = wp.to_torch(articulation._data.default_joint_pos)
    within_bounds = (default_joint_pos_torch[env_ids][:, joint_ids] >= limits[..., 0]) & (
        default_joint_pos_torch[env_ids][:, joint_ids] <= limits[..., 1]
    )
    assert torch.all(within_bounds)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
def test_joint_effort_limits(sim, num_articulations, device, add_ground_plane):
    """Validate joint effort limits via joint_effort_out_of_limit()."""
    # Create articulation
    articulation_cfg = generate_articulation_cfg(articulation_type="panda")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)

    # Minimal env wrapper exposing scene["robot"]
    class _Env:
        def __init__(self, art):
            self.scene = {"robot": art}

    env = _Env(articulation)
    robot_all = SceneEntityCfg(name="robot")

    sim.reset()
    assert articulation.is_initialized

    # Case A: no clipping → should NOT terminate
    wp.to_torch(articulation._data.computed_torque).zero_()
    wp.to_torch(articulation._data.applied_torque).zero_()
    out = joint_effort_out_of_limit(env, robot_all)  # [N]
    assert torch.all(~out)

    # Case B: simulate clipping → should terminate
    wp.to_torch(articulation._data.computed_torque).fill_(100.0)  # pretend controller commanded 100
    wp.to_torch(articulation._data.applied_torque).fill_(50.0)  # pretend actuator clipped to 50
    out = joint_effort_out_of_limit(env, robot_all)  # [N]
    assert torch.all(out)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_external_force_buffer(sim, num_articulations, device):
    """Test if external force buffer correctly updates in the force value is zero case.

    This test verifies that:
    1. External forces can be applied correctly
    2. Force buffers are updated properly
    3. Zero forces are handled correctly

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # play the simulator
    sim.reset()

    # find bodies to apply the force
    body_ids, _ = articulation.find_bodies("base")

    # reset root state
    articulation.write_root_pose_to_sim_index(root_pose=wp.to_torch(articulation.data.default_root_pose).clone())
    articulation.write_root_velocity_to_sim_index(root_velocity=wp.to_torch(articulation.data.default_root_vel).clone())

    # reset dof state
    joint_pos, joint_vel = (
        wp.to_torch(articulation.data.default_joint_pos),
        wp.to_torch(articulation.data.default_joint_vel),
    )
    articulation.write_joint_position_to_sim_index(position=joint_pos)
    articulation.write_joint_velocity_to_sim_index(velocity=joint_vel)

    # reset articulation
    articulation.reset()

    # perform simulation
    for step in range(5):
        # initiate force tensor
        external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)

        if step == 0 or step == 3:
            # set a non-zero force
            force = 1
        else:
            # set a zero force
            force = 0

        # set force value
        external_wrench_b[:, :, 0] = force
        external_wrench_b[:, :, 3] = force

        # apply force
        articulation.permanent_wrench_composer.set_forces_and_torques_index(
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            body_ids=body_ids,
        )

        # check if the articulation's force and torque buffers are correctly updated
        for i in range(num_articulations):
            assert wp.to_torch(articulation.permanent_wrench_composer.composed_force)[i, 0, 0].item() == force
            assert wp.to_torch(articulation.permanent_wrench_composer.composed_torque)[i, 0, 0].item() == force

        # Check if the instantaneous wrench is correctly added to the permanent wrench
        articulation.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            body_ids=body_ids,
        )

        # apply action to the articulation
        articulation.set_joint_position_target_index(target=wp.to_torch(articulation.data.default_joint_pos).clone())
        articulation.write_data_to_sim()

        # perform step
        sim.step()

        # update buffers
        articulation.update(sim.cfg.dt)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_external_force_on_single_body(sim, num_articulations, device):
    """Test application of external force on the base of the articulation.

    This test verifies that:
    1. External forces can be applied to specific bodies
    2. The forces affect the articulation's motion correctly
    3. The articulation responds to the forces as expected

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)
    # Play the simulator
    sim.reset()

    # Find bodies to apply the force
    body_ids, _ = articulation.find_bodies("base")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 1] = 1000.0

    # Now we are ready!
    for _ in range(5):
        # reset root state
        articulation.write_root_pose_to_sim_index(root_pose=wp.to_torch(articulation.data.default_root_pose).clone())
        articulation.write_root_velocity_to_sim_index(
            root_velocity=wp.to_torch(articulation.data.default_root_vel).clone()
        )
        # reset dof state
        joint_pos, joint_vel = (
            wp.to_torch(articulation.data.default_joint_pos),
            wp.to_torch(articulation.data.default_joint_vel),
        )
        articulation.write_joint_position_to_sim_index(position=joint_pos)
        articulation.write_joint_velocity_to_sim_index(velocity=joint_vel)
        # reset articulation
        articulation.reset()
        # apply force
        articulation.permanent_wrench_composer.set_forces_and_torques_index(
            forces=external_wrench_b[..., :3], torques=external_wrench_b[..., 3:], body_ids=body_ids
        )
        # perform simulation
        for _ in range(100):
            # apply action to the articulation
            articulation.set_joint_position_target_index(
                target=wp.to_torch(articulation.data.default_joint_pos).clone()
            )
            articulation.write_data_to_sim()
            # perform step
            sim.step()
            # update buffers
            articulation.update(sim.cfg.dt)
        # check condition that the articulations have fallen down
        for i in range(num_articulations):
            assert wp.to_torch(articulation.data.root_pos_w)[i, 2].item() < 0.2


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_external_force_on_single_body_at_position(sim, num_articulations, device):
    """Test application of external force on the base of the articulation at a given position.

    This test verifies that:
    1. External forces can be applied to specific bodies at a given position
    2. External forces can be applied to specific bodies in the global frame
    3. External forces are calculated and composed correctly
    4. The forces affect the articulation's motion correctly
    5. The articulation responds to the forces as expected

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)
    # Play the simulator
    sim.reset()

    # Find bodies to apply the force
    body_ids, _ = articulation.find_bodies("base")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 2] = 500.0
    external_wrench_positions_b = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    external_wrench_positions_b[..., 1] = 1.0

    desired_force = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    desired_force[..., 2] = 1000.0
    desired_torque = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    desired_torque[..., 0] = 1000.0

    # Now we are ready!
    for i in range(5):
        # reset root state
        root_pose = wp.to_torch(articulation.data.default_root_pose).clone()
        root_pose[0, 0] = 2.5  # space them apart by 2.5m

        articulation.write_root_pose_to_sim_index(root_pose=root_pose)
        articulation.write_root_velocity_to_sim_index(
            root_velocity=wp.to_torch(articulation.data.default_root_vel).clone()
        )
        # reset dof state
        joint_pos, joint_vel = (
            wp.to_torch(articulation.data.default_joint_pos),
            wp.to_torch(articulation.data.default_joint_vel),
        )
        articulation.write_joint_position_to_sim_index(position=joint_pos)
        articulation.write_joint_velocity_to_sim_index(velocity=joint_vel)
        # reset articulation
        articulation.reset()
        # apply force
        is_global = False

        if i % 2 == 0:
            body_com_pos_w = wp.to_torch(articulation.data.body_com_pos_w)[:, body_ids, :3]
            # is_global = True
            external_wrench_positions_b[..., 0] = 0.0
            external_wrench_positions_b[..., 1] = 1.0
            external_wrench_positions_b[..., 2] = 0.0
            external_wrench_positions_b += body_com_pos_w
        else:
            external_wrench_positions_b[..., 0] = 0.0
            external_wrench_positions_b[..., 1] = 1.0
            external_wrench_positions_b[..., 2] = 0.0

        articulation.permanent_wrench_composer.set_forces_and_torques_index(
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            positions=external_wrench_positions_b,
            body_ids=body_ids,
            is_global=is_global,
        )
        articulation.permanent_wrench_composer.add_forces_and_torques_index(
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            positions=external_wrench_positions_b,
            body_ids=body_ids,
            is_global=is_global,
        )
        # perform simulation
        for _ in range(100):
            # apply action to the articulation
            articulation.set_joint_position_target_index(
                target=wp.to_torch(articulation.data.default_joint_pos).clone()
            )
            articulation.write_data_to_sim()
            # perform step
            sim.step()
            # update buffers
            articulation.update(sim.cfg.dt)
        # check condition that the articulations have fallen down
        for i in range(num_articulations):
            assert wp.to_torch(articulation.data.root_pos_w)[i, 2].item() < 0.2


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_external_force_on_multiple_bodies(sim, num_articulations, device):
    """Test application of external force on the legs of the articulation.

    This test verifies that:
    1. External forces can be applied to multiple bodies
    2. The forces affect the articulation's motion correctly
    3. The articulation responds to the forces as expected

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # Play the simulator
    sim.reset()

    # Find bodies to apply the force
    body_ids, _ = articulation.find_bodies(".*_SHANK")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 1] = 100.0

    # Now we are ready!
    for _ in range(5):
        # reset root state
        articulation.write_root_pose_to_sim_index(root_pose=wp.to_torch(articulation.data.default_root_pose).clone())
        articulation.write_root_velocity_to_sim_index(
            root_velocity=wp.to_torch(articulation.data.default_root_vel).clone()
        )
        # reset dof state
        joint_pos, joint_vel = (
            wp.to_torch(articulation.data.default_joint_pos),
            wp.to_torch(articulation.data.default_joint_vel),
        )
        articulation.write_joint_position_to_sim_index(position=joint_pos)
        articulation.write_joint_velocity_to_sim_index(velocity=joint_vel)
        # reset articulation
        articulation.reset()
        # apply force
        articulation.permanent_wrench_composer.set_forces_and_torques_index(
            forces=external_wrench_b[..., :3], torques=external_wrench_b[..., 3:], body_ids=body_ids
        )
        # perform simulation
        for _ in range(100):
            # apply action to the articulation
            articulation.set_joint_position_target_index(
                target=wp.to_torch(articulation.data.default_joint_pos).clone()
            )
            articulation.write_data_to_sim()
            # perform step
            sim.step()
            # update buffers
            articulation.update(sim.cfg.dt)
        # check condition
        for i in range(num_articulations):
            # since there is a moment applied on the articulation, the articulation should rotate
            assert wp.to_torch(articulation.data.root_ang_vel_w)[i, 2].item() > 0.1


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_external_force_on_multiple_bodies_at_position(sim, num_articulations, device):
    """Test application of external force on the legs of the articulation at a given position.

    This test verifies that:
    1. External forces can be applied to multiple bodies at a given position
    2. External forces can be applied to multiple bodies in the global frame
    3. External forces are calculated and composed correctly
    4. The forces affect the articulation's motion correctly
    5. The articulation responds to the forces as expected

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # Play the simulator
    sim.reset()

    # Find bodies to apply the force
    body_ids, _ = articulation.find_bodies(".*_SHANK")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 2] = 500.0
    external_wrench_positions_b = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    external_wrench_positions_b[..., 1] = 1.0

    desired_force = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    desired_force[..., 2] = 1000.0
    desired_torque = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    desired_torque[..., 0] = 1000.0

    # Now we are ready!
    for i in range(5):
        # reset root state
        articulation.write_root_pose_to_sim_index(root_pose=wp.to_torch(articulation.data.default_root_pose).clone())
        articulation.write_root_velocity_to_sim_index(
            root_velocity=wp.to_torch(articulation.data.default_root_vel).clone()
        )
        # reset dof state
        joint_pos, joint_vel = (
            wp.to_torch(articulation.data.default_joint_pos),
            wp.to_torch(articulation.data.default_joint_vel),
        )
        articulation.write_joint_position_to_sim_index(position=joint_pos)
        articulation.write_joint_velocity_to_sim_index(velocity=joint_vel)
        # reset articulation
        articulation.reset()

        is_global = False
        if i % 2 == 0:
            body_com_pos_w = wp.to_torch(articulation.data.body_com_pos_w)[:, body_ids, :3]
            is_global = True
            external_wrench_positions_b[..., 0] = 0.0
            external_wrench_positions_b[..., 1] = 1.0
            external_wrench_positions_b[..., 2] = 0.0
            external_wrench_positions_b += body_com_pos_w
        else:
            external_wrench_positions_b[..., 0] = 0.0
            external_wrench_positions_b[..., 1] = 1.0
            external_wrench_positions_b[..., 2] = 0.0

        # apply force
        articulation.permanent_wrench_composer.set_forces_and_torques_index(
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            positions=external_wrench_positions_b,
            body_ids=body_ids,
            is_global=is_global,
        )
        articulation.permanent_wrench_composer.add_forces_and_torques_index(
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            positions=external_wrench_positions_b,
            body_ids=body_ids,
            is_global=is_global,
        )
        # perform simulation
        for _ in range(100):
            # apply action to the articulation
            articulation.set_joint_position_target_index(
                target=wp.to_torch(articulation.data.default_joint_pos).clone()
            )
            articulation.write_data_to_sim()
            # perform step
            sim.step()
            # update buffers
            articulation.update(sim.cfg.dt)
        # check condition
        for i in range(num_articulations):
            # since there is a moment applied on the articulation, the articulation should rotate
            assert torch.abs(wp.to_torch(articulation.data.root_ang_vel_w)[i, 2]).item() > 0.1


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_loading_gains_from_usd(sim, num_articulations, device):
    """Test that gains are loaded from USD file if actuator model has them as None.

    This test verifies that:
    1. Gains are loaded correctly from USD file
    2. Default gains are applied when not specified
    3. The gains match the expected values

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid", stiffness=None, damping=None)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # Play sim
    sim.reset()

    # Expected gains
    # -- Stiffness values
    expected_stiffness = {
        ".*_waist.*": 20.0,
        ".*_upper_arm.*": 10.0,
        "pelvis": 10.0,
        ".*_lower_arm": 2.0,
        ".*_thigh:0": 10.0,
        ".*_thigh:1": 20.0,
        ".*_thigh:2": 10.0,
        ".*_shin": 5.0,
        ".*_foot.*": 2.0,
    }
    indices_list, _, values_list = string_utils.resolve_matching_names_values(
        expected_stiffness, articulation.joint_names
    )
    expected_stiffness = torch.zeros(articulation.num_instances, articulation.num_joints, device=articulation.device)
    expected_stiffness[:, indices_list] = torch.tensor(values_list, device=articulation.device)
    # -- Damping values
    expected_damping = {
        ".*_waist.*": 5.0,
        ".*_upper_arm.*": 5.0,
        "pelvis": 5.0,
        ".*_lower_arm": 1.0,
        ".*_thigh:0": 5.0,
        ".*_thigh:1": 5.0,
        ".*_thigh:2": 5.0,
        ".*_shin": 0.1,
        ".*_foot.*": 1.0,
    }
    indices_list, _, values_list = string_utils.resolve_matching_names_values(
        expected_damping, articulation.joint_names
    )
    expected_damping = torch.zeros_like(expected_stiffness)
    expected_damping[:, indices_list] = torch.tensor(values_list, device=articulation.device)

    # Check that gains are loaded from USD file
    torch.testing.assert_close(articulation.actuators["body"].stiffness, expected_stiffness)
    torch.testing.assert_close(articulation.actuators["body"].damping, expected_damping)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.isaacsim_ci
def test_setting_gains_from_cfg(sim, num_articulations, device, add_ground_plane):
    """Test that gains are loaded from the configuration correctly.

    This test verifies that:
    1. Gains are loaded correctly from configuration
    2. The gains match the expected values
    3. The gains are applied correctly to the actuators

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid")
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=sim.device
    )

    # Play sim
    sim.reset()

    # Expected gains
    expected_stiffness = torch.full(
        (articulation.num_instances, articulation.num_joints), 10.0, device=articulation.device
    )
    expected_damping = torch.full_like(expected_stiffness, 2.0)

    # Check that gains are loaded from USD file
    torch.testing.assert_close(articulation.actuators["body"].stiffness, expected_stiffness)
    torch.testing.assert_close(articulation.actuators["body"].damping, expected_damping)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_setting_gains_from_cfg_dict(sim, num_articulations, device):
    """Test that gains are loaded from the configuration dictionary correctly.

    This test verifies that:
    1. Gains are loaded correctly from configuration dictionary
    2. The gains match the expected values
    3. The gains are applied correctly to the actuators

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid")
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=sim.device
    )
    # Play sim
    sim.reset()

    # Expected gains
    expected_stiffness = torch.full(
        (articulation.num_instances, articulation.num_joints), 10.0, device=articulation.device
    )
    expected_damping = torch.full_like(expected_stiffness, 2.0)

    # Check that gains are loaded from USD file
    torch.testing.assert_close(articulation.actuators["body"].stiffness, expected_stiffness)
    torch.testing.assert_close(articulation.actuators["body"].damping, expected_damping)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("vel_limit_sim", [1e5, None])
@pytest.mark.parametrize("vel_limit", [1e2, None])
@pytest.mark.parametrize("add_ground_plane", [False])
@pytest.mark.isaacsim_ci
def test_setting_velocity_limit_implicit(sim, num_articulations, device, vel_limit_sim, vel_limit, add_ground_plane):
    """Test setting of velocity limit for implicit actuators.

    This test verifies that:
    1. Velocity limits can be set correctly for implicit actuators
    2. The limits are applied correctly to the simulation
    3. The limits are handled correctly when both sim and non-sim limits are set

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
        vel_limit_sim: The velocity limit to set in simulation
        vel_limit: The velocity limit to set in actuator
    """
    # create simulation
    articulation_cfg = generate_articulation_cfg(
        articulation_type="single_joint_implicit",
        velocity_limit_sim=vel_limit_sim,
        velocity_limit=vel_limit,
    )
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg,
        num_articulations=num_articulations,
        device=device,
    )
    # Play sim
    if vel_limit_sim is not None and vel_limit is not None:
        with pytest.raises(ValueError):
            sim.reset()
        return
    sim.reset()

    # read the values set into the simulation
    physx_vel_limit = wp.to_torch(articulation.root_view.get_dof_max_velocities()).to(device)
    # check data buffer
    torch.testing.assert_close(wp.to_torch(articulation.data.joint_velocity_limits), physx_vel_limit)
    # check actuator has simulation velocity limit
    torch.testing.assert_close(articulation.actuators["joint"].velocity_limit_sim, physx_vel_limit)
    # check that both values match for velocity limit
    torch.testing.assert_close(
        articulation.actuators["joint"].velocity_limit_sim,
        articulation.actuators["joint"].velocity_limit,
    )

    if vel_limit_sim is None:
        # Case 2: both velocity limit and velocity limit sim are not set
        #  This is the case where the velocity limit keeps its USD default value
        # Case 3: velocity limit sim is not set but velocity limit is set
        #   For backwards compatibility, we do not set velocity limit to simulation
        #   Thus, both default to USD default value.
        limit = articulation_cfg.spawn.joint_drive_props.max_velocity
    else:
        # Case 4: only velocity limit sim is set
        #   In this case, the velocity limit is set to the USD value
        limit = vel_limit_sim

    # check max velocity is what we set
    expected_velocity_limit = torch.full_like(physx_vel_limit, limit)
    torch.testing.assert_close(physx_vel_limit, expected_velocity_limit)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("vel_limit_sim", [1e5, None])
@pytest.mark.parametrize("vel_limit", [1e2, None])
@pytest.mark.isaacsim_ci
def test_setting_velocity_limit_explicit(sim, num_articulations, device, vel_limit_sim, vel_limit):
    """Test setting of velocity limit for explicit actuators."""
    articulation_cfg = generate_articulation_cfg(
        articulation_type="single_joint_explicit",
        velocity_limit_sim=vel_limit_sim,
        velocity_limit=vel_limit,
    )
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg,
        num_articulations=num_articulations,
        device=device,
    )
    # Play sim
    sim.reset()

    # collect limit init values
    physx_vel_limit = wp.to_torch(articulation.root_view.get_dof_max_velocities()).to(device)
    actuator_vel_limit = articulation.actuators["joint"].velocity_limit
    actuator_vel_limit_sim = articulation.actuators["joint"].velocity_limit_sim

    # check data buffer for joint_velocity_limits_sim
    torch.testing.assert_close(wp.to_torch(articulation.data.joint_velocity_limits), physx_vel_limit)
    # check actuator velocity_limit_sim is set to physx
    torch.testing.assert_close(actuator_vel_limit_sim, physx_vel_limit)

    if vel_limit is not None:
        expected_actuator_vel_limit = torch.full(
            (articulation.num_instances, articulation.num_joints),
            vel_limit,
            device=articulation.device,
        )
        # check actuator is set
        torch.testing.assert_close(actuator_vel_limit, expected_actuator_vel_limit)
        # check physx is not velocity_limit
        assert not torch.allclose(actuator_vel_limit, physx_vel_limit)
    else:
        # check actuator velocity_limit is the same as the PhysX default
        torch.testing.assert_close(actuator_vel_limit, physx_vel_limit)

    # simulation velocity limit is set to USD value unless user overrides
    if vel_limit_sim is not None:
        limit = vel_limit_sim
    else:
        limit = articulation_cfg.spawn.joint_drive_props.max_velocity
    # check physx is set to expected value
    expected_vel_limit = torch.full_like(physx_vel_limit, limit)
    torch.testing.assert_close(physx_vel_limit, expected_vel_limit)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("effort_limit_sim", [1e5, None])
@pytest.mark.parametrize("effort_limit", [1e2, 80.0, None])
@pytest.mark.isaacsim_ci
def test_setting_effort_limit_implicit(sim, num_articulations, device, effort_limit_sim, effort_limit):
    """Test setting of effort limit for implicit actuators.

    This test verifies the effort limit resolution logic for actuator models implemented in :class:`ActuatorBase`:
    - Case 1: If USD value == actuator config value: values match correctly
    - Case 2: If USD value != actuator config value: actuator config value is used
    - Case 3: If actuator config value is None: USD value is used as default
    """
    articulation_cfg = generate_articulation_cfg(
        articulation_type="single_joint_implicit",
        effort_limit_sim=effort_limit_sim,
        effort_limit=effort_limit,
    )
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg,
        num_articulations=num_articulations,
        device=device,
    )
    # Play sim
    if effort_limit_sim is not None and effort_limit is not None:
        with pytest.raises(ValueError):
            sim.reset()
        return
    sim.reset()

    # obtain the physx effort limits
    physx_effort_limit = wp.to_torch(articulation.root_view.get_dof_max_forces()).to(device=device)

    # check that the two are equivalent
    torch.testing.assert_close(
        articulation.actuators["joint"].effort_limit_sim,
        articulation.actuators["joint"].effort_limit,
    )
    torch.testing.assert_close(articulation.actuators["joint"].effort_limit_sim, physx_effort_limit)

    # decide the limit based on what is set
    if effort_limit_sim is None and effort_limit is None:
        limit = articulation_cfg.spawn.joint_drive_props.max_effort
    elif effort_limit_sim is not None and effort_limit is None:
        limit = effort_limit_sim
    elif effort_limit_sim is None and effort_limit is not None:
        limit = effort_limit

    # check that the max force is what we set
    expected_effort_limit = torch.full_like(physx_effort_limit, limit)
    torch.testing.assert_close(physx_effort_limit, expected_effort_limit)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("effort_limit_sim", [1e5, None])
@pytest.mark.parametrize("effort_limit", [80.0, 1e2, None])
@pytest.mark.isaacsim_ci
def test_setting_effort_limit_explicit(sim, num_articulations, device, effort_limit_sim, effort_limit):
    """Test setting of effort limit for explicit actuators.

    This test verifies the effort limit resolution logic for actuator models implemented in :class:`ActuatorBase`:
    - Case 1: If USD value == actuator config value: values match correctly
    - Case 2: If USD value != actuator config value: actuator config value is used
    - Case 3: If actuator config value is None: USD value is used as default

    """

    articulation_cfg = generate_articulation_cfg(
        articulation_type="single_joint_explicit",
        effort_limit_sim=effort_limit_sim,
        effort_limit=effort_limit,
    )
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg,
        num_articulations=num_articulations,
        device=device,
    )
    # Play sim
    sim.reset()

    # usd default effort limit is set to 80
    usd_default_effort_limit = 80.0

    # collect limit init values
    physx_effort_limit = wp.to_torch(articulation.root_view.get_dof_max_forces()).to(device)
    actuator_effort_limit = articulation.actuators["joint"].effort_limit
    actuator_effort_limit_sim = articulation.actuators["joint"].effort_limit_sim

    # check actuator effort_limit_sim is set to physx
    torch.testing.assert_close(actuator_effort_limit_sim, physx_effort_limit)

    if effort_limit is not None:
        expected_actuator_effort_limit = torch.full_like(actuator_effort_limit, effort_limit)
        # check actuator is set
        torch.testing.assert_close(actuator_effort_limit, expected_actuator_effort_limit)

        # check physx effort limit does not match the one explicit actuator has
        assert not (torch.allclose(actuator_effort_limit, physx_effort_limit))
    else:
        # When effort_limit is None, actuator should use USD default values
        expected_actuator_effort_limit = torch.full_like(physx_effort_limit, usd_default_effort_limit)
        torch.testing.assert_close(actuator_effort_limit, expected_actuator_effort_limit)

    # when using explicit actuators, the limits are set to high unless user overrides
    if effort_limit_sim is not None:
        limit = effort_limit_sim
    else:
        limit = ActuatorBase._DEFAULT_MAX_EFFORT_SIM  # type: ignore
    # check physx internal value matches the expected sim value
    expected_effort_limit = torch.full_like(physx_effort_limit, limit)
    torch.testing.assert_close(actuator_effort_limit_sim, expected_effort_limit)
    torch.testing.assert_close(physx_effort_limit, expected_effort_limit)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_reset(sim, num_articulations, device):
    """Test that reset method works properly."""
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid")
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
    )

    # Play the simulator
    sim.reset()

    # Now we are ready!
    # reset articulation
    articulation.reset()

    # Reset should zero external forces and torques
    assert not articulation._instantaneous_wrench_composer.active
    assert not articulation._permanent_wrench_composer.active
    assert torch.count_nonzero(wp.to_torch(articulation._instantaneous_wrench_composer.composed_force)) == 0
    assert torch.count_nonzero(wp.to_torch(articulation._instantaneous_wrench_composer.composed_torque)) == 0
    assert torch.count_nonzero(wp.to_torch(articulation._permanent_wrench_composer.composed_force)) == 0
    assert torch.count_nonzero(wp.to_torch(articulation._permanent_wrench_composer.composed_torque)) == 0

    if num_articulations > 1:
        num_bodies = articulation.num_bodies
        articulation.permanent_wrench_composer.set_forces_and_torques_index(
            forces=torch.ones((num_articulations, num_bodies, 3), device=device),
            torques=torch.ones((num_articulations, num_bodies, 3), device=device),
        )
        articulation.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=torch.ones((num_articulations, num_bodies, 3), device=device),
            torques=torch.ones((num_articulations, num_bodies, 3), device=device),
        )
        articulation.reset(env_ids=torch.tensor([0], device=device))
        assert articulation._instantaneous_wrench_composer.active
        assert articulation._permanent_wrench_composer.active
        assert (
            torch.count_nonzero(wp.to_torch(articulation._instantaneous_wrench_composer.composed_force))
            == num_bodies * 3
        )
        assert (
            torch.count_nonzero(wp.to_torch(articulation._instantaneous_wrench_composer.composed_torque))
            == num_bodies * 3
        )
        assert (
            torch.count_nonzero(wp.to_torch(articulation._permanent_wrench_composer.composed_force)) == num_bodies * 3
        )
        assert (
            torch.count_nonzero(wp.to_torch(articulation._permanent_wrench_composer.composed_torque)) == num_bodies * 3
        )


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.isaacsim_ci
def test_apply_joint_command(sim, num_articulations, device, add_ground_plane):
    """Test applying of joint position target functions correctly for a robotic arm."""
    articulation_cfg = generate_articulation_cfg(articulation_type="panda")
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
    )

    # Play the simulator
    sim.reset()

    for _ in range(100):
        # perform step
        sim.step()
        # update buffers
        articulation.update(sim.cfg.dt)

    # reset dof state
    joint_pos = wp.to_torch(articulation.data.default_joint_pos).clone()
    joint_pos[:, 3] = 0.0

    # apply action to the articulation
    articulation.set_joint_position_target_index(target=joint_pos)
    articulation.write_data_to_sim()

    for _ in range(100):
        # perform step
        sim.step()
        # update buffers
        articulation.update(sim.cfg.dt)

    # Check that current joint position is not the same as default joint position, meaning
    # the articulation moved. We can't check that it reached its desired joint position as the gains
    # are not properly tuned
    assert not torch.allclose(wp.to_torch(articulation.data.joint_pos), joint_pos)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True, False])
@pytest.mark.isaacsim_ci
def test_body_root_state(sim, num_articulations, device, with_offset):
    """Test for reading the `body_state_w` property.

    This test verifies that:
    1. Body states can be read correctly
    2. States are correct with and without offsets
    3. States are consistent across different devices

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
        with_offset: Whether to test with offset
    """
    sim._app_control_on_stop_handle = None
    articulation_cfg = generate_articulation_cfg(articulation_type="single_joint_implicit")
    articulation, env_pos = generate_articulation(articulation_cfg, num_articulations, device)
    env_idx = torch.tensor([x for x in range(num_articulations)], device=device, dtype=torch.int32)
    # Check that boundedness of articulation is correct
    assert ctypes.c_long.from_address(id(articulation)).value == 1, "Boundedness of articulation is incorrect"
    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized, "Articulation is not initialized"
    # Check that fixed base
    assert articulation.is_fixed_base, "Articulation is not a fixed base"

    # change center of mass offset from link frame
    if with_offset:
        offset = [0.5, 0.0, 0.0]
    else:
        offset = [0.0, 0.0, 0.0]

    # create com offsets
    num_bodies = articulation.num_bodies
    com = wp.to_torch(articulation.root_view.get_coms())
    link_offset = [1.0, 0.0, 0.0]  # the offset from CenterPivot to Arm frames
    new_com = torch.tensor(offset, device=device).repeat(num_articulations, 1, 1)
    com[:, 1, :3] = new_com.squeeze(-2)
    articulation.root_view.set_coms(
        wp.from_torch(com.cpu(), dtype=wp.float32), wp.from_torch(env_idx.cpu(), dtype=wp.int32)
    )

    # check they are set
    torch.testing.assert_close(wp.to_torch(articulation.root_view.get_coms()), com.cpu())

    for i in range(50):
        # perform step
        sim.step()
        # update buffers
        articulation.update(sim.cfg.dt)

        # get state properties
        root_link_pose_w = wp.to_torch(articulation.data.root_link_pose_w)
        root_link_vel_w = wp.to_torch(articulation.data.root_link_vel_w)
        root_com_pose_w = wp.to_torch(articulation.data.root_com_pose_w)
        root_com_vel_w = wp.to_torch(articulation.data.root_com_vel_w)
        body_link_pose_w = wp.to_torch(articulation.data.body_link_pose_w)
        body_link_vel_w = wp.to_torch(articulation.data.body_link_vel_w)
        body_com_pose_w = wp.to_torch(articulation.data.body_com_pose_w)
        body_com_vel_w = wp.to_torch(articulation.data.body_com_vel_w)

        if with_offset:
            # get joint state
            joint_pos = wp.to_torch(articulation.data.joint_pos).unsqueeze(-1)
            joint_vel = wp.to_torch(articulation.data.joint_vel).unsqueeze(-1)

            # LINK state
            # angular velocity should be the same for both COM and link frames
            torch.testing.assert_close(root_com_vel_w[..., 3:], root_link_vel_w[..., 3:])
            torch.testing.assert_close(body_com_vel_w[..., 3:], body_link_vel_w[..., 3:])

            # lin_vel arm
            lin_vel_gt = torch.zeros(num_articulations, num_bodies, 3, device=device)
            vx = -(link_offset[0]) * joint_vel * torch.sin(joint_pos)
            vy = torch.zeros(num_articulations, 1, 1, device=device)
            vz = (link_offset[0]) * joint_vel * torch.cos(joint_pos)
            lin_vel_gt[:, 1, :] = torch.cat([vx, vy, vz], dim=-1).squeeze(-2)

            # linear velocity of root link should be zero
            torch.testing.assert_close(lin_vel_gt[:, 0, :], root_link_vel_w[..., :3], atol=1e-3, rtol=1e-1)
            # linear velocity of pendulum link should be
            torch.testing.assert_close(lin_vel_gt, body_link_vel_w[..., :3], atol=1e-3, rtol=1e-1)

            # ang_vel
            torch.testing.assert_close(root_com_vel_w[..., 3:], root_link_vel_w[..., 3:])
            torch.testing.assert_close(body_com_vel_w[..., 3:], body_link_vel_w[..., 3:])

            # COM state
            # position and orientation shouldn't match for the _state_com_w but everything else will
            pos_gt = torch.zeros(num_articulations, num_bodies, 3, device=device)
            px = (link_offset[0] + offset[0]) * torch.cos(joint_pos)
            py = torch.zeros(num_articulations, 1, 1, device=device)
            pz = (link_offset[0] + offset[0]) * torch.sin(joint_pos)
            pos_gt[:, 1, :] = torch.cat([px, py, pz], dim=-1).squeeze(-2)
            pos_gt += env_pos.unsqueeze(-2).repeat(1, num_bodies, 1)
            torch.testing.assert_close(pos_gt[:, 0, :], root_com_pose_w[..., :3], atol=1e-3, rtol=1e-1)
            torch.testing.assert_close(pos_gt, body_com_pose_w[..., :3], atol=1e-3, rtol=1e-1)

            # orientation
            com_quat_b = wp.to_torch(articulation.data.body_com_quat_b)
            com_quat_w = math_utils.quat_mul(body_link_pose_w[..., 3:], com_quat_b)
            torch.testing.assert_close(com_quat_w, body_com_pose_w[..., 3:])
            torch.testing.assert_close(com_quat_w[:, 0, :], root_com_pose_w[..., 3:])

            # angular velocity should be the same for both COM and link frames
            torch.testing.assert_close(root_com_vel_w[..., 3:], root_link_vel_w[..., 3:])
            torch.testing.assert_close(body_com_vel_w[..., 3:], body_link_vel_w[..., 3:])
        else:
            # single joint center of masses are at link frames so they will be the same
            torch.testing.assert_close(root_link_pose_w, root_com_pose_w)
            torch.testing.assert_close(root_com_vel_w, root_link_vel_w)
            torch.testing.assert_close(body_link_pose_w, body_com_pose_w)
            torch.testing.assert_close(body_com_vel_w, body_link_vel_w)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True, False])
@pytest.mark.parametrize("state_location", ["com", "link"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_write_root_state(sim, num_articulations, device, with_offset, state_location, gravity_enabled):
    """Test the setters for root_state using both the link frame and center of mass as reference frame.

    This test verifies that:
    1. Root states can be written correctly
    2. States are correct with and without offsets
    3. States can be written for both COM and link frames
    4. States are consistent across different devices

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
        with_offset: Whether to test with offset
        state_location: Whether to test COM or link frame
    """
    sim._app_control_on_stop_handle = None
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, env_pos = generate_articulation(articulation_cfg, num_articulations, device)
    env_idx = torch.tensor([x for x in range(num_articulations)], device=device, dtype=torch.int32)

    # Play sim
    sim.reset()

    # change center of mass offset from link frame
    if with_offset:
        offset = torch.tensor([1.0, 0.0, 0.0]).repeat(num_articulations, 1, 1)
    else:
        offset = torch.tensor([0.0, 0.0, 0.0]).repeat(num_articulations, 1, 1)

    # create com offsets
    com = wp.to_torch(articulation.root_view.get_coms())
    new_com = offset
    com[:, 0, :3] = new_com.squeeze(-2)
    articulation.root_view.set_coms(
        wp.from_torch(com.cpu(), dtype=wp.float32), wp.from_torch(env_idx.cpu(), dtype=wp.int32)
    )

    # check they are set
    torch.testing.assert_close(wp.to_torch(articulation.root_view.get_coms()), com)

    rand_state = torch.zeros(num_articulations, 13, device=device)
    rand_state[..., :7] = wp.to_torch(articulation.data.default_root_pose)
    rand_state[..., :3] += env_pos
    # make quaternion a unit vector
    rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)

    env_idx = env_idx.to(device)
    for i in range(10):
        # perform step
        sim.step()
        # update buffers
        articulation.update(sim.cfg.dt)

        if state_location == "com":
            if i % 2 == 0:
                articulation.write_root_com_pose_to_sim_index(root_pose=rand_state[..., :7])
                articulation.write_root_com_velocity_to_sim_index(root_velocity=rand_state[..., 7:])
            else:
                articulation.write_root_com_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=env_idx)
                articulation.write_root_com_velocity_to_sim_index(root_velocity=rand_state[..., 7:], env_ids=env_idx)
        elif state_location == "link":
            if i % 2 == 0:
                articulation.write_root_link_pose_to_sim_index(root_pose=rand_state[..., :7])
                articulation.write_root_link_velocity_to_sim_index(root_velocity=rand_state[..., 7:])
            else:
                articulation.write_root_link_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=env_idx)
                articulation.write_root_link_velocity_to_sim_index(root_velocity=rand_state[..., 7:], env_ids=env_idx)

        if state_location == "com":
            torch.testing.assert_close(rand_state[..., :7], wp.to_torch(articulation.data.root_com_pose_w))
            torch.testing.assert_close(rand_state[..., 7:], wp.to_torch(articulation.data.root_com_vel_w))
        elif state_location == "link":
            torch.testing.assert_close(rand_state[..., :7], wp.to_torch(articulation.data.root_link_pose_w))
            torch.testing.assert_close(rand_state[..., 7:], wp.to_torch(articulation.data.root_link_vel_w))


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_body_incoming_joint_wrench_b_single_joint(sim, num_articulations, device):
    """Test the data.body_incoming_joint_wrench_b buffer is populated correctly and statically correct for single joint.

    This test verifies that:
    1. The body incoming joint wrench buffer has correct shape
    2. The wrench values are statically correct for a single joint
    3. The wrench values match expected values from gravity and external forces

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="single_joint_implicit")
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
    )

    # Play the simulator
    sim.reset()
    # apply external force
    external_force_vector_b = torch.zeros((num_articulations, articulation.num_bodies, 3), device=device)
    external_force_vector_b[:, 1, 1] = 10.0  # 10 N in Y direction
    external_torque_vector_b = torch.zeros((num_articulations, articulation.num_bodies, 3), device=device)
    external_torque_vector_b[:, 1, 2] = 10.0  # 10 Nm in z direction

    # apply action to the articulation
    joint_pos = torch.ones_like(wp.to_torch(articulation.data.joint_pos)) * 1.5708 / 2.0
    articulation.write_joint_position_to_sim_index(
        position=torch.ones_like(wp.to_torch(articulation.data.joint_pos)),
    )
    articulation.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(wp.to_torch(articulation.data.joint_vel)),
    )
    articulation.set_joint_position_target_index(target=joint_pos)
    articulation.write_data_to_sim()
    for _ in range(50):
        articulation.permanent_wrench_composer.set_forces_and_torques_index(
            forces=external_force_vector_b, torques=external_torque_vector_b
        )
        articulation.write_data_to_sim()
        # perform step
        sim.step()
        # update buffers
        articulation.update(sim.cfg.dt)

        # check shape
        assert wp.to_torch(articulation.data.body_incoming_joint_wrench_b).shape == (
            num_articulations,
            articulation.num_bodies,
            6,
        )

    # calculate expected static
    mass = wp.to_torch(articulation.data.body_mass).to("cpu")
    pos_w = wp.to_torch(articulation.data.body_pos_w)
    quat_w = wp.to_torch(articulation.data.body_quat_w)

    mass_link2 = mass[:, 1].view(num_articulations, -1)
    gravity = torch.tensor(sim.cfg.gravity, device="cpu").repeat(num_articulations, 1).view((num_articulations, 3))

    # NOTE: the com and link pose for single joint are colocated
    weight_vector_w = mass_link2 * gravity
    # expected wrench from link mass and external wrench
    expected_wrench = torch.zeros((num_articulations, 6), device=device)
    expected_wrench[:, :3] = math_utils.quat_apply(
        math_utils.quat_conjugate(quat_w[:, 0, :]),
        weight_vector_w.to(device) + math_utils.quat_apply(quat_w[:, 1, :], external_force_vector_b[:, 1, :]),
    )
    expected_wrench[:, 3:] = math_utils.quat_apply(
        math_utils.quat_conjugate(quat_w[:, 0, :]),
        torch.cross(
            pos_w[:, 1, :].to(device) - pos_w[:, 0, :].to(device),
            weight_vector_w.to(device) + math_utils.quat_apply(quat_w[:, 1, :], external_force_vector_b[:, 1, :]),
            dim=-1,
        )
        + math_utils.quat_apply(quat_w[:, 1, :], external_torque_vector_b[:, 1, :]),
    )

    # check value of last joint wrench
    torch.testing.assert_close(
        expected_wrench,
        wp.to_torch(articulation.data.body_incoming_joint_wrench_b)[:, 1, :].squeeze(1),
        atol=1e-2,
        rtol=1e-3,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_setting_articulation_root_prim_path(sim, device):
    """Test that the articulation root prim path can be set explicitly."""
    sim._app_control_on_stop_handle = None
    # Create articulation
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid")
    articulation_cfg.articulation_root_prim_path = "/torso"
    articulation, _ = generate_articulation(articulation_cfg, 1, device)

    # Check that boundedness of articulation is correct
    assert ctypes.c_long.from_address(id(articulation)).value == 1

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation._is_initialized


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_setting_invalid_articulation_root_prim_path(sim, device):
    """Test that the articulation root prim path can be set explicitly."""
    sim._app_control_on_stop_handle = None
    # Create articulation
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid")
    articulation_cfg.articulation_root_prim_path = "/non_existing_prim_path"
    articulation, _ = generate_articulation(articulation_cfg, 1, device=device)

    # Check that boundedness of articulation is correct
    assert ctypes.c_long.from_address(id(articulation)).value == 1

    # Play sim
    with pytest.raises(RuntimeError):
        sim.reset()


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_write_joint_state_data_consistency(sim, num_articulations, device, gravity_enabled):
    """Test the setters for root_state using both the link frame and center of mass as reference frame.

    This test verifies that after write_joint_state_to_sim operations:
    1. state, com_state, link_state value consistency
    2. body_pose, link
    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    sim._app_control_on_stop_handle = None
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, env_pos = generate_articulation(articulation_cfg, num_articulations, device)
    env_idx = torch.tensor([x for x in range(num_articulations)])

    # Play sim
    sim.reset()

    limits = torch.zeros(num_articulations, articulation.num_joints, 2, device=device)
    limits[..., 0] = (torch.rand(num_articulations, articulation.num_joints, device=device) + 5.0) * -1.0
    limits[..., 1] = torch.rand(num_articulations, articulation.num_joints, device=device) + 5.0
    articulation.write_joint_position_limit_to_sim_index(limits=limits)

    from torch.distributions import Uniform

    joint_pos_limits = wp.to_torch(articulation.data.joint_pos_limits)
    joint_vel_limits = wp.to_torch(articulation.data.joint_vel_limits)
    pos_dist = Uniform(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    vel_dist = Uniform(-joint_vel_limits, joint_vel_limits)

    original_body_link_pose_w = wp.to_torch(articulation.data.body_link_pose_w).clone()
    original_body_com_vel_w = wp.to_torch(articulation.data.body_com_vel_w).clone()

    rand_joint_pos = pos_dist.sample()
    rand_joint_vel = vel_dist.sample()

    articulation.write_joint_position_to_sim_index(position=rand_joint_pos)
    articulation.write_joint_velocity_to_sim_index(velocity=rand_joint_vel)
    # make sure valued updated
    body_link_pose_w = wp.to_torch(articulation.data.body_link_pose_w)
    body_com_vel_w = wp.to_torch(articulation.data.body_com_vel_w)
    original_body_states = torch.cat([original_body_link_pose_w, original_body_com_vel_w], dim=-1)
    body_state_w = torch.cat([body_link_pose_w, body_com_vel_w], dim=-1)
    assert torch.count_nonzero(original_body_states[:, 1:] != body_state_w[:, 1:]) > (
        len(original_body_states[:, 1:]) / 2
    )
    # validate body - link consistency
    body_link_vel_w = wp.to_torch(articulation.data.body_link_vel_w)
    torch.testing.assert_close(body_link_pose_w, wp.to_torch(articulation.data.body_link_pose_w))
    # skip lin_vel because it differs from link frame, this should be fine because we are only checking
    # if velocity update is triggered, which can be determined by comparing angular velocity
    torch.testing.assert_close(body_com_vel_w[..., 3:], body_link_vel_w[..., 3:])

    # validate link - com conistency
    body_com_pos_b = wp.to_torch(articulation.data.body_com_pos_b)
    body_com_quat_b = wp.to_torch(articulation.data.body_com_quat_b)
    expected_com_pos, expected_com_quat = math_utils.combine_frame_transforms(
        body_link_pose_w[..., :3].view(-1, 3),
        body_link_pose_w[..., 3:].view(-1, 4),
        body_com_pos_b.view(-1, 3),
        body_com_quat_b.view(-1, 4),
    )
    body_com_pos_w = wp.to_torch(articulation.data.body_com_pos_w)
    body_com_quat_w = wp.to_torch(articulation.data.body_com_quat_w)
    torch.testing.assert_close(expected_com_pos.view(len(env_idx), -1, 3), body_com_pos_w)
    torch.testing.assert_close(expected_com_quat.view(len(env_idx), -1, 4), body_com_quat_w)

    # validate body - com consistency
    body_com_lin_vel_w = wp.to_torch(articulation.data.body_com_lin_vel_w)
    body_com_ang_vel_w = wp.to_torch(articulation.data.body_com_ang_vel_w)
    torch.testing.assert_close(body_com_vel_w[..., :3], body_com_lin_vel_w)
    torch.testing.assert_close(body_com_vel_w[..., 3:], body_com_ang_vel_w)

    # validate pos_w, quat_w, pos_b, quat_b is consistent with pose_w and pose_b
    expected_com_pose_w = torch.cat((body_com_pos_w, body_com_quat_w), dim=2)
    expected_com_pose_b = torch.cat((body_com_pos_b, body_com_quat_b), dim=2)
    body_pos_w = wp.to_torch(articulation.data.body_pos_w)
    body_quat_w = wp.to_torch(articulation.data.body_quat_w)
    expected_body_pose_w = torch.cat((body_pos_w, body_quat_w), dim=2)
    body_link_pos_w = wp.to_torch(articulation.data.body_link_pos_w)
    body_link_quat_w = wp.to_torch(articulation.data.body_link_quat_w)
    expected_body_link_pose_w = torch.cat((body_link_pos_w, body_link_quat_w), dim=2)
    body_com_pose_w = wp.to_torch(articulation.data.body_com_pose_w)
    body_com_pose_b = wp.to_torch(articulation.data.body_com_pose_b)
    body_pose_w = wp.to_torch(articulation.data.body_pose_w)
    body_link_pose_w_fresh = wp.to_torch(articulation.data.body_link_pose_w)
    torch.testing.assert_close(body_com_pose_w, expected_com_pose_w)
    torch.testing.assert_close(body_com_pose_b, expected_com_pose_b)
    torch.testing.assert_close(body_pose_w, expected_body_pose_w)
    torch.testing.assert_close(body_link_pose_w_fresh, expected_body_link_pose_w)

    # validate pose_w is consistent with individual properties
    body_vel_w = wp.to_torch(articulation.data.body_vel_w)
    body_com_vel_w_fresh = wp.to_torch(articulation.data.body_com_vel_w)
    torch.testing.assert_close(body_pose_w, body_link_pose_w)
    torch.testing.assert_close(body_vel_w, body_com_vel_w)
    torch.testing.assert_close(body_link_pose_w_fresh, body_link_pose_w)
    torch.testing.assert_close(body_com_pose_w, wp.to_torch(articulation.data.body_com_pose_w))
    torch.testing.assert_close(body_vel_w, body_com_vel_w_fresh)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_spatial_tendons(sim, num_articulations, device):
    """Test spatial tendons apis.
    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation has spatial tendons
    3. All buffers have correct shapes
    4. The articulation can be simulated
    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    # skip test if Isaac Sim version is less than 5.0
    if has_kit() and get_isaac_sim_version().major < 5:
        pytest.skip("Spatial tendons are not supported in Isaac Sim < 5.0. Please update to Isaac Sim 5.0 or later.")
        return
    articulation_cfg = generate_articulation_cfg(articulation_type="spatial_tendon_test_asset")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)

    # Check that boundedness of articulation is correct
    assert ctypes.c_long.from_address(id(articulation)).value == 1

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized
    # Check that fixed base
    assert articulation.is_fixed_base
    # Check buffers that exists and have correct shapes
    assert wp.to_torch(articulation.data.root_pos_w).shape == (num_articulations, 3)
    assert wp.to_torch(articulation.data.root_quat_w).shape == (num_articulations, 4)
    assert wp.to_torch(articulation.data.joint_pos).shape == (num_articulations, 3)
    assert wp.to_torch(articulation.data.body_mass).shape == (num_articulations, articulation.num_bodies)
    assert wp.to_torch(articulation.data.body_inertia).shape == (num_articulations, articulation.num_bodies, 9)
    assert articulation.num_spatial_tendons == 1

    articulation.set_spatial_tendon_stiffness_index(stiffness=torch.tensor([10.0], device=device))
    articulation.set_spatial_tendon_limit_stiffness_index(limit_stiffness=torch.tensor([10.0], device=device))
    articulation.set_spatial_tendon_damping_index(damping=torch.tensor([10.0], device=device))
    articulation.set_spatial_tendon_offset_index(offset=torch.tensor([10.0], device=device))

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)


@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_write_joint_frictions_to_sim(sim, num_articulations, device, add_ground_plane):
    """Test applying of joint position target functions correctly for a robotic arm."""
    articulation_cfg = generate_articulation_cfg(articulation_type="panda")
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
    )

    # Play the simulator
    sim.reset()

    for _ in range(100):
        # perform step
        sim.step()
        # update buffers
        articulation.update(sim.cfg.dt)

    # apply action to the articulation
    dynamic_friction = torch.rand(num_articulations, articulation.num_joints, device=device)
    viscous_friction = torch.rand(num_articulations, articulation.num_joints, device=device)
    friction = torch.rand(num_articulations, articulation.num_joints, device=device)

    # Guarantee that the dynamic friction is not greater than the static friction
    dynamic_friction = torch.min(dynamic_friction, friction)

    # The static friction must be set first to be sure the dynamic friction is not greater than static
    # when both are set.
    articulation.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=friction,
        joint_dynamic_friction_coeff=dynamic_friction,
        joint_viscous_friction_coeff=viscous_friction,
    )
    articulation.write_data_to_sim()

    for _ in range(100):
        # perform step
        sim.step()
        # update buffers
        articulation.update(sim.cfg.dt)

    friction_props_from_sim = wp.to_torch(articulation.root_view.get_dof_friction_properties())
    joint_friction_coeff_sim = friction_props_from_sim[:, :, 0]
    joint_dynamic_friction_coeff_sim = friction_props_from_sim[:, :, 1]
    joint_viscous_friction_coeff_sim = friction_props_from_sim[:, :, 2]
    assert torch.allclose(joint_dynamic_friction_coeff_sim, dynamic_friction.cpu())
    assert torch.allclose(joint_viscous_friction_coeff_sim, viscous_friction.cpu())
    assert torch.allclose(joint_friction_coeff_sim, friction.cpu())

    # For Isaac Sim >= 5.0: also test the combined API that can set dynamic and viscous via
    # write_joint_friction_coefficient_to_sim; reset the sim to isolate this path.
    if has_kit() and get_isaac_sim_version().major >= 5:
        # Reset simulator to ensure a clean state for the alternative API path
        sim.reset()

        # Warm up a few steps to populate buffers
        for _ in range(100):
            sim.step()
            articulation.update(sim.cfg.dt)

        # New random coefficients
        dynamic_friction_2 = torch.rand(num_articulations, articulation.num_joints, device=device)
        viscous_friction_2 = torch.rand(num_articulations, articulation.num_joints, device=device)
        friction_2 = torch.rand(num_articulations, articulation.num_joints, device=device)

        # Guarantee that the dynamic friction is not greater than the static friction
        dynamic_friction_2 = torch.min(dynamic_friction_2, friction_2)

        # Use the combined setter to write all three at once
        articulation.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=friction_2,
            joint_dynamic_friction_coeff=dynamic_friction_2,
            joint_viscous_friction_coeff=viscous_friction_2,
        )
        articulation.write_data_to_sim()

        # Step to let sim ingest new params and refresh data buffers
        for _ in range(100):
            sim.step()
            articulation.update(sim.cfg.dt)

        friction_props_from_sim_2 = wp.to_torch(articulation.root_view.get_dof_friction_properties())
        joint_friction_coeff_sim_2 = friction_props_from_sim_2[:, :, 0]
        friction_dynamic_coef_sim_2 = friction_props_from_sim_2[:, :, 1]
        friction_viscous_coeff_sim_2 = friction_props_from_sim_2[:, :, 2]

        # Validate values propagated
        assert torch.allclose(friction_viscous_coeff_sim_2, viscous_friction_2.cpu())
        assert torch.allclose(friction_dynamic_coef_sim_2, dynamic_friction_2.cpu())
        assert torch.allclose(joint_friction_coeff_sim_2, friction_2.cpu())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
