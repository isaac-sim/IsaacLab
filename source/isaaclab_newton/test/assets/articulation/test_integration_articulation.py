# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

import ctypes
import torch

import pytest
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager, PhysicsEvent
from newton import ModelBuilder

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import ActuatorBase, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs.mdp.terminations import joint_effort_out_of_limit
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import build_simulation_context
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab_assets import ANYMAL_C_CFG, FRANKA_PANDA_CFG  # isort:skip

# from isaaclab_assets import SHADOW_HAND_CFG  # isort:skip

SOLVER_CFGs = {
    "anymal": SimulationCfg(
        dt=0.005,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=80,
                ls_parallel=True,
                ls_iterations=20,
                cone="elliptic",
                impratio=100,
            )
        ),
    ),
}


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
        pytest.skip("Shadow hand is not supported yet.")
        # articulation_cfg = SHADOW_HAND_CFG
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
                rot=(0.7071055, 0.7071081, 0, 0),
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

    def set_builder():
        stage = get_current_stage()
        builder = ModelBuilder()
        num_envs = num_articulations
        for i in range(num_envs):
            proto = ModelBuilder()
            proto.add_usd(stage, root_path=f"/World/Env_{i}", load_visual_shapes=True)
            builder.add_world(proto)

        NewtonManager.set_builder(builder)
        NewtonManager._num_envs = num_articulations

    NewtonManager.register_callback(lambda _: set_builder(), PhysicsEvent.MODEL_INIT)

    return articulation, translations


@pytest.fixture
def sim(request):
    """Create simulation context with the specified device."""
    sim_cfg = SimulationCfg(dt=0.01, create_stage_in_memory=True)
    device = request.getfixturevalue("device")
    sim_cfg.device = device

    if "gravity_enabled" in request.fixturenames:
        gravity_enabled = request.getfixturevalue("gravity_enabled")
        if "gravity_enabled" in request.fixturenames:
            if gravity_enabled:
                sim_cfg.gravity = (0.0, 0.0, -9.81)
            else:
                sim_cfg.gravity = (0.0, 0.0, 0.0)
        else:
            sim_cfg.gravity = (0.0, 0.0, -9.81)  # default to gravity enabled
    if "add_ground_plane" in request.fixturenames:
        add_ground_plane = request.getfixturevalue("add_ground_plane")
    else:
        add_ground_plane = False  # default to no ground plane
    if "newton_cfg" in request.fixturenames:
        newton_cfg = request.getfixturevalue("newton_cfg")
        sim_cfg.newton_cfg = newton_cfg

    with build_simulation_context(auto_add_lighting=True, add_ground_plane=add_ground_plane, sim_cfg=sim_cfg) as sim:
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
    # -- joint related, -6 for the 6 DOFs of the root body
    assert articulation.num_joints == (NewtonManager.get_model().joint_dof_count) / num_articulations - 6
    # -- link related
    assert articulation.num_bodies == (NewtonManager.get_model().body_count) / num_articulations
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.body_names]
    assert prim_path_body_names == articulation.body_names
    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert actuator.is_implicit_model == is_implicit_model_cfg

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step(render=False)
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
    assert wp.to_torch(articulation.data.body_inertia).shape == (num_articulations, articulation.num_bodies, 3, 3)

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.num_joints == NewtonManager.get_model().joint_dof_count / num_articulations - 6
    # -- link related
    assert articulation.num_bodies == NewtonManager.get_model().body_count / num_articulations
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = articulation.root_view.body_names
    assert prim_path_body_names == articulation.body_names
    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert actuator.is_implicit_model == is_implicit_model_cfg

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step(render=False)
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
    assert wp.to_torch(articulation.data.body_inertia).shape == (num_articulations, articulation.num_bodies, 3, 3)

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.num_joints == NewtonManager.get_model().joint_dof_count / num_articulations
    # -- link related
    assert articulation.num_bodies == NewtonManager.get_model().body_count / num_articulations
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = articulation.root_view.body_names
    assert prim_path_body_names == articulation.body_names
    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert actuator.is_implicit_model == is_implicit_model_cfg

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step(render=False)
        # update articulation
        articulation.update(sim.cfg.dt)

        # check that the root is at the correct state - its default state as it is fixed base
        default_root_state = wp.to_torch(articulation.data.default_root_state).clone()
        default_root_state[:, :3] = default_root_state[:, :3] + translations

        torch.testing.assert_close(wp.to_torch(articulation.data.root_state_w), default_root_state)


# FIXME: https://github.com/newton-physics/newton/issues/1377
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
    assert wp.to_torch(articulation.data.body_inertia).shape == (num_articulations, articulation.num_bodies, 3, 3)

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.num_joints == NewtonManager.get_model().joint_dof_count / num_articulations
    # -- link related
    assert articulation.num_bodies == NewtonManager.get_model().body_count / num_articulations
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = articulation.root_view.body_names
    assert prim_path_body_names == articulation.body_names
    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert actuator.is_implicit_model == is_implicit_model_cfg

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step(render=False)
        # update articulation
        articulation.update(sim.cfg.dt)

        # check that the root is at the correct state - its default state as it is fixed base
        default_root_state = wp.to_torch(articulation.data.default_root_state).clone()
        default_root_state[:, :3] = default_root_state[:, :3] + translations

        torch.testing.assert_close(wp.to_torch(articulation.data.root_state_w), default_root_state)


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
    assert wp.to_torch(articulation.data.body_inertia).shape == (num_articulations, articulation.num_bodies, 3, 3)

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.num_joints == NewtonManager.get_model().joint_dof_count / num_articulations
    # -- link related
    assert articulation.num_bodies == NewtonManager.get_model().body_count / num_articulations
    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert actuator.is_implicit_model == is_implicit_model_cfg

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step(render=False)
        # update articulation
        articulation.update(sim.cfg.dt)


# FIXME: Weird error on that one. Would need more time to look into it.
@pytest.mark.skip("Weird error on that one. Would need more time to look into it")
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
    assert articulation.num_joints == NewtonManager.get_model().joint_dof_count / num_articulations
    # -- link related
    assert articulation.num_bodies == NewtonManager.get_model().body_count / num_articulations
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = articulation.root_view.body_names
    assert prim_path_body_names == articulation.body_names

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step(render=False)
        # update articulation
        articulation.update(sim.cfg.dt)

        # check that the root is at the correct state - its default state as it is fixed base
        default_root_state = wp.to_torch(articulation.data.default_root_state).clone()
        default_root_state[:, :3] = default_root_state[:, :3] + translations

        torch.testing.assert_close(wp.to_torch(articulation.data.root_state_w), default_root_state)


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
    assert articulation.num_joints == NewtonManager.get_model().joint_dof_count / num_articulations - 6
    # -- link related
    assert articulation.num_bodies == NewtonManager.get_model().body_count / num_articulations
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = articulation.root_view.body_names
    assert prim_path_body_names == articulation.body_names

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step(render=False)
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


# FIXME: https://github.com/newton-physics/newton/issues/1380
@pytest.mark.skip("https://github.com/newton-physics/newton/issues/1380")
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
    default_joint_pos = wp.to_torch(articulation.data.default_joint_pos).clone()

    # Set new joint limits
    limits = torch.zeros(num_articulations, articulation.num_joints, 2, device=device)
    limits[..., 0] = (torch.rand(num_articulations, articulation.num_joints, device=device) + 5.0) * -1.0
    limits[..., 1] = torch.rand(num_articulations, articulation.num_joints, device=device) + 5.0
    articulation.write_joint_position_limit_to_sim(limits[..., 0], limits[..., 1])

    # Check new limits are in place
    torch.testing.assert_close(wp.to_torch(articulation.data.joint_pos_limits), limits)
    torch.testing.assert_close(wp.to_torch(articulation.data.default_joint_pos), default_joint_pos)

    # Set new joint limits with indexing
    env_ids = torch.arange(1, device=device)
    joint_ids = torch.arange(2, device=device)
    limits = torch.zeros(env_ids.shape[0], joint_ids.shape[0], 2, device=device)
    limits[..., 0] = (torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) + 5.0) * -1.0
    limits[..., 1] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) + 5.0
    articulation.write_joint_position_limit_to_sim(limits[..., 0], limits[..., 1], env_ids=env_ids, joint_ids=joint_ids)

    # Check new limits are in place
    torch.testing.assert_close(wp.to_torch(articulation.data.joint_pos_limits)[env_ids][:, joint_ids], limits)
    torch.testing.assert_close(wp.to_torch(articulation.data.default_joint_pos), default_joint_pos)

    # Set new joint limits that invalidate default joint pos
    limits = torch.zeros(num_articulations, articulation.num_joints, 2, device=device)
    limits[..., 0] = torch.rand(num_articulations, articulation.num_joints, device=device) * -0.1
    limits[..., 1] = torch.rand(num_articulations, articulation.num_joints, device=device) * 0.1
    articulation.write_joint_position_limit_to_sim(limits[..., 0], limits[..., 1])

    # Check if all values are within the bounds
    within_bounds = (wp.to_torch(articulation.data.default_joint_pos) >= limits[..., 0]) & (
        wp.to_torch(articulation.data.default_joint_pos) <= limits[..., 1]
    )
    assert torch.all(within_bounds)

    # Set new joint limits that invalidate default joint pos with indexing
    limits = torch.zeros(env_ids.shape[0], joint_ids.shape[0], 2, device=device)
    limits[..., 0] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) * -0.1
    limits[..., 1] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) * 0.1
    articulation.write_joint_position_limit_to_sim(limits[..., 0], limits[..., 1], env_ids=env_ids, joint_ids=joint_ids)

    # Check if all values are within the bounds
    within_bounds = (wp.to_torch(articulation.data.default_joint_pos)[env_ids][:, joint_ids] >= limits[..., 0]) & (
        wp.to_torch(articulation.data.default_joint_pos)[env_ids][:, joint_ids] <= limits[..., 1]
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
    wp.to_torch(articulation.data.computed_torque).zero_()
    wp.to_torch(articulation.data.applied_torque).zero_()
    out = joint_effort_out_of_limit(env, robot_all)  # [N]
    assert torch.all(~out)

    # Case B: simulate clipping → should terminate
    wp.to_torch(articulation.data.computed_torque).fill_(100.0)  # pretend controller commanded 100
    wp.to_torch(articulation.data.applied_torque).fill_(50.0)  # pretend actuator clipped to 50
    out = joint_effort_out_of_limit(env, robot_all)  # [N]
    assert torch.all(out)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("newton_cfg", [NewtonCfg(solver_cfg=SOLVER_CFGs["anymal"])])
@pytest.mark.isaacsim_ci
def test_external_force_buffer(sim, num_articulations, device, newton_cfg):
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
    _, _, body_ids = articulation.find_bodies("base")

    # reset root state
    root_state = wp.to_torch(articulation.data.default_root_state).clone()
    articulation.write_root_state_to_sim(root_state)

    # reset dof state
    joint_pos, joint_vel = (
        wp.to_torch(articulation.data.default_joint_pos),
        wp.to_torch(articulation.data.default_joint_vel),
    )
    articulation.write_joint_state_to_sim(joint_pos, joint_vel)

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
        # TODO: Replace with wrench composer once the deprecation is complete
        articulation.set_external_force_and_torque(
            external_wrench_b[..., :3],
            external_wrench_b[..., 3:],
            body_ids=body_ids,
        )

        # check if the articulation's force and torque buffers are correctly updated
        for i in range(num_articulations):
            assert wp.to_torch(articulation.permanent_wrench_composer.composed_force)[i, 0, 0].item() == force
            assert wp.to_torch(articulation.permanent_wrench_composer.composed_torque)[i, 0, 0].item() == force

        # Check if the instantaneous wrench is correctly added to the permanent wrench
        articulation.instantaneous_wrench_composer.add_forces_and_torques(
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            body_ids=body_ids,
        )

        # apply action to the articulation
        articulation.set_joint_position_target(wp.to_torch(articulation.data.default_joint_pos).clone())
        articulation.write_data_to_sim()

        # perform step
        sim.step(render=False)

        # update buffers
        articulation.update(sim.cfg.dt)


# FIXME: CPU is failing here. It looks like it's related to the value override we've seen before in the RigidObject
# tests.
# FIXME: Do we want to error out when the shapes provided by the user are incorrect? We would need an extra check
@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("newton_cfg", [NewtonCfg(solver_cfg=SOLVER_CFGs["anymal"])])
@pytest.mark.isaacsim_ci
def test_external_force_on_single_body(sim, num_articulations, device, newton_cfg):
    """Test application of external force on the base of the articulation.

    This test verifies that:
    1. External forces can be applied to specific bodies
    2. The forces affect the articulation's motion correctly
    3. The articulation responds to the forces as expected

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    if device == "cpu":
        pytest.skip("CPU is failing here. Needs further investigation.")
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)
    # Play the simulator
    sim.reset()

    # Find bodies to apply the force
    _, _, body_ids = articulation.find_bodies("base")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 1] = 200.0

    # Now we are ready!
    for _ in range(5):
        # reset root state
        root_state = wp.to_torch(articulation.data.default_root_state).clone()

        articulation.write_root_pose_to_sim(root_state[:, :7])
        articulation.write_root_velocity_to_sim(root_state[:, 7:])
        # reset dof state
        joint_pos, joint_vel = (
            wp.to_torch(articulation.data.default_joint_pos),
            wp.to_torch(articulation.data.default_joint_vel),
        )
        articulation.write_joint_state_to_sim(joint_pos, joint_vel)
        # reset articulation
        articulation.reset()
        # apply force
        # TODO: Replace with wrench composer once the deprecation is complete
        articulation.set_external_force_and_torque(
            external_wrench_b[..., :3], external_wrench_b[..., 3:], body_ids=body_ids
        )
        # perform simulation
        for _ in range(100):
            # apply action to the articulation
            articulation.set_joint_position_target(wp.to_torch(articulation.data.default_joint_pos).clone())
            articulation.write_data_to_sim()
            # perform step
            sim.step(render=False)
            # update buffers
            articulation.update(sim.cfg.dt)
        # check condition that the articulations have fallen down
        for i in range(num_articulations):
            assert wp.to_torch(articulation.data.root_pos_w)[i, 2].item() < 0.2


# FIXME: CPU is failing here. It looks like it's related to the value override we've seen before in the RigidObject
# tests.
@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("newton_cfg", [NewtonCfg(solver_cfg=SOLVER_CFGs["anymal"])])
@pytest.mark.isaacsim_ci
def test_external_force_on_single_body_at_position(sim, num_articulations, device, newton_cfg):
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
    if device == "cpu":
        pytest.skip("CPU is failing here. Needs further investigation.")
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)
    # Play the simulator
    sim.reset()

    # Find bodies to apply the force
    _, _, body_ids = articulation.find_bodies("base")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 2] = 100.0
    external_wrench_positions_b = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    external_wrench_positions_b[..., 1] = 1.0

    desired_force = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    desired_force[..., 2] = 200.0
    desired_torque = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    desired_torque[..., 0] = 200.0

    # Now we are ready!
    for i in range(5):
        # reset root state
        root_state = wp.to_torch(articulation.data.default_root_state).clone()
        root_state[0, 0] = 2.5  # space them apart by 2.5m

        articulation.write_root_pose_to_sim(root_state[:, :7])
        articulation.write_root_velocity_to_sim(root_state[:, 7:])
        # reset dof state
        joint_pos, joint_vel = (
            wp.to_torch(articulation.data.default_joint_pos),
            wp.to_torch(articulation.data.default_joint_vel),
        )
        articulation.write_joint_state_to_sim(joint_pos, joint_vel)
        # reset articulation
        articulation.reset()
        # apply force
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

        articulation.permanent_wrench_composer.set_forces_and_torques(
            body_ids=body_ids,
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            positions=external_wrench_positions_b,
            is_global=is_global,
        )
        articulation.permanent_wrench_composer.add_forces_and_torques(
            body_ids=body_ids,
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            positions=external_wrench_positions_b,
            is_global=is_global,
        )
        if not is_global:
            assert wp.to_torch(articulation.permanent_wrench_composer.composed_force)[0, 0, 2].item() == 200.0
            assert wp.to_torch(articulation.permanent_wrench_composer.composed_torque)[0, 0, 0].item() == 200.0
        # perform simulation
        for _ in range(100):
            # apply action to the articulation
            articulation.set_joint_position_target(wp.to_torch(articulation.data.default_joint_pos).clone())
            articulation.write_data_to_sim()
            # perform step
            sim.step(render=False)
            # update buffers
            articulation.update(sim.cfg.dt)
        # check condition that the articulations have fallen down
        for i in range(num_articulations):
            assert wp.to_torch(articulation.data.root_pos_w)[i, 2].item() < 0.2


# FIXME: Why is the behavior so different from PhysX? We should make a simpler test. Something with fixed joints.
@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("newton_cfg", [NewtonCfg(solver_cfg=SOLVER_CFGs["anymal"])])
@pytest.mark.parametrize("enable_gravity", [False])
@pytest.mark.isaacsim_ci
def test_external_force_on_multiple_bodies(sim, num_articulations, device, newton_cfg, enable_gravity):
    """Test application of external force on the legs of the articulation.

    This test verifies that:
    1. External forces can be applied to multiple bodies
    2. The forces affect the articulation's motion correctly
    3. The articulation responds to the forces as expected

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    if device == "cpu":
        pytest.skip("CPU is failing here. Needs further investigation.")
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, translations = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # Play the simulator
    sim.reset()

    # Find bodies to apply the force
    _, _, body_ids = articulation.find_bodies(".*_SHANK")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 1] = 10.0

    # Add translations to the root pose
    root_pose = wp.to_torch(articulation.data.default_root_state).clone()[:, :7]
    root_pose[:, :3] += translations

    # Now we are ready!
    for _ in range(5):
        # reset root state
        articulation.write_root_pose_to_sim(root_pose)
        articulation.write_root_velocity_to_sim(wp.to_torch(articulation.data.default_root_state).clone()[:, 7:])
        # reset dof state
        joint_pos, joint_vel = (
            wp.to_torch(articulation.data.default_joint_pos),
            wp.to_torch(articulation.data.default_joint_vel),
        )
        articulation.write_joint_state_to_sim(joint_pos, joint_vel)
        # reset articulation
        articulation.reset()
        # apply force
        # TODO: Replace with wrench composer once the deprecation is complete
        articulation.set_external_force_and_torque(
            external_wrench_b[..., :3], external_wrench_b[..., 3:], body_ids=body_ids
        )
        # perform simulation
        for _ in range(100):
            # apply action to the articulation
            articulation.set_joint_position_target(wp.to_torch(articulation.data.default_joint_pos).clone())
            articulation.write_data_to_sim()
            # perform step
            sim.step(render=False)
            # update buffers
            articulation.update(sim.cfg.dt)
        # check condition
        for i in range(num_articulations):
            # since there is a moment applied on the articulation, the articulation should rotate
            assert wp.to_torch(articulation.data.root_ang_vel_w)[i, 0].item() > 0.1


# FIXME: Why is the behavior so different from PhysX? We should make a simpler test. Something with fixed joints.
@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("newton_cfg", [NewtonCfg(solver_cfg=SOLVER_CFGs["anymal"])])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_external_force_on_multiple_bodies_at_position(sim, num_articulations, device, newton_cfg, gravity_enabled):
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
    if device == "cpu":
        pytest.skip("CPU is failing here. Needs further investigation.")
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, translations = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # Play the simulator
    sim.reset()

    # Find bodies to apply the force
    _, _, body_ids = articulation.find_bodies(".*_SHANK")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 2] = 5.0
    external_wrench_positions_b = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    external_wrench_positions_b[..., 1] = 1.0

    root_pose = wp.to_torch(articulation.data.default_root_state).clone()[:, :7]
    root_pose[:, :3] += translations
    # Now we are ready!
    for i in range(5):
        # reset root state
        articulation.write_root_pose_to_sim(root_pose)
        articulation.write_root_velocity_to_sim(wp.to_torch(articulation.data.default_root_state).clone()[:, 7:])
        # reset dof state
        joint_pos, joint_vel = (
            wp.to_torch(articulation.data.default_joint_pos),
            wp.to_torch(articulation.data.default_joint_vel),
        )
        articulation.write_joint_state_to_sim(joint_pos, joint_vel)
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
        articulation.permanent_wrench_composer.set_forces_and_torques(
            body_ids=body_ids,
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            positions=external_wrench_positions_b,
            is_global=is_global,
        )
        articulation.permanent_wrench_composer.add_forces_and_torques(
            body_ids=body_ids,
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            positions=external_wrench_positions_b,
            is_global=is_global,
        )
        # perform simulation
        for _ in range(100):
            # apply action to the articulation
            articulation.set_joint_position_target(wp.to_torch(articulation.data.default_joint_pos).clone())
            articulation.write_data_to_sim()
            # perform step
            sim.step(render=False)
            # update buffers
            articulation.update(sim.cfg.dt)
        # check condition
        for i in range(num_articulations):
            # since there is a moment applied on the articulation, the articulation should rotate
            assert torch.abs(wp.to_torch(articulation.data.root_ang_vel_w)[i, 0]).item() > 0.1


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
    torch.testing.assert_close(wp.to_torch(articulation.data.joint_stiffness), expected_stiffness)
    torch.testing.assert_close(wp.to_torch(articulation.data.joint_damping), expected_damping)


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
    torch.testing.assert_close(wp.to_torch(articulation.data.joint_stiffness), expected_stiffness)
    torch.testing.assert_close(wp.to_torch(articulation.data.joint_damping), expected_damping)


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
    torch.testing.assert_close(wp.to_torch(articulation.data.joint_stiffness), expected_stiffness)
    torch.testing.assert_close(wp.to_torch(articulation.data.joint_damping), expected_damping)


# FIXME: Waiting on: https://github.com/newton-physics/newton/pull/1392
# FIXME: What do we want to do about velocity limits. Vel_limit_sim and vel_limit.
# FIXME: We should probably wait for the new actuators to do this test.
@pytest.mark.skip(reason="TODO: Fix that...")
@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("vel_limit", [1e2])
@pytest.mark.parametrize("vel_limit_sim", [1e2])
@pytest.mark.isaacsim_ci
def test_setting_velocity_limit_implicit(sim, num_articulations, device, vel_limit, vel_limit_sim):
    """Test setting of velocity limit for implicit actuators.

    This test verifies that:
    1. Velocity limits can be set correctly for implicit actuators
    2. The limits are applied correctly to the simulation
    3. The limits are handled correctly when both sim and non-sim limits are set

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
        vel_limit: The velocity limit to set in actuator
    """
    # create simulation
    articulation_cfg = generate_articulation_cfg(
        articulation_type="single_joint_implicit",
        velocity_limit=vel_limit,
        velocity_limit_sim=vel_limit_sim,
    )
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg,
        num_articulations=num_articulations,
        device=device,
    )
    # Play sim
    sim.reset()

    # read the values set into the simulation. No such thin
    newton_vel_limit = wp.to_torch(
        articulation.root_view.get_attribute("joint_velocity_limit", NewtonManager.get_model())
    )
    # check data buffer
    torch.testing.assert_close(wp.to_torch(articulation.data.joint_vel_limits), newton_vel_limit)

    # check max velocity is what we set
    expected_velocity_limit = torch.full_like(newton_vel_limit, vel_limit)
    torch.testing.assert_close(newton_vel_limit, expected_velocity_limit)


# FIXME: Waiting on: https://github.com/newton-physics/newton/pull/1392
# FIXME: What do we want to do about velocity limits. Vel_limit_sim and vel_limit.
# FIXME: We should probably wait for the new actuators to do this test.
@pytest.mark.skip(reason="TODO: Fix that...")
@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("vel_limit_sim", [1e2])
@pytest.mark.parametrize("vel_limit", [1e2])
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
    newton_vel_limit = wp.to_torch(
        articulation.root_view.get_attribute("joint_velocity_limit", NewtonManager.get_model())
    )
    actuator_vel_limit = articulation.data.joint_vel_limits

    # check data buffer for joint_velocity_limits_sim
    torch.testing.assert_close(articulation.data.joint_velocity_limits, newton_vel_limit)
    # check actuator velocity_limit_sim is set to physx
    torch.testing.assert_close(actuator_vel_limit, newton_vel_limit)

    if vel_limit is not None:
        expected_actuator_vel_limit = torch.full(
            (articulation.num_instances, articulation.num_joints),
            vel_limit,
            device=articulation.device,
        )
        # check actuator is set
        torch.testing.assert_close(actuator_vel_limit, expected_actuator_vel_limit)
        # check physx is not velocity_limit
        assert not torch.allclose(actuator_vel_limit, newton_vel_limit)
    else:
        # check actuator velocity_limit is the same as the PhysX default
        torch.testing.assert_close(actuator_vel_limit, newton_vel_limit)

    # simulation velocity limit is set to USD value unless user overrides
    if vel_limit_sim is not None:
        limit = vel_limit_sim
    # else:
    #    limit = articulation_cfg.spawn.joint_drive_props.max_velocity
    # check physx is set to expected value
    expected_vel_limit = torch.full_like(newton_vel_limit, limit)
    torch.testing.assert_close(newton_vel_limit, expected_vel_limit)


# FIXME: Waiting on: https://github.com/newton-physics/newton/pull/1392
# FIXME: What do we want to do about effort limits. Effort_limit_sim and effort_limit.
# FIXME: We should probably wait for the new actuators to do this test.
@pytest.mark.skip(reason="TODO: Fix that...")
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
    physx_effort_limit = articulation.root_physx_view.get_dof_max_forces().to(device=device)

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


# FIXME: Waiting on: https://github.com/newton-physics/newton/pull/1392
# FIXME: What do we want to do about effort limits. Effort_limit_sim and effort_limit.
# FIXME: We should probably wait for the new actuators to do this test.
@pytest.mark.skip(reason="TODO: Fix that...")
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
    physx_effort_limit = articulation.root_physx_view.get_dof_max_forces().to(device)
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
    assert not articulation.instantaneous_wrench_composer.active
    assert not articulation.permanent_wrench_composer.active
    assert torch.count_nonzero(wp.to_torch(articulation.instantaneous_wrench_composer.composed_force)) == 0
    assert torch.count_nonzero(wp.to_torch(articulation.instantaneous_wrench_composer.composed_torque)) == 0
    assert torch.count_nonzero(wp.to_torch(articulation.permanent_wrench_composer.composed_force)) == 0
    assert torch.count_nonzero(wp.to_torch(articulation.permanent_wrench_composer.composed_torque)) == 0

    if num_articulations > 1:
        num_bodies = articulation.num_bodies
        # TODO: Replace with wrench composer once the deprecation is complete
        articulation.set_external_force_and_torque(
            forces=torch.ones((num_articulations, num_bodies, 3), device=device),
            torques=torch.ones((num_articulations, num_bodies, 3), device=device),
        )
        articulation.instantaneous_wrench_composer.add_forces_and_torques(
            forces=torch.ones((num_articulations, num_bodies, 3), device=device),
            torques=torch.ones((num_articulations, num_bodies, 3), device=device),
        )
        articulation.reset(env_ids=torch.tensor([0], device=device))
        assert articulation.instantaneous_wrench_composer.active
        assert articulation.permanent_wrench_composer.active
        assert (
            torch.count_nonzero(wp.to_torch(articulation.instantaneous_wrench_composer.composed_force))
            == num_bodies * 3
        )
        assert (
            torch.count_nonzero(wp.to_torch(articulation.instantaneous_wrench_composer.composed_torque))
            == num_bodies * 3
        )
        assert torch.count_nonzero(wp.to_torch(articulation.permanent_wrench_composer.composed_force)) == num_bodies * 3
        assert (
            torch.count_nonzero(wp.to_torch(articulation.permanent_wrench_composer.composed_torque)) == num_bodies * 3
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
        sim.step(render=False)
        # update buffers
        articulation.update(sim.cfg.dt)

    # reset dof state
    joint_pos = wp.to_torch(articulation.data.default_joint_pos)
    joint_pos[:, 3] = 0.0

    # apply action to the articulation
    articulation.set_joint_position_target(joint_pos)
    articulation.write_data_to_sim()

    for _ in range(100):
        # perform step
        sim.step(render=False)
        # update buffers
        articulation.update(sim.cfg.dt)

    # Check that current joint position is not the same as default joint position, meaning
    # the articulation moved. We can't check that it reached its desired joint position as the gains
    # are not properly tuned
    assert not torch.allclose(wp.to_torch(articulation.data.joint_pos), joint_pos)


# FIXME: This test is not working as expected. It looks like the pendulum is not spinning at all.
# FIXME: Could also be related to inertia update issues in MujocoWarp.
@pytest.mark.skip(reason="This test is not working as expected. It looks like the pendulum is not spinning at all.")
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
    env_idx = torch.tensor([x for x in range(num_articulations)], device=device)
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
        offset = torch.tensor([0.5, 0.0, 0.0], device=device).repeat(num_articulations, 1, 1)
    else:
        offset = torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_articulations, 1, 1)

    # create com offsets
    num_bodies = articulation.num_bodies
    link_offset = [1.0, 0.0, 0.0]  # the offset from CenterPivot to Arm frames
    # Newton only stores the position of the center of mass, so we need to get the position from the pose.
    com = wp.to_torch(articulation.data.body_com_pos_b) + offset
    articulation.set_coms(com, env_ids=env_idx)

    # check they are set
    torch.testing.assert_close(wp.to_torch(articulation.data.body_com_pos_b), com)

    for i in range(50):
        # perform step
        sim.step(render=False)
        # update buffers
        articulation.update(sim.cfg.dt)

        # get state properties
        root_state_w = wp.to_torch(articulation.data.root_state_w)
        root_link_state_w = wp.to_torch(articulation.data.root_link_state_w)
        root_com_state_w = wp.to_torch(articulation.data.root_com_state_w)
        body_state_w = wp.to_torch(articulation.data.body_state_w)
        body_link_state_w = wp.to_torch(articulation.data.body_link_state_w)
        body_com_state_w = wp.to_torch(articulation.data.body_com_state_w)

        if with_offset:
            # get joint state
            joint_pos = wp.to_torch(articulation.data.joint_pos).unsqueeze(-1)
            joint_vel = wp.to_torch(articulation.data.joint_vel).unsqueeze(-1)

            # LINK state
            # pose
            torch.testing.assert_close(root_state_w[..., :7], root_link_state_w[..., :7])
            torch.testing.assert_close(body_state_w[..., :7], body_link_state_w[..., :7])

            # lin_vel arm
            lin_vel_gt = torch.zeros(num_articulations, num_bodies, 3, device=device)
            vx = -(link_offset[0]) * joint_vel * torch.sin(joint_pos)
            vy = torch.zeros(num_articulations, 1, 1, device=device)
            vz = (link_offset[0]) * joint_vel * torch.cos(joint_pos)
            lin_vel_gt[:, 1, :] = torch.cat([vx, vy, vz], dim=-1).squeeze(-2)

            # linear velocity of root link should be zero
            torch.testing.assert_close(lin_vel_gt[:, 0, :], root_link_state_w[..., 7:10], atol=1e-3, rtol=1e-1)
            # linear velocity of pendulum link should be
            torch.testing.assert_close(lin_vel_gt, body_link_state_w[..., 7:10], atol=1e-3, rtol=1e-1)

            # ang_vel
            torch.testing.assert_close(root_state_w[..., 10:], root_link_state_w[..., 10:])
            torch.testing.assert_close(body_state_w[..., 10:], body_link_state_w[..., 10:])

            # COM state
            # position and orientation shouldn't match for the _state_com_w but everything else will
            pos_gt = torch.zeros(num_articulations, num_bodies, 3, device=device)
            px = (link_offset[0] + offset[0]) * torch.cos(joint_pos)
            py = torch.zeros(num_articulations, 1, 1, device=device)
            pz = (link_offset[0] + offset[0]) * torch.sin(joint_pos)
            pos_gt[:, 1, :] = torch.cat([px, py, pz], dim=-1).squeeze(-2)
            pos_gt += env_pos.unsqueeze(-2).repeat(1, num_bodies, 1)
            torch.testing.assert_close(pos_gt[:, 0, :], root_com_state_w[..., :3], atol=1e-3, rtol=1e-1)
            torch.testing.assert_close(pos_gt, body_com_state_w[..., :3], atol=1e-3, rtol=1e-1)

            # orientation
            com_quat_b = wp.to_torch(articulation.data.body_com_quat_b)
            com_quat_w = math_utils.quat_mul(body_link_state_w[..., 3:7], com_quat_b)
            torch.testing.assert_close(com_quat_w, body_com_state_w[..., 3:7])
            torch.testing.assert_close(com_quat_w[:, 0, :], root_com_state_w[..., 3:7])

            # linear vel, and angular vel
            torch.testing.assert_close(root_state_w[..., 7:], root_com_state_w[..., 7:])
            torch.testing.assert_close(body_state_w[..., 7:], body_com_state_w[..., 7:])
        else:
            # single joint center of masses are at link frames so they will be the same
            torch.testing.assert_close(root_state_w, root_link_state_w)
            torch.testing.assert_close(root_state_w, root_com_state_w)
            torch.testing.assert_close(body_state_w, body_link_state_w)
            torch.testing.assert_close(body_state_w, body_com_state_w)


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
    env_idx = torch.tensor([x for x in range(num_articulations)])

    # Play sim
    sim.reset()

    # change center of mass offset from link frame
    if with_offset:
        offset = torch.tensor([1.0, 0.0, 0.0]).repeat(num_articulations, 1, 1)
    else:
        offset = torch.tensor([0.0, 0.0, 0.0]).repeat(num_articulations, 1, 1)

    # create com offsets
    com = wp.to_torch(articulation.data.body_com_pos_b)
    new_com = offset
    com[:, 0, :3] = new_com.squeeze(-2)
    articulation.set_coms(com, env_ids=env_idx)

    # check they are set
    torch.testing.assert_close(wp.to_torch(articulation.data.body_com_pos_b), com)

    rand_state = torch.zeros_like(wp.to_torch(articulation.data.root_state_w))
    rand_state[..., :7] = wp.to_torch(articulation.data.default_root_state)[..., :7]
    rand_state[..., :3] += env_pos
    # make quaternion a unit vector
    rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)

    env_idx = env_idx.to(device)
    for i in range(10):
        # perform step
        sim.step(render=False)
        # update buffers
        articulation.update(sim.cfg.dt)

        if state_location == "com":
            if i % 2 == 0:
                articulation.write_root_com_state_to_sim(rand_state)
            else:
                articulation.write_root_com_state_to_sim(rand_state, env_ids=env_idx)
        elif state_location == "link":
            if i % 2 == 0:
                articulation.write_root_link_state_to_sim(rand_state)
            else:
                articulation.write_root_link_state_to_sim(rand_state, env_ids=env_idx)

        if state_location == "com":
            torch.testing.assert_close(rand_state, wp.to_torch(articulation.data.root_com_state_w))
        elif state_location == "link":
            torch.testing.assert_close(rand_state, wp.to_torch(articulation.data.root_link_state_w))


# FIXME: Functionality is not available yet.
@pytest.mark.skip(reason="Functionality is not available yet.")
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
    joint_pos = torch.ones_like(articulation.data.joint_pos) * 1.5708 / 2.0
    articulation.write_joint_state_to_sim(
        torch.ones_like(articulation.data.joint_pos), torch.zeros_like(articulation.data.joint_vel)
    )
    articulation.set_joint_position_target(joint_pos)
    articulation.write_data_to_sim()
    for _ in range(50):
        # TODO: Replace with wrench composer once the deprecation is complete
        articulation.set_external_force_and_torque(forces=external_force_vector_b, torques=external_torque_vector_b)
        articulation.write_data_to_sim()
        # perform step
        sim.step(render=False)
        # update buffers
        articulation.update(sim.cfg.dt)

        # check shape
        assert articulation.data.body_incoming_joint_wrench_b.shape == (num_articulations, articulation.num_bodies, 6)

    # calculate expected static
    mass = articulation.data.default_mass
    pos_w = articulation.data.body_pos_w
    quat_w = articulation.data.body_quat_w

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
        articulation.data.body_incoming_joint_wrench_b[:, 1, :].squeeze(1),
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
    with pytest.raises(KeyError):
        sim.reset()


# FIXME: Articulation.write_joint_position_limit_to_sim should not take two variables as arguments. but a single one.
# FIXME: Should have a new method that can do both.
# FIXME: Forward Kinematics call is needed to update the body_state after writing the joint position limits.
# Do we want to update it automatically after writing the joint position limits? It could be expensive and useless.
# FIXME: Double danger... New to update the articulation too...
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
    articulation.write_joint_position_limit_to_sim(limits[..., 0], limits[..., 1])

    from torch.distributions import Uniform

    pos_dist = Uniform(
        wp.to_torch(articulation.data.joint_pos_limits)[..., 0], wp.to_torch(articulation.data.joint_pos_limits)[..., 1]
    )
    vel_dist = Uniform(
        -wp.to_torch(articulation.data.joint_vel_limits), wp.to_torch(articulation.data.joint_vel_limits)
    )

    original_body_states = wp.to_torch(articulation.data.body_state_w).clone()

    rand_joint_pos = pos_dist.sample()
    rand_joint_vel = vel_dist.sample()

    articulation.write_joint_state_to_sim(rand_joint_pos, rand_joint_vel)
    # FIXME: Should this be needed?
    NewtonManager.forward_kinematics()
    articulation.update(sim.cfg.dt)

    # make sure the values are updated
    assert torch.count_nonzero(original_body_states[:, 1:] != wp.to_torch(articulation.data.body_state_w)[:, 1:]) > (
        len(original_body_states[:, 1:]) / 2
    )
    # validate body - link consistency
    torch.testing.assert_close(
        wp.to_torch(articulation.data.body_state_w)[..., :7], wp.to_torch(articulation.data.body_link_state_w)[..., :7]
    )
    # skip 7:10 because they differs from link frame, this should be fine because we are only checking
    # if velocity update is triggered, which can be determined by comparing angular velocity
    torch.testing.assert_close(
        wp.to_torch(articulation.data.body_state_w)[..., 10:],
        wp.to_torch(articulation.data.body_link_state_w)[..., 10:],
    )

    # validate link - com conistency
    expected_com_pos, expected_com_quat = math_utils.combine_frame_transforms(
        wp.to_torch(articulation.data.body_link_state_w)[..., :3].view(-1, 3),
        wp.to_torch(articulation.data.body_link_state_w)[..., 3:7].view(-1, 4),
        wp.to_torch(articulation.data.body_com_pos_b).view(-1, 3),
        wp.to_torch(articulation.data.body_com_quat_b).view(-1, 4),
    )
    torch.testing.assert_close(
        expected_com_pos.view(len(env_idx), -1, 3), wp.to_torch(articulation.data.body_com_pos_w)
    )
    torch.testing.assert_close(
        expected_com_quat.view(len(env_idx), -1, 4), wp.to_torch(articulation.data.body_com_quat_w)
    )

    # validate body - com consistency
    torch.testing.assert_close(
        wp.to_torch(articulation.data.body_state_w)[..., 7:10], wp.to_torch(articulation.data.body_com_lin_vel_w)
    )
    torch.testing.assert_close(
        wp.to_torch(articulation.data.body_state_w)[..., 10:], wp.to_torch(articulation.data.body_com_ang_vel_w)
    )

    # validate pos_w, quat_w, pos_b, quat_b is consistent with pose_w and pose_b
    expected_com_pose_w = torch.cat(
        (wp.to_torch(articulation.data.body_com_pos_w), wp.to_torch(articulation.data.body_com_quat_w)), dim=2
    )
    expected_com_pose_b = torch.cat(
        (wp.to_torch(articulation.data.body_com_pos_b), wp.to_torch(articulation.data.body_com_quat_b)), dim=2
    )
    expected_body_pose_w = torch.cat(
        (wp.to_torch(articulation.data.body_pos_w), wp.to_torch(articulation.data.body_quat_w)), dim=2
    )
    expected_body_link_pose_w = torch.cat(
        (wp.to_torch(articulation.data.body_link_pos_w), wp.to_torch(articulation.data.body_link_quat_w)), dim=2
    )
    torch.testing.assert_close(wp.to_torch(articulation.data.body_com_pose_w), expected_com_pose_w)
    torch.testing.assert_close(wp.to_torch(articulation.data.body_com_pose_b), expected_com_pose_b)
    torch.testing.assert_close(wp.to_torch(articulation.data.body_pose_w), expected_body_pose_w)
    torch.testing.assert_close(wp.to_torch(articulation.data.body_link_pose_w), expected_body_link_pose_w)

    # validate pose_w is consistent state[..., :7]
    torch.testing.assert_close(
        wp.to_torch(articulation.data.body_pose_w), wp.to_torch(articulation.data.body_state_w)[..., :7]
    )
    torch.testing.assert_close(
        wp.to_torch(articulation.data.body_vel_w), wp.to_torch(articulation.data.body_state_w)[..., 7:]
    )
    torch.testing.assert_close(
        wp.to_torch(articulation.data.body_link_pose_w), wp.to_torch(articulation.data.body_link_state_w)[..., :7]
    )
    torch.testing.assert_close(
        wp.to_torch(articulation.data.body_com_pose_w), wp.to_torch(articulation.data.body_com_state_w)[..., :7]
    )
    torch.testing.assert_close(
        wp.to_torch(articulation.data.body_vel_w), wp.to_torch(articulation.data.body_state_w)[..., 7:]
    )


@pytest.mark.skip(reason="Functionality is not available yet.")
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
    assert articulation.data.root_pos_w.shape == (num_articulations, 3)
    assert articulation.data.root_quat_w.shape == (num_articulations, 4)
    assert articulation.data.joint_pos.shape == (num_articulations, 3)
    assert articulation.data.mass.shape == (num_articulations, articulation.num_bodies)
    assert articulation.data.inertia.shape == (num_articulations, articulation.num_bodies, 9)
    assert articulation.num_spatial_tendons == 1

    articulation.set_spatial_tendon_stiffness(torch.tensor([10.0]))
    articulation.set_spatial_tendon_limit_stiffness(torch.tensor([10.0]))
    articulation.set_spatial_tendon_damping(torch.tensor([10.0]))
    articulation.set_spatial_tendon_offset(torch.tensor([10.0]))

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step(render=False)
        # update articulation
        articulation.update(sim.cfg.dt)


# FIXME: Functionality is not available yet.
@pytest.mark.skip(reason="Functionality is not available yet.")
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
        sim.step(render=False)
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
    articulation.write_joint_friction_coefficient_to_sim(friction)
    articulation.write_joint_dynamic_friction_coefficient_to_sim(dynamic_friction)
    articulation.write_joint_viscous_friction_coefficient_to_sim(viscous_friction)
    articulation.write_data_to_sim()

    for _ in range(100):
        # perform step
        sim.step(render=False)
        # update buffers
        articulation.update(sim.cfg.dt)

    friction_props_from_sim = articulation.root_physx_view.get_dof_friction_properties()
    joint_friction_coeff_sim = friction_props_from_sim[:, :, 0]
    joint_dynamic_friction_coeff_sim = friction_props_from_sim[:, :, 1]
    joint_viscous_friction_coeff_sim = friction_props_from_sim[:, :, 2]
    assert torch.allclose(joint_dynamic_friction_coeff_sim, dynamic_friction.cpu())
    assert torch.allclose(joint_viscous_friction_coeff_sim, viscous_friction.cpu())

    assert torch.allclose(joint_friction_coeff_sim, friction.cpu())

    # For Isaac Sim >= 5.0: also test the combined API that can set dynamic and viscous via
    # write_joint_friction_coefficient_to_sim; reset the sim to isolate this path.
    # Reset simulator to ensure a clean state for the alternative API path
    sim.reset()

    # Warm up a few steps to populate buffers
    for _ in range(100):
        sim.step(render=False)
        articulation.update(sim.cfg.dt)

    # New random coefficients
    dynamic_friction_2 = torch.rand(num_articulations, articulation.num_joints, device=device)
    viscous_friction_2 = torch.rand(num_articulations, articulation.num_joints, device=device)
    friction_2 = torch.rand(num_articulations, articulation.num_joints, device=device)

    # Guarantee that the dynamic friction is not greater than the static friction
    dynamic_friction_2 = torch.min(dynamic_friction_2, friction_2)

    # Use the combined setter to write all three at once
    articulation.write_joint_friction_coefficient_to_sim(
        joint_friction_coeff=friction_2,
        joint_dynamic_friction_coeff=dynamic_friction_2,
        joint_viscous_friction_coeff=viscous_friction_2,
    )
    articulation.write_data_to_sim()

    # Step to let sim ingest new params and refresh data buffers
    for _ in range(100):
        sim.step(render=False)
        articulation.update(sim.cfg.dt)

    friction_props_from_sim_2 = articulation.root_view.get_dof_friction_properties()
    joint_friction_coeff_sim_2 = friction_props_from_sim_2[:, :, 0]
    friction_dynamic_coef_sim_2 = friction_props_from_sim_2[:, :, 1]
    friction_viscous_coeff_sim_2 = friction_props_from_sim_2[:, :, 2]

    # Validate values propagated
    assert torch.allclose(friction_viscous_coeff_sim_2, viscous_friction_2.cpu())
    assert torch.allclose(friction_dynamic_coef_sim_2, dynamic_friction_2.cpu())
    assert torch.allclose(joint_friction_coeff_sim_2, friction_2.cpu())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
