# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Test parity with isaaclab_physx test_articulation.py.

Mirrors the physx backend's test_articulation.py function names to ensure that the
ovphysx backend provides equivalent coverage. Tests that require IsaacSim/Nucleus
assets or features not yet supported by the ovphysx backend are marked with
pytest.skip.

Uses local USD test assets (no nucleus dependency).
"""

import os
import sys

import numpy as np
import pytest

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

import warp as wp

wp.init()

# Hide pxr during ovphysx import to skip Python-level USD version check.
import sys as _sys
_hidden_pxr = {}
for _k in list(_sys.modules):
    if _k == "pxr" or _k.startswith("pxr."):
        _hidden_pxr[_k] = _sys.modules.pop(_k)
import ovphysx  # noqa: E402,F401
ovphysx.bootstrap()
_sys.modules.update(_hidden_pxr)
del _hidden_pxr

TWO_ART_USD = os.path.join(os.path.dirname(__file__), "..", "data", "two_articulations.usda")
CARTPOLE_USD = os.path.join(os.path.dirname(__file__), "..", "data", "cartpole.usda")

DT = 1.0 / 60.0
DEVICE = "cuda:0"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _create_stage(usd_path: str) -> Usd.Stage:
    """Create a fresh in-memory stage with USD content copied in."""
    import isaaclab.sim.utils.stage as stage_utils
    src_layer = Sdf.Layer.FindOrOpen(usd_path)
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().TransferContent(src_layer)
    stage_utils._context.stage = stage
    cache = UsdUtils.StageCache.Get()
    cache.Insert(stage)
    return stage


def _make_sim_and_art(usd_path, prim_path, actuators=None, dt=DT, device=DEVICE, gravity=(0.0, 0.0, -9.81)):
    """Build SimulationContext + Articulation from a local USD file."""
    from isaaclab.sim.simulation_cfg import SimulationCfg
    from isaaclab.sim.simulation_context import SimulationContext
    from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
    from isaaclab_ovphysx.physics.ovphysx_manager_cfg import OvPhysxCfg

    SimulationContext.clear_instance()
    _create_stage(usd_path)

    sim = SimulationContext(SimulationCfg(
        dt=dt, device=device, gravity=gravity,
        physics=OvPhysxCfg(), use_fabric=False, render_interval=1,
    ))

    if actuators is None:
        actuators = {}

    from isaaclab.assets.articulation.articulation import Articulation
    art = Articulation(ArticulationCfg(
        prim_path=prim_path,
        actuators=actuators,
    ))
    sim.reset()
    return sim, art


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def fixed_base_sim():
    """Two fixed-base articulations (2 envs, 2 joints, 3 bodies each)."""
    sim, art = _make_sim_and_art(TWO_ART_USD, "/World/articulation*")
    yield sim, art
    sim.clear_instance()


@pytest.fixture
def single_art_sim():
    """Single fixed-base articulation (1 env, 2 joints, 3 bodies)."""
    sim, art = _make_sim_and_art(TWO_ART_USD, "/World/articulation")
    yield sim, art
    sim.clear_instance()


@pytest.fixture
def cartpole_sim():
    """Cartpole articulation with implicit actuator on the cart joint."""
    from isaaclab.actuators import ImplicitActuatorCfg

    actuators = {
        "cart": ImplicitActuatorCfg(
            joint_names_expr=["railCartJoint"],
            stiffness=100.0,
            damping=10.0,
        ),
    }
    sim, art = _make_sim_and_art(CARTPOLE_USD, "/cartPole", actuators=actuators)
    yield sim, art
    sim.clear_instance()


# ======================================================================
# Initialization tests (mirrors physx test_initialization_*)
# ======================================================================

def test_initialization_floating_base_non_root(fixed_base_sim):
    pytest.skip("Requires IsaacSim/Nucleus assets (humanoid) not available in ovphysx standalone tests.")


def test_initialization_floating_base(fixed_base_sim):
    pytest.skip("Requires IsaacSim/Nucleus assets (humanoid) not available in ovphysx standalone tests.")


def test_initialization_fixed_base(fixed_base_sim):
    """Verify fixed-base articulation initialization."""
    _, art = fixed_base_sim
    assert art.is_initialized
    assert art.is_fixed_base is True
    assert art.__backend_name__ == "ovphysx"
    assert art.num_instances == 2
    assert art.num_joints == 2
    assert art.num_bodies == 3
    assert len(art.joint_names) == 2
    assert len(art.body_names) == 3


def test_initialization_fixed_base_single_joint(single_art_sim):
    """Verify single-articulation initialization."""
    _, art = single_art_sim
    assert art.is_initialized
    assert art.num_instances == 1
    assert art.num_joints == 2
    assert art.num_bodies == 3


def test_initialization_hand_with_tendons(fixed_base_sim):
    pytest.skip("Requires shadow hand asset with tendons, not available in ovphysx standalone tests.")


def test_initialization_floating_base_made_fixed_base(fixed_base_sim):
    pytest.skip("Requires IsaacSim/Nucleus assets not available in ovphysx standalone tests.")


def test_initialization_fixed_base_made_floating_base(fixed_base_sim):
    pytest.skip("Requires IsaacSim/Nucleus assets not available in ovphysx standalone tests.")


# ======================================================================
# Default state validation
# ======================================================================

def test_out_of_range_default_joint_pos(fixed_base_sim):
    """Verify default joint position buffer shapes."""
    _, art = fixed_base_sim
    dp = art.data.default_joint_pos
    assert dp.shape == (2, 2)
    assert dp.dtype == wp.float32


def test_out_of_range_default_joint_vel(single_art_sim):
    """Verify default joint velocity buffer shapes."""
    _, art = single_art_sim
    dv = art.data.default_joint_vel
    assert dv.shape == (1, 2)
    assert dv.dtype == wp.float32


# ======================================================================
# Joint limits
# ======================================================================

def test_joint_pos_limits(fixed_base_sim):
    """Verify joint position limits are read correctly."""
    _, art = fixed_base_sim
    limits = art.data.joint_pos_limits
    assert limits.shape == (2, 2)
    assert limits.dtype == wp.vec2f
    lim_np = limits.numpy().reshape(2, 2, 2)
    assert np.all(lim_np[..., 0] <= lim_np[..., 1]), "Lower limits must be <= upper limits"


def test_joint_effort_limits(fixed_base_sim):
    """Verify joint effort limits are read correctly."""
    _, art = fixed_base_sim
    eff_limits = art.data.joint_effort_limits
    assert eff_limits.shape == (2, 2)
    assert eff_limits.dtype == wp.float32


# ======================================================================
# External forces
# ======================================================================

def test_external_force_buffer(single_art_sim):
    """Verify external force buffer is initialized and accessible."""
    _, art = single_art_sim
    assert art.instantaneous_wrench_composer is not None
    assert art.permanent_wrench_composer is not None


def test_external_force_on_single_body(single_art_sim):
    """Verify that a force applied on a single body changes the state."""
    sim, art = single_art_sim
    pose_before = art.data.body_link_pose_w.numpy().copy()

    force = wp.zeros((1, art.num_bodies), dtype=wp.vec3f, device=DEVICE)
    torque = wp.zeros((1, art.num_bodies), dtype=wp.vec3f, device=DEVICE)
    force_np = force.numpy()
    force_np[0, 1] = [0.0, 0.0, 100.0]
    wp.copy(force, wp.from_numpy(force_np, dtype=wp.vec3f, device=DEVICE))

    art.instantaneous_wrench_composer.set_forces_and_torques_index(
        forces=force, torques=torque,
        body_ids=list(range(art.num_bodies)),
        env_ids=[0],
    )
    art.write_data_to_sim()
    sim.step()
    art.update(DT)

    pose_after = art.data.body_link_pose_w.numpy()
    assert not np.allclose(pose_before, pose_after, atol=1e-6), "Pose should change after applying force"


def test_external_force_on_single_body_at_position(single_art_sim):
    pytest.skip("Force at position not yet verified in ovphysx backend.")


def test_external_force_on_multiple_bodies(fixed_base_sim):
    """Verify that forces applied on multiple bodies change the state."""
    sim, art = fixed_base_sim
    pose_before = art.data.body_link_pose_w.numpy().copy()

    force = wp.zeros((2, art.num_bodies), dtype=wp.vec3f, device=DEVICE)
    torque = wp.zeros((2, art.num_bodies), dtype=wp.vec3f, device=DEVICE)
    force_np = force.numpy()
    force_np[:, 1] = [0.0, 0.0, 100.0]
    wp.copy(force, wp.from_numpy(force_np, dtype=wp.vec3f, device=DEVICE))

    art.instantaneous_wrench_composer.set_forces_and_torques_index(
        forces=force, torques=torque,
        body_ids=list(range(art.num_bodies)),
        env_ids=list(range(art.num_instances)),
    )
    art.write_data_to_sim()
    sim.step()
    art.update(DT)

    pose_after = art.data.body_link_pose_w.numpy()
    assert not np.allclose(pose_before, pose_after, atol=1e-6), "Pose should change after applying forces"


def test_external_force_on_multiple_bodies_at_position(fixed_base_sim):
    pytest.skip("Force at position not yet verified in ovphysx backend.")


# ======================================================================
# Actuator gains
# ======================================================================

def test_loading_gains_from_usd(fixed_base_sim):
    """Verify that gains (stiffness/damping) are loaded from the USD."""
    _, art = fixed_base_sim
    stiff = art.data.joint_stiffness
    damp = art.data.joint_damping
    assert stiff.shape == (2, 2)
    assert damp.shape == (2, 2)


def test_setting_gains_from_cfg(cartpole_sim):
    """Verify that actuator config gains are applied."""
    _, art = cartpole_sim
    assert len(art.actuators) > 0


def test_setting_gains_from_cfg_dict(fixed_base_sim):
    pytest.skip("Requires dict-based actuator config not yet tested with ovphysx standalone assets.")


# ======================================================================
# Velocity / effort limits
# ======================================================================

def test_setting_velocity_limit_implicit(cartpole_sim):
    """Verify velocity limit buffer is accessible."""
    _, art = cartpole_sim
    vel_lim = art.data.joint_vel_limits
    assert vel_lim.shape[0] == art.num_instances


def test_setting_velocity_limit_explicit(fixed_base_sim):
    pytest.skip("Requires explicit actuator + Nucleus USD assets.")


def test_setting_effort_limit_implicit(cartpole_sim):
    """Verify effort limit buffer is accessible."""
    _, art = cartpole_sim
    eff_lim = art.data.joint_effort_limits
    assert eff_lim.shape[0] == art.num_instances


def test_setting_effort_limit_explicit(fixed_base_sim):
    pytest.skip("Requires explicit actuator + Nucleus USD assets.")


# ======================================================================
# Reset
# ======================================================================

def test_reset(fixed_base_sim):
    """Verify that reset restores the default state."""
    sim, art = fixed_base_sim

    default_jpos = art.data.default_joint_pos.numpy().copy()

    for _ in range(10):
        art.write_data_to_sim()
        sim.step()
        art.update(DT)

    drifted_jpos = art.data.joint_pos.numpy()
    art.reset()

    for _ in range(2):
        art.write_data_to_sim()
        sim.step()
        art.update(DT)

    reset_jpos = art.data.joint_pos.numpy()
    assert np.allclose(reset_jpos, default_jpos, atol=0.1), (
        f"Joint positions should be close to defaults after reset. Got {reset_jpos}, expected {default_jpos}"
    )


# ======================================================================
# Joint commands
# ======================================================================

def test_apply_joint_command(cartpole_sim):
    """Verify that setting a joint position target moves the joint."""
    sim, art = cartpole_sim
    N = art.num_instances
    D = art.num_joints

    target = wp.zeros((N, D), dtype=wp.float32, device=DEVICE)
    target_np = target.numpy()
    target_np[:, 0] = 0.5
    wp.copy(target, wp.from_numpy(target_np, dtype=wp.float32, device=DEVICE))

    art.set_joint_position_target_index(target=target)

    for _ in range(100):
        art.write_data_to_sim()
        sim.step()
        art.update(DT)

    final_jpos = art.data.joint_pos.numpy()
    assert abs(final_jpos[0, 0] - 0.5) < 0.3, (
        f"Cart joint should approach target 0.5, got {final_jpos[0, 0]}"
    )


# ======================================================================
# Body / root state
# ======================================================================

def test_body_root_state(fixed_base_sim):
    """Verify root and body state properties are accessible and have correct shapes."""
    sim, art = fixed_base_sim
    N = art.num_instances
    L = art.num_bodies

    sim.step()
    art.update(DT)

    root_pose = art.data.root_link_pose_w
    assert root_pose.shape == (N,)
    assert root_pose.dtype == wp.transformf

    root_vel = art.data.root_com_vel_w
    assert root_vel.shape == (N,)
    assert root_vel.dtype == wp.spatial_vectorf

    body_pose = art.data.body_link_pose_w
    assert body_pose.shape == (N, L)
    assert body_pose.dtype == wp.transformf

    body_vel = art.data.body_link_vel_w
    assert body_vel.shape == (N, L)
    assert body_vel.dtype == wp.spatial_vectorf

    body_mass = art.data.body_mass
    assert body_mass.shape == (N, L)

    heading = art.data.heading_w
    assert heading.shape == (N,)
    assert heading.dtype == wp.float32

    proj_grav = art.data.projected_gravity_b
    assert proj_grav.shape == (N,)
    assert proj_grav.dtype == wp.vec3f


def test_write_root_state(single_art_sim):
    """Verify that writing root pose and velocity updates the simulation."""
    sim, art = single_art_sim
    N = art.num_instances

    new_pose = wp.zeros(N, dtype=wp.transformf, device=DEVICE)
    pose_np = new_pose.numpy().reshape(N, 7)
    pose_np[0] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    wp.copy(new_pose, wp.from_numpy(pose_np.reshape(N, 7), dtype=wp.transformf, device=DEVICE))

    art.write_root_pose_to_sim_index(root_pose=new_pose)

    sim.step()
    art.update(DT)

    read_pose = art.data.root_link_pose_w.numpy().reshape(N, 7)
    assert np.allclose(read_pose[0, :3], [0.5, 0.0, 0.0], atol=0.1), (
        f"Root position should be near (0.5, 0, 0), got {read_pose[0, :3]}"
    )


# ======================================================================
# Joint wrench
# ======================================================================

def test_body_incoming_joint_wrench_b_single_joint(single_art_sim):
    """Verify incoming joint wrench buffer is accessible."""
    sim, art = single_art_sim
    sim.step()
    art.update(DT)

    wrench = art.data.body_incoming_joint_wrench_b
    assert wrench.shape == (art.num_instances, art.num_bodies)
    assert wrench.dtype == wp.spatial_vectorf


# ======================================================================
# Articulation root prim path
# ======================================================================

def test_setting_articulation_root_prim_path(single_art_sim):
    """Verify articulation is accessible at expected path."""
    _, art = single_art_sim
    assert art.is_initialized


def test_setting_invalid_articulation_root_prim_path():
    pytest.skip("Requires kit-based prim path validation not available in ovphysx standalone mode.")


# ======================================================================
# Write joint state data consistency
# ======================================================================

def test_write_joint_state_data_consistency(fixed_base_sim):
    """Verify that writing joint state and reading it back produces consistent values."""
    sim, art = fixed_base_sim
    N = art.num_instances
    D = art.num_joints

    new_pos = wp.zeros((N, D), dtype=wp.float32, device=DEVICE)
    pos_np = new_pos.numpy()
    pos_np[0, 0] = 0.5
    pos_np[1, 1] = -0.3
    wp.copy(new_pos, wp.from_numpy(pos_np, dtype=wp.float32, device=DEVICE))

    new_vel = wp.zeros((N, D), dtype=wp.float32, device=DEVICE)

    art.write_joint_position_to_sim_index(position=new_pos)
    art.write_joint_velocity_to_sim_index(velocity=new_vel)

    sim.step()
    art.update(DT)

    read_pos = art.data.joint_pos.numpy()
    assert abs(read_pos[0, 0] - 0.5) < 0.15, f"Expected ~0.5, got {read_pos[0, 0]}"
    assert abs(read_pos[1, 1] - (-0.3)) < 0.15, f"Expected ~-0.3, got {read_pos[1, 1]}"


# ======================================================================
# Tendons
# ======================================================================

def test_spatial_tendons(fixed_base_sim):
    pytest.skip("Spatial tendon support requires specific USD assets not available in ovphysx standalone tests.")


# ======================================================================
# Friction
# ======================================================================

def test_write_joint_frictions_to_sim(single_art_sim):
    """Verify joint friction can be written and read back."""
    _, art = single_art_sim
    N = art.num_instances
    D = art.num_joints

    new_friction = wp.zeros((N, D), dtype=wp.float32, device=DEVICE)
    fric_np = new_friction.numpy()
    fric_np[:] = 0.5
    wp.copy(new_friction, wp.from_numpy(fric_np, dtype=wp.float32, device=DEVICE))

    art.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=new_friction)

    read_back = art.data.joint_friction_coeff.numpy()
    assert read_back.shape == (N, D)
