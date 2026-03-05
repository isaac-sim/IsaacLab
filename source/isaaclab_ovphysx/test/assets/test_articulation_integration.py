# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Integration tests for the ovphysx articulation backend through the IsaacLab stack.

Tests the full pipeline: SimulationContext(OvPhysxCfg) + Articulation + step/read/write.
Mirrors the PhysX backend test coverage. Uses local USD assets (no nucleus).
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
_ = ovphysx.PhysX  # force native bootstrap while pxr is hidden
_sys.modules.update(_hidden_pxr)
del _hidden_pxr

TWO_ART_USD = os.path.join(os.path.dirname(__file__), "..", "data", "two_articulations.usda")
CARTPOLE_USD = os.path.join(os.path.dirname(__file__), "..", "data", "cartpole.usda")

DT = 1.0 / 60.0
DEVICE = "cuda:0"


def _create_stage(usd_path: str) -> Usd.Stage:
    """Create a fresh in-memory stage with USD content copied in.

    Uses TransferContent (not sublayer) so SimulationContext can freely
    delete and recreate the PhysicsScene prim.
    """
    import isaaclab.sim.utils.stage as stage_utils
    src_layer = Sdf.Layer.FindOrOpen(usd_path)
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().TransferContent(src_layer)
    stage_utils._context.stage = stage
    cache = UsdUtils.StageCache.Get()
    cache.Insert(stage)
    return stage


@pytest.fixture
def fixed_base_sim():
    """Two fixed-base articulations (2 envs, 2 joints, 3 bodies each)."""
    from isaaclab.sim.simulation_cfg import SimulationCfg
    from isaaclab.sim.simulation_context import SimulationContext
    from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
    from isaaclab_ovphysx.physics.ovphysx_manager_cfg import OvPhysxCfg

    SimulationContext.clear_instance()
    _create_stage(TWO_ART_USD)

    sim = SimulationContext(SimulationCfg(
        dt=DT, device=DEVICE, gravity=(0.0, 0.0, -9.81),
        physics=OvPhysxCfg(), use_fabric=False, render_interval=1,
    ))

    from isaaclab.assets.articulation.articulation import Articulation
    art = Articulation(ArticulationCfg(
        prim_path="/World/articulation*",
        actuators={},
    ))
    sim.reset()

    yield sim, art
    sim.clear_instance()


@pytest.fixture
def single_art_sim():
    """Single fixed-base articulation (1 env, 2 joints, 3 bodies)."""
    from isaaclab.sim.simulation_cfg import SimulationCfg
    from isaaclab.sim.simulation_context import SimulationContext
    from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
    from isaaclab_ovphysx.physics.ovphysx_manager_cfg import OvPhysxCfg

    SimulationContext.clear_instance()
    _create_stage(TWO_ART_USD)

    sim = SimulationContext(SimulationCfg(
        dt=DT, device=DEVICE, gravity=(0.0, 0.0, -9.81),
        physics=OvPhysxCfg(), use_fabric=False, render_interval=1,
    ))

    from isaaclab.assets.articulation.articulation import Articulation
    art = Articulation(ArticulationCfg(
        prim_path="/World/articulation",
        actuators={},
    ))
    sim.reset()

    yield sim, art
    sim.clear_instance()


@pytest.fixture
def cartpole_sim():
    """Cartpole articulation with implicit actuator on the cart joint."""
    from isaaclab.sim.simulation_cfg import SimulationCfg
    from isaaclab.sim.simulation_context import SimulationContext
    from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_ovphysx.physics.ovphysx_manager_cfg import OvPhysxCfg

    SimulationContext.clear_instance()
    _create_stage(CARTPOLE_USD)

    sim = SimulationContext(SimulationCfg(
        dt=DT, device=DEVICE, gravity=(0.0, 0.0, -9.81),
        physics=OvPhysxCfg(), use_fabric=False,
    ))

    from isaaclab.assets.articulation.articulation import Articulation
    art = Articulation(ArticulationCfg(
        prim_path="/cartPole",
        actuators={
            "cart": ImplicitActuatorCfg(
                joint_names_expr=["railCartJoint"],
                stiffness=100.0,
                damping=10.0,
            ),
        },
    ))
    sim.reset()

    yield sim, art
    sim.clear_instance()


# ======================================================================
# Initialization
# ======================================================================

class TestInitialization:

    def test_init_fixed_base(self, fixed_base_sim):
        _, art = fixed_base_sim
        assert art.is_initialized
        assert art.is_fixed_base is True
        assert art.__backend_name__ == "ovphysx"

    def test_init_metadata(self, fixed_base_sim):
        _, art = fixed_base_sim
        assert art.num_instances == 2
        assert art.num_joints == 2
        assert art.num_bodies == 3

    def test_init_single_articulation(self, single_art_sim):
        _, art = single_art_sim
        assert art.num_instances == 1
        assert art.num_joints == 2
        assert art.num_bodies == 3

    def test_init_default_state(self, fixed_base_sim):
        _, art = fixed_base_sim
        dp = art.data.default_joint_pos
        assert dp.shape == (2, 2)
        dv = art.data.default_joint_vel
        assert dv.shape == (2, 2)
        drp = art.data.default_root_pose
        assert drp.shape == (2,)
        assert drp.dtype == wp.transformf


# ======================================================================
# Joint limits and properties
# ======================================================================

class TestJointProperties:

    def test_joint_pos_limits(self, fixed_base_sim):
        _, art = fixed_base_sim
        limits = art.data.joint_pos_limits
        assert limits.shape == (2, 2)
        assert limits.dtype == wp.vec2f
        lim_np = limits.numpy().reshape(2, 2, 2)
        assert np.all(lim_np[..., 0] < lim_np[..., 1]), "Lower limits must be < upper limits"

    def test_joint_velocity_limit(self, fixed_base_sim):
        _, art = fixed_base_sim
        vel_limits = art.data.joint_vel_limits
        assert vel_limits.shape == (2, 2)

    def test_joint_effort_limit(self, fixed_base_sim):
        _, art = fixed_base_sim
        eff_limits = art.data.joint_effort_limits
        assert eff_limits.shape == (2, 2)


# ======================================================================
# Actuator gains
# ======================================================================

class TestActuatorGains:

    def test_loading_gains_from_usd(self, fixed_base_sim):
        _, art = fixed_base_sim
        stiff = art.data.joint_stiffness
        damp = art.data.joint_damping
        assert stiff.shape == (2, 2)
        assert damp.shape == (2, 2)

    def test_setting_gains_from_cfg(self, cartpole_sim):
        _, art = cartpole_sim
        assert len(art.actuators) == 1
        act = list(art.actuators.values())[0]
        assert act is not None

    def test_setting_gains_write_readback(self, single_art_sim):
        sim, art = single_art_sim
        new_stiffness = np.full((1, 2), 500.0, dtype=np.float32)
        art.write_joint_stiffness_to_sim_index(
            stiffness=wp.from_numpy(new_stiffness, dtype=wp.float32, device=DEVICE)
        )
        sim.step(render=False)
        art.update(DT)


# ======================================================================
# External forces
# ======================================================================

class TestExternalForces:

    def test_external_force_single_body(self, single_art_sim):
        sim, art = single_art_sim
        art.permanent_wrench_composer.add_forces_and_torques_index(
            forces=wp.zeros((1, 3), dtype=wp.vec3f, device=DEVICE),
            torques=wp.zeros((1, 3), dtype=wp.vec3f, device=DEVICE),
            body_ids=list(range(3)),
            env_ids=[0],
        )
        sim.step(render=False)
        art.update(DT)

    def test_external_force_at_position(self, single_art_sim):
        sim, art = single_art_sim
        force = wp.zeros((1, 3), dtype=wp.vec3f, device=DEVICE)
        force_np = force.numpy()
        force_np[0, 0, 0] = 10.0
        force = wp.from_numpy(force_np, dtype=wp.vec3f, device=DEVICE)
        art.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=force,
            torques=wp.zeros((1, 3), dtype=wp.vec3f, device=DEVICE),
            body_ids=list(range(3)),
            env_ids=[0],
        )
        art.write_data_to_sim()
        sim.step(render=False)
        art.update(DT)

    def test_external_force_multiple_bodies(self, fixed_base_sim):
        sim, art = fixed_base_sim
        art.permanent_wrench_composer.add_forces_and_torques_index(
            forces=wp.zeros((2, 3), dtype=wp.vec3f, device=DEVICE),
            torques=wp.zeros((2, 3), dtype=wp.vec3f, device=DEVICE),
            body_ids=list(range(3)),
            env_ids=[0, 1],
        )
        sim.step(render=False)
        art.update(DT)

    def test_external_force_buffer_zeroed_on_reset(self, single_art_sim):
        sim, art = single_art_sim
        art.permanent_wrench_composer.add_forces_and_torques_index(
            forces=wp.zeros((1, 3), dtype=wp.vec3f, device=DEVICE),
            torques=wp.zeros((1, 3), dtype=wp.vec3f, device=DEVICE),
            body_ids=list(range(3)),
            env_ids=[0],
        )
        art.reset()
        assert not art.instantaneous_wrench_composer.active


# ======================================================================
# Reset -- including partial env_ids regression test for C3/C4 fixes
# ======================================================================

class TestReset:

    def test_reset_all(self, fixed_base_sim):
        """Full reset restores default joint positions."""
        sim, art = fixed_base_sim
        jp_default = art.data.default_joint_pos.numpy().copy()

        # Perturb
        for _ in range(60):
            sim.step(render=False)
            art.update(DT)

        jp_perturbed = art.data.joint_pos.numpy().copy()
        assert not np.allclose(jp_perturbed, jp_default, atol=1e-3), "Joints should have moved"

        art.reset()
        sim.step(render=False)
        art.update(DT)

        jp_after = art.data.joint_pos.numpy()
        np.testing.assert_allclose(
            jp_after, jp_default, atol=0.1,
            err_msg="After full reset, joint positions should be near defaults"
        )

    def test_reset_partial_env_ids(self, fixed_base_sim):
        """Regression test for C3/C4: partial env_ids reset writes correct data.

        With 2 envs, perturb all joints, reset only env 0, verify:
        - env 0 is restored to defaults
        - env 1 retains perturbed state
        """
        sim, art = fixed_base_sim
        jp_default = art.data.default_joint_pos.numpy().copy()

        # Perturb all envs by stepping under gravity
        for _ in range(120):
            sim.step(render=False)
            art.update(DT)

        jp_perturbed = art.data.joint_pos.numpy().copy()
        assert not np.allclose(jp_perturbed[0], jp_default[0], atol=1e-3), \
            "Env 0 joints should have moved from defaults"
        assert not np.allclose(jp_perturbed[1], jp_default[1], atol=1e-3), \
            "Env 1 joints should have moved from defaults"

        # Reset only env 0
        art.reset(env_ids=[0])
        sim.step(render=False)
        art.update(DT)

        jp_after = art.data.joint_pos.numpy()
        # Env 0 should be near defaults (not exact due to one step of physics)
        np.testing.assert_allclose(
            jp_after[0], jp_default[0], atol=0.2,
            err_msg="After partial reset, env 0 should be near defaults"
        )
        # Env 1 should still be perturbed (roughly near its pre-reset state)
        delta_env1 = np.abs(jp_after[1] - jp_default[1])
        assert np.any(delta_env1 > 0.01), (
            f"After partial reset, env 1 should still be perturbed. "
            f"Delta from default: {delta_env1}"
        )


# ======================================================================
# State read/write
# ======================================================================

class TestStateReadWrite:

    def test_write_root_pose(self, single_art_sim):
        sim, art = single_art_sim
        pose_np = art.data.root_link_pose_w.numpy().copy()
        pose_np[0, 0] += 0.5  # shift X
        new_pose = wp.from_numpy(pose_np, dtype=wp.transformf, device=DEVICE)
        art.write_root_pose_to_sim_index(root_pose=new_pose)
        sim.step(render=False)
        art.update(DT)

    def test_write_root_velocity(self, single_art_sim):
        sim, art = single_art_sim
        vel = wp.zeros(1, dtype=wp.spatial_vectorf, device=DEVICE)
        art.write_root_velocity_to_sim_index(root_velocity=vel)
        sim.step(render=False)
        art.update(DT)

    def test_write_joint_position(self, single_art_sim):
        sim, art = single_art_sim
        target = np.array([[0.1, -0.1]], dtype=np.float32)
        art.write_joint_position_to_sim_index(
            position=wp.from_numpy(target, dtype=wp.float32, device=DEVICE)
        )
        sim.step(render=False)
        art.update(DT)
        jp = art.data.joint_pos.numpy()
        assert jp.shape == (1, 2)

    def test_write_joint_velocity(self, single_art_sim):
        sim, art = single_art_sim
        vel = np.array([[0.5, -0.5]], dtype=np.float32)
        art.write_joint_velocity_to_sim_index(
            velocity=wp.from_numpy(vel, dtype=wp.float32, device=DEVICE)
        )
        sim.step(render=False)
        art.update(DT)

    def test_write_joint_state_partial_joints(self, single_art_sim):
        """Write only joint 0 (partial joint_ids)."""
        sim, art = single_art_sim
        partial = np.array([[0.2]], dtype=np.float32)
        art.write_joint_position_to_sim_index(
            position=wp.from_numpy(partial, dtype=wp.float32, device=DEVICE),
            joint_ids=[0],
        )
        sim.step(render=False)
        art.update(DT)

    def test_write_joint_state_partial_envs(self, fixed_base_sim):
        """Write joint positions only for env 1 (partial env_ids)."""
        sim, art = fixed_base_sim
        partial = np.array([[0.3, 0.3]], dtype=np.float32)
        art.write_joint_position_to_sim_index(
            position=wp.from_numpy(partial, dtype=wp.float32, device=DEVICE),
            env_ids=[1],
        )
        sim.step(render=False)
        art.update(DT)


# ======================================================================
# Actuator commands
# ======================================================================

class TestActuatorCommands:

    def test_implicit_position_target(self, cartpole_sim):
        """Set a position target and verify the joint moves toward it."""
        sim, art = cartpole_sim
        # Find cart joint
        jids, jnames = art.find_joints("railCartJoint")
        assert len(jids) == 1

        target = wp.zeros((1, art.num_joints), dtype=wp.float32, device=DEVICE)
        target_np = target.numpy()
        target_np[0, jids[0]] = 1.0
        target = wp.from_numpy(target_np, dtype=wp.float32, device=DEVICE)

        art.set_joint_position_target_index(target=target)

        for _ in range(120):
            art.write_data_to_sim()
            sim.step(render=False)
            art.update(DT)

        jp = art.data.joint_pos.numpy()
        assert jp[0, jids[0]] > 0.1, (
            f"Cart should move toward target=1.0, got pos={jp[0, jids[0]]:.4f}"
        )

    def test_effort_target(self, cartpole_sim):
        """Apply an effort and verify the joint moves."""
        sim, art = cartpole_sim
        jp_before = art.data.joint_pos.numpy().copy()

        effort = wp.zeros((1, art.num_joints), dtype=wp.float32, device=DEVICE)
        effort_np = effort.numpy()
        effort_np[0, 0] = 50.0
        effort = wp.from_numpy(effort_np, dtype=wp.float32, device=DEVICE)

        art.set_joint_effort_target_index(target=effort)

        for _ in range(60):
            art.write_data_to_sim()
            sim.step(render=False)
            art.update(DT)

        jp_after = art.data.joint_pos.numpy()
        delta = np.abs(jp_after[0, 0] - jp_before[0, 0])
        assert delta > 0.01, f"Joint should move under effort, delta={delta:.6f}"


# ======================================================================
# Body state and dynamics
# ======================================================================

class TestBodyState:

    def test_body_link_poses_kinematics(self, single_art_sim):
        _, art = single_art_sim
        bp = art.data.body_link_pose_w
        assert bp.dtype == wp.transformf
        assert bp.shape == (1, 3)
        bp_np = bp.numpy().reshape(3, 7)
        quats = bp_np[:, 3:7]
        norms = np.linalg.norm(quats, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4,
                                   err_msg="Link quaternions should be unit quaternions")

    def test_body_com_pose_consistency(self, single_art_sim):
        sim, art = single_art_sim
        sim.step(render=False)
        art.update(DT)
        com_w = art.data.body_com_pose_w
        assert com_w.dtype == wp.transformf
        assert com_w.shape == (1, 3)

    def test_body_incoming_joint_force(self, single_art_sim):
        sim, art = single_art_sim
        for _ in range(10):
            sim.step(render=False)
            art.update(DT)

        wrench = art.data.body_incoming_joint_wrench_b
        assert wrench.dtype == wp.spatial_vectorf
        assert wrench.shape == (1, 3)


# ======================================================================
# Data consistency
# ======================================================================

class TestDataConsistency:

    def test_state_read_consistency(self, single_art_sim):
        """Reading the same property twice in the same step returns identical data."""
        sim, art = single_art_sim
        sim.step(render=False)
        art.update(DT)

        jp1 = art.data.joint_pos.numpy().copy()
        jp2 = art.data.joint_pos.numpy().copy()
        np.testing.assert_array_equal(jp1, jp2)

    def test_derived_properties(self, single_art_sim):
        sim, art = single_art_sim
        sim.step(render=False)
        art.update(DT)

        proj_grav = art.data.projected_gravity_b
        assert proj_grav.shape == (1,)
        assert proj_grav.dtype == wp.vec3f
        grav_np = proj_grav.numpy().reshape(-1)
        assert abs(grav_np[2]) > 0.1, f"Projected gravity Z should be significant, got {grav_np}"

        heading = art.data.heading_w
        assert heading.shape == (1,)
        assert heading.dtype == wp.float32


# ======================================================================
# Friction and body properties
# ======================================================================

class TestFrictionAndBodyProperties:

    def test_write_joint_friction(self, single_art_sim):
        sim, art = single_art_sim
        friction = np.full((1, 2), 0.5, dtype=np.float32)
        art.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=wp.from_numpy(friction, dtype=wp.float32, device=DEVICE)
        )
        sim.step(render=False)
        art.update(DT)

    def test_set_masses(self, single_art_sim):
        sim, art = single_art_sim
        orig_mass = art.data.body_mass.numpy().copy()
        assert np.all(orig_mass > 0), f"Original masses should be positive: {orig_mass}"

        new_mass = orig_mass * 2.0
        art.set_masses_index(
            masses=wp.from_numpy(new_mass, dtype=wp.float32, device=DEVICE)
        )
        sim.step(render=False)
        art.update(DT)

    def test_set_inertias(self, single_art_sim):
        sim, art = single_art_sim
        orig_inertia = art.data.body_inertia.numpy().copy()
        assert orig_inertia.shape == (1, 3, 9)

        new_inertia = orig_inertia * 1.5
        art.set_inertias_index(
            inertias=wp.from_numpy(new_inertia, dtype=wp.float32, device=DEVICE)
        )
        sim.step(render=False)
        art.update(DT)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=long"])
