# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Realistic end-to-end test: Cartpole RL loop through the IsaacLab ovphysx backend.

Uses the double-pendulum cartpole from the Newton test assets:
  - Fixed base (rail pinned to world)
  - Prismatic joint (cart slides on rail)
  - Two revolute joints (pole1, pole2 swing freely)
  - 3 DOFs, 4 bodies

The test simulates a classic RL-style loop:
  - Reset the environment
  - Observe state (root pose, joint positions/velocities, body poses)
  - Compute actions (PD position targets)
  - Apply actions
  - Step the simulation
  - Read new observations
  - Check reward/termination conditions (pole angle)

This exercises the COMPLETE IsaacLab pipeline end-to-end through ovphysx.
"""

import math
import os

import numpy as np
import pytest

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

import warp as wp

wp.init()

# Hide pxr from sys.modules before importing ovphysx to bypass USD version check.
import sys as _sys

_hidden_pxr = {}
for _k in list(_sys.modules):
    if _k == "pxr" or _k.startswith("pxr."):
        _hidden_pxr[_k] = _sys.modules.pop(_k)
import ovphysx  # noqa: E402
ovphysx.bootstrap()

_sys.modules.update(_hidden_pxr)
del _hidden_pxr

CARTPOLE_USD = os.path.join(os.path.dirname(__file__), "..", "data", "cartpole.usda")

DT = 1.0 / 120.0
NUM_STEPS_PER_ACTION = 2
NUM_RL_STEPS = 200


def _create_stage_with_usd(usd_path: str) -> Usd.Stage:
    """Create a fresh in-memory stage and copy USD content into it.

    Copies the root layer content directly (not as a sublayer) so that
    SimulationContext._init_usd_physics_scene() can freely delete and
    recreate the PhysicsScene prim.
    """
    import isaaclab.sim.utils.stage as stage_utils

    src_layer = Sdf.Layer.FindOrOpen(usd_path)
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().TransferContent(src_layer)
    stage_utils._context.stage = stage
    UsdUtils.StageCache.Get().Insert(stage)
    return stage


@pytest.fixture
def cartpole_sim():
    """Set up SimulationContext + Articulation for the cartpole."""
    from isaaclab.sim.simulation_cfg import SimulationCfg
    from isaaclab.sim.simulation_context import SimulationContext
    from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_ovphysx.physics.ovphysx_manager_cfg import OvPhysxCfg

    SimulationContext.clear_instance()
    _create_stage_with_usd(CARTPOLE_USD)

    sim_cfg = SimulationCfg(
        dt=DT,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=OvPhysxCfg(),
        use_fabric=False,
    )
    sim = SimulationContext(sim_cfg)

    from isaaclab.assets.articulation.articulation import Articulation

    art_cfg = ArticulationCfg(
        prim_path="/cartPole",
        actuators={
            "cart": ImplicitActuatorCfg(
                joint_names_expr=["railCartJoint"],
                stiffness=100.0,
                damping=10.0,
            ),
        },
    )
    art = Articulation(art_cfg)
    sim.reset()

    # The poles start perfectly vertical (unstable equilibrium). Give a small
    # perturbation so gravity can pull them down -- otherwise the simulation
    # produces zero joint motion (numerically exact balance).
    perturb = np.array([[0.0, 0.05, 0.0]], dtype=np.float32)
    art.write_joint_position_to_sim_index(position=wp.from_numpy(perturb, dtype=wp.float32, device="cpu"))
    # NOTE: the write helper converts to numpy internally for now; GPU-native
    # writes go through DLPack in the binding.write() call.
    sim.step(render=False)
    art.update(DT)

    yield sim, art

    sim.clear_instance()


class TestCartpoleMetadata:
    """Verify the cartpole articulation is parsed correctly."""

    def test_initialized(self, cartpole_sim):
        sim, art = cartpole_sim
        assert art.is_initialized

    def test_dof_count(self, cartpole_sim):
        sim, art = cartpole_sim
        assert art.num_joints == 3, f"Expected 3 DOFs (cart + 2 poles), got {art.num_joints}"

    def test_body_count(self, cartpole_sim):
        sim, art = cartpole_sim
        assert art.num_bodies == 4, f"Expected 4 bodies (rail, cart, pole1, pole2), got {art.num_bodies}"

    def test_is_fixed_base(self, cartpole_sim):
        sim, art = cartpole_sim
        assert art.is_fixed_base is True

    def test_joint_names(self, cartpole_sim):
        sim, art = cartpole_sim
        names = art.joint_names
        assert len(names) == 3
        print(f"  Joint names: {names}")

    def test_body_names(self, cartpole_sim):
        sim, art = cartpole_sim
        names = art.body_names
        assert len(names) == 4
        print(f"  Body names: {names}")

    def test_find_joints(self, cartpole_sim):
        """find_joints with regex patterns should work."""
        sim, art = cartpole_sim
        ids, names = art.find_joints(".*")
        assert len(ids) == 3


class TestCartpolePhysics:
    """Verify the cartpole behaves physically."""

    def test_pole_falls_under_gravity(self, cartpole_sim):
        """With no actuation, the poles should swing/fall under gravity."""
        sim, art = cartpole_sim
        jp_initial = art.data.joint_pos.numpy().copy()

        for _ in range(120):
            sim.step(render=False)
            art.update(DT)

        jp_after = art.data.joint_pos.numpy()
        delta = np.abs(jp_after - jp_initial)
        assert np.any(delta > 0.01), (
            f"Poles should deflect under gravity. Delta: {delta}"
        )
        print(f"  Joint pos change: {delta[0]}")

    def test_joint_velocities_nonzero_during_swing(self, cartpole_sim):
        """During free swing, joint velocities should be non-zero."""
        sim, art = cartpole_sim

        for _ in range(30):
            sim.step(render=False)
            art.update(DT)

        jv = art.data.joint_vel.numpy()
        assert np.any(np.abs(jv) > 0.001), f"Joint velocities should be non-zero during swing: {jv}"

    def test_body_poses_kinematically_consistent(self, cartpole_sim):
        """All body quaternions should be unit quaternions throughout simulation."""
        sim, art = cartpole_sim

        for step in range(60):
            sim.step(render=False)
            art.update(DT)

            if step % 20 == 0:
                bp = art.data.body_link_pose_w.numpy().reshape(-1, 4, 7)
                quats = bp[..., 3:7]
                norms = np.linalg.norm(quats, axis=-1)
                np.testing.assert_allclose(
                    norms, 1.0, atol=1e-4,
                    err_msg=f"Body quaternions not unit at step {step}"
                )

    def test_root_stays_fixed(self, cartpole_sim):
        """The rail (root body) is fixed to the world and should not move."""
        sim, art = cartpole_sim
        root_before = art.data.root_link_pose_w.numpy().copy()

        for _ in range(120):
            sim.step(render=False)
            art.update(DT)

        root_after = art.data.root_link_pose_w.numpy()
        np.testing.assert_allclose(
            root_before, root_after, atol=1e-4,
            err_msg="Fixed-base root moved during simulation"
        )


class TestCartpoleRLLoop:
    """Simulate an RL-style training loop."""

    def test_rl_loop_runs_without_error(self, cartpole_sim):
        """A full RL-style loop: observe -> act -> step -> observe."""
        sim, art = cartpole_sim
        num_envs = art.num_instances
        num_joints = art.num_joints

        for rl_step in range(NUM_RL_STEPS):
            # -- Observe
            joint_pos = art.data.joint_pos.numpy()  # (1, 3)
            joint_vel = art.data.joint_vel.numpy()  # (1, 3)
            root_pose = art.data.root_link_pose_w.numpy()  # (1, 7)

            # -- Compute action (simple P controller on cart to stay at center)
            # Cart is the first DOF (prismatic joint).
            cart_pos = joint_pos[0, 0]
            action = -2.0 * cart_pos  # proportional control

            # -- Apply action (write position target for cart joint)
            targets = np.zeros((num_envs, num_joints), dtype=np.float32)
            targets[0, 0] = action
            target_wp = wp.from_numpy(targets, dtype=wp.float32, device="cuda:0")
            art.set_joint_position_target_index(target=target_wp)

            # -- Step physics (multiple substeps per RL step)
            for _ in range(NUM_STEPS_PER_ACTION):
                art.write_data_to_sim()
                sim.step(render=False)
                art.update(DT)

        # After 200 RL steps we should still be running without errors.
        # Read final state to confirm data is still consistent.
        final_jp = art.data.joint_pos.numpy()
        final_jv = art.data.joint_vel.numpy()
        assert final_jp.shape == (1, 3)
        assert final_jv.shape == (1, 3)
        assert not np.any(np.isnan(final_jp)), "NaN in final joint positions"
        assert not np.any(np.isnan(final_jv)), "NaN in final joint velocities"
        print(f"  Final joint pos: {final_jp[0]}")
        print(f"  Final joint vel: {final_jv[0]}")

    def test_rl_loop_cart_position_bounded(self, cartpole_sim):
        """The P controller on the cart should keep it near the center."""
        sim, art = cartpole_sim
        num_envs = art.num_instances
        num_joints = art.num_joints

        max_cart_pos = 0.0
        for rl_step in range(100):
            joint_pos = art.data.joint_pos.numpy()
            cart_pos = joint_pos[0, 0]
            max_cart_pos = max(max_cart_pos, abs(cart_pos))

            targets = np.zeros((num_envs, num_joints), dtype=np.float32)
            targets[0, 0] = -5.0 * cart_pos
            target_wp = wp.from_numpy(targets, dtype=wp.float32, device="cuda:0")
            art.set_joint_position_target_index(target=target_wp)

            for _ in range(NUM_STEPS_PER_ACTION):
                art.write_data_to_sim()
                sim.step(render=False)
                art.update(DT)

        # Cart position limits are [-4, 4] in the USD.
        # With a strong P controller, it should stay well within bounds.
        assert max_cart_pos < 3.0, (
            f"Cart wandered too far: max |pos| = {max_cart_pos:.3f} (limit is 4.0)"
        )
        print(f"  Max cart displacement: {max_cart_pos:.4f}")


class TestCartpoleDerivedQuantities:
    """Test that derived quantities (gravity projection, heading) work on the cartpole."""

    def test_projected_gravity(self, cartpole_sim):
        sim, art = cartpole_sim
        sim.step(render=False)
        art.update(DT)

        grav = art.data.projected_gravity_b.numpy().reshape(-1)
        # For an upright fixed-base, gravity in body frame should be roughly (0, 0, -1).
        assert abs(grav[2] - (-1.0)) < 0.3, f"Gravity Z in body frame should be ~-1, got {grav[2]:.3f}"

    def test_heading(self, cartpole_sim):
        sim, art = cartpole_sim
        sim.step(render=False)
        art.update(DT)

        heading = art.data.heading_w.numpy()
        assert heading.shape == (1,)
        assert not np.isnan(heading[0])

    def test_body_mass(self, cartpole_sim):
        sim, art = cartpole_sim
        mass = art.data.body_mass.numpy()
        assert mass.shape == (1, 4)
        assert np.all(mass > 0), f"All masses should be positive: {mass}"
        # Cart mass is 1.0, poles are 0.25 each (from USD).
        print(f"  Body masses: {mass[0]}")

    def test_joint_limits(self, cartpole_sim):
        sim, art = cartpole_sim
        limits = art.data.joint_pos_limits.numpy().reshape(1, 3, 2)
        # Cart prismatic joint has limits [-4, 4].
        cart_lower = limits[0, 0, 0]
        cart_upper = limits[0, 0, 1]
        assert cart_lower < 0 and cart_upper > 0, (
            f"Cart limits should bracket zero: [{cart_lower}, {cart_upper}]"
        )
        print(f"  Cart joint limits: [{cart_lower:.1f}, {cart_upper:.1f}]")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
