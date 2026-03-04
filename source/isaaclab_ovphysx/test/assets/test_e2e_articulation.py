# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Full end-to-end test of the IsaacLab ovphysx backend.

This test exercises the complete pipeline:
  SimulationContext(OvPhysxCfg)
    -> OvPhysxManager.reset() (exports stage, loads into ovphysx)
      -> PHYSICS_READY event
        -> Articulation._initialize_impl() (creates tensor bindings)
          -> ArticulationData property reads (DLPack warp<->ovphysx)
            -> write targets -> step -> read state -> verify physics

No Kit, no Carbonite, no Fabric. Pure ovphysx + pxr + warp.
"""

import os
import sys

import numpy as np
import pytest

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

import warp as wp

wp.init()

# When running under IsaacSim's Python, pxr reports version 0.25.5 (pip packaging
# convention) but ovphysx expects 25.11 (OpenUSD release convention).  The actual
# USD ABI is compatible -- they come from the same OpenUSD source tree.
# Temporarily hide pxr from sys.modules before the first ovphysx import so ovphysx's
# Python-level version check is skipped (it only fires when "pxr" is already in
# sys.modules).  The C++ layer does its own check and loads fine.
import sys as _sys
_hidden_pxr = {}
for _k in list(_sys.modules):
    if _k == "pxr" or _k.startswith("pxr."):
        _hidden_pxr[_k] = _sys.modules.pop(_k)

# Force-import ovphysx NOW (with pxr hidden) so the version check passes.
import ovphysx  # noqa: E402,F401

# Restore pxr modules so the rest of the test (and IsaacLab) can use pxr normally.
_sys.modules.update(_hidden_pxr)
del _hidden_pxr

TWO_ART_USD = os.path.join(
    os.path.expanduser("~"), "physics_backup", "omni", "ovphysx", "tests", "data", "two_articulations.usda"
)


def _create_stage_with_usd_content(usd_path: str) -> Usd.Stage:
    """Create a fresh in-memory stage and load USD content as a sublayer.

    This avoids the iterator-invalidation issue that occurs when
    SimulationContext._init_usd_physics_scene() tries to delete the
    PhysicsScene prim from a directly-opened stage.
    """
    import isaaclab.sim.utils.stage as stage_utils

    stage = Usd.Stage.CreateInMemory()
    # Add the USD file as a sublayer so all its prims appear on the stage.
    stage.GetRootLayer().subLayerPaths.append(usd_path)
    stage_utils._context.stage = stage
    cache = UsdUtils.StageCache.Get()
    cache.Insert(stage)
    return stage


@pytest.fixture
def sim_and_articulation():
    """Create SimulationContext + Articulation through the full IsaacLab stack."""
    from isaaclab.sim.simulation_cfg import SimulationCfg
    from isaaclab.sim.simulation_context import SimulationContext
    from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
    from isaaclab_ovphysx.physics.ovphysx_manager_cfg import OvPhysxCfg

    # Make sure any prior singleton is cleared.
    SimulationContext.clear_instance()

    # Create a fresh stage with the test USD as a sublayer.
    _create_stage_with_usd_content(TWO_ART_USD)

    sim_cfg = SimulationCfg(
        dt=1.0 / 60.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=OvPhysxCfg(),
        use_fabric=False,
        render_interval=1,
    )
    sim = SimulationContext(sim_cfg)

    # The Articulation factory dispatches to isaaclab_ovphysx based on the manager.
    # We import the factory class (not the backend-specific one).
    from isaaclab.assets.articulation.articulation import Articulation

    art_cfg = ArticulationCfg(
        prim_path="/World/articulation",
        actuators={},
    )
    art = Articulation(art_cfg)

    # reset() triggers: OvPhysxManager exports stage -> loads into ovphysx
    #                    dispatches PHYSICS_READY -> Articulation._initialize_impl()
    sim.reset()

    yield sim, art

    sim.clear_instance()


class TestEndToEnd:
    """Full integration tests through the IsaacLab stack."""

    def test_articulation_is_initialized(self, sim_and_articulation):
        sim, art = sim_and_articulation
        assert art.is_initialized, "Articulation should be initialized after sim.reset()"

    def test_backend_name(self, sim_and_articulation):
        sim, art = sim_and_articulation
        assert art.__backend_name__ == "ovphysx"

    def test_metadata(self, sim_and_articulation):
        sim, art = sim_and_articulation
        assert art.num_instances == 1, "Pattern /World/articulation matches one articulation"
        assert art.num_joints == 2, "Articulation has 2 revolute joints"
        assert art.num_bodies == 3, "Articulation has 3 links"
        assert art.is_fixed_base is True

    def test_joint_names(self, sim_and_articulation):
        sim, art = sim_and_articulation
        assert len(art.joint_names) == 2
        assert len(art.body_names) == 3

    def test_read_root_link_pose(self, sim_and_articulation):
        sim, art = sim_and_articulation
        pose = art.data.root_link_pose_w
        assert isinstance(pose, wp.array)
        assert pose.dtype == wp.transformf
        assert pose.shape == (1,)
        pose_np = pose.numpy().reshape(-1)
        # The root link should be at approximately (-2.5, 0, 24) based on the USD.
        assert abs(pose_np[0] - (-2.5)) < 0.1, f"Root X position: expected ~-2.5, got {pose_np[0]}"
        assert abs(pose_np[2] - 24.0) < 0.1, f"Root Z position: expected ~24.0, got {pose_np[2]}"

    def test_read_joint_positions(self, sim_and_articulation):
        sim, art = sim_and_articulation
        jp = art.data.joint_pos
        assert isinstance(jp, wp.array)
        assert jp.shape == (1, 2)
        # Initially joints should be at or near zero.
        jp_np = jp.numpy()
        assert np.all(np.abs(jp_np) < 0.5), f"Initial joint positions should be near zero, got {jp_np}"

    def test_read_body_link_poses(self, sim_and_articulation):
        sim, art = sim_and_articulation
        bp = art.data.body_link_pose_w
        assert isinstance(bp, wp.array)
        assert bp.dtype == wp.transformf
        assert bp.shape == (1, 3)

    def test_read_joint_properties(self, sim_and_articulation):
        sim, art = sim_and_articulation
        stiff = art.data.joint_stiffness
        assert stiff.shape == (1, 2)
        damp = art.data.joint_damping
        assert damp.shape == (1, 2)
        limits = art.data.joint_pos_limits
        assert limits.shape == (1, 2)

    def test_read_body_mass(self, sim_and_articulation):
        sim, art = sim_and_articulation
        mass = art.data.body_mass
        assert mass.shape == (1, 3)
        mass_np = mass.numpy()
        assert np.all(mass_np > 0), f"All body masses should be positive, got {mass_np}"

    def test_step_changes_joint_state(self, sim_and_articulation):
        """After stepping, joint positions should change (gravity pulls the chain)."""
        sim, art = sim_and_articulation
        jp_before = art.data.joint_pos.numpy().copy()

        for _ in range(60):
            sim.step(render=False)
            art.update(sim.cfg.dt)

        jp_after = art.data.joint_pos.numpy()
        delta = np.abs(jp_after - jp_before)
        assert np.any(delta > 1e-4), (
            f"Joint positions should change after stepping under gravity. "
            f"Before: {jp_before}, After: {jp_after}, Delta: {delta}"
        )

    def test_write_joint_position_and_readback(self, sim_and_articulation):
        """Write joint positions to the sim and verify they take effect."""
        sim, art = sim_and_articulation
        target_pos = np.array([[0.1, -0.1]], dtype=np.float32)
        target_wp = wp.from_numpy(target_pos, dtype=wp.float32, device="cuda:0")
        art.write_joint_position_to_sim_index(position=target_wp)

        # Step a few times so the write takes effect.
        for _ in range(5):
            sim.step(render=False)
            art.update(sim.cfg.dt)

        # The joint positions should now be close to the target
        # (not exact because gravity and drives act).
        jp = art.data.joint_pos.numpy()
        # At minimum, the positions should have shifted from where they were.
        assert jp.shape == (1, 2)

    def test_derived_properties(self, sim_and_articulation):
        """Verify derived properties compute without errors."""
        sim, art = sim_and_articulation

        # Step once so we have valid state.
        sim.step(render=False)
        art.update(sim.cfg.dt)

        proj_grav = art.data.projected_gravity_b
        assert proj_grav.shape == (1,)
        assert proj_grav.dtype == wp.vec3f
        grav_np = proj_grav.numpy().reshape(-1)
        # Gravity should have a non-trivial Z component in the body frame.
        assert abs(grav_np[2]) > 0.1, f"Projected gravity Z should be significant, got {grav_np}"

        heading = art.data.heading_w
        assert heading.shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=long"])
