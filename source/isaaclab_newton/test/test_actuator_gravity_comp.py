# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for gravity compensation USD attributes and their propagation to Newton/MJCF.

These tests verify the full pipeline: USD attributes -> Newton ModelBuilder -> MuJoCo solver -> MJCF XML.
They do NOT require Isaac Sim / Omniverse Kit -- only ``pxr`` (OpenUSD), ``newton``, and ``mujoco``.
"""

import unittest.mock as mock
from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch
from newton.solvers import SolverMuJoCo

from pxr import Gf, Usd, UsdGeom, UsdPhysics

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _create_two_body_articulation(stage):
    """Create a minimal two-body articulation on *stage* and return key prims."""
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")

    root = UsdGeom.Xform.Define(stage, "/World/Robot")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    # Body1: rigid body + collision cube
    body1_xf = UsdGeom.Xform.Define(stage, "/World/Robot/Body1")
    body1 = body1_xf.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body1)
    mass1 = UsdPhysics.MassAPI.Apply(body1)
    mass1.GetMassAttr().Set(1.0)
    col1 = UsdGeom.Cube.Define(stage, "/World/Robot/Body1/Collision")
    UsdPhysics.CollisionAPI.Apply(col1.GetPrim())

    # Body2: rigid body + collision sphere
    body2_xf = UsdGeom.Xform.Define(stage, "/World/Robot/Body2")
    body2 = body2_xf.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body2)
    mass2 = UsdPhysics.MassAPI.Apply(body2)
    mass2.GetMassAttr().Set(1.0)
    col2 = UsdGeom.Sphere.Define(stage, "/World/Robot/Body2/Collision")
    UsdPhysics.CollisionAPI.Apply(col2.GetPrim())

    # Joint1: world -> Body1
    joint1 = UsdPhysics.RevoluteJoint.Define(stage, "/World/Robot/Joint1")
    joint1.GetBody0Rel().SetTargets(["/World/Robot/Body1"])
    joint1.GetAxisAttr().Set("Z")
    joint1.GetLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
    joint1.GetLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint1.GetLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint1.GetLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

    # Joint2: Body1 -> Body2
    joint2 = UsdPhysics.RevoluteJoint.Define(stage, "/World/Robot/Joint2")
    joint2.GetBody0Rel().SetTargets(["/World/Robot/Body1"])
    joint2.GetBody1Rel().SetTargets(["/World/Robot/Body2"])
    joint2.GetAxisAttr().Set("Y")
    joint2.GetLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
    joint2.GetLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint2.GetLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint2.GetLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

    return body1, body2, joint1.GetPrim(), joint2.GetPrim()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def _make_mock_env(stage, prim_paths):
    """Build a minimal mock env that satisfies set_gravity_compensation's interface.

    The function accesses:
      - env.sim.is_playing()  -> False (prestartup)
      - env.sim.stage         -> the USD stage
      - env.scene[name]       -> an object with cfg.prim_path
      - env.scene.num_envs    -> len(prim_paths)
    """
    asset = SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/Robot"))
    scene = mock.MagicMock()
    scene.__getitem__ = mock.MagicMock(return_value=asset)
    scene.num_envs = len(prim_paths)
    sim = SimpleNamespace(is_playing=lambda: False, stage=stage)
    return SimpleNamespace(sim=sim, scene=scene)


def _make_asset_cfg(name="robot"):
    return SimpleNamespace(name=name)


@pytest.mark.isaacsim_ci
class TestSetGravityCompensationEvent:
    """Test the set_gravity_compensation MDP event function directly."""

    def test_uniform_body_gravcomp(self):
        """Uniform float scale sets mjc:gravcomp on all rigid bodies."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        body1, body2, _, _ = _create_two_body_articulation(stage)

        env = _make_mock_env(stage, ["/World/Robot"])

        from isaaclab.envs.mdp.events import set_gravity_compensation

        with mock.patch("isaaclab.sim.find_matching_prim_paths", return_value=["/World/Robot"]):
            set_gravity_compensation(env, None, _make_asset_cfg(), body_gravity_compensation_scale=0.75)

        assert body1.GetAttribute("mjc:gravcomp").Get() == pytest.approx(0.75)
        assert body2.GetAttribute("mjc:gravcomp").Get() == pytest.approx(0.75)

    def test_dict_body_gravcomp_with_regex(self):
        """Dict form applies different scales based on body name regex."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        body1, body2, _, _ = _create_two_body_articulation(stage)

        env = _make_mock_env(stage, ["/World/Robot"])

        from isaaclab.envs.mdp.events import set_gravity_compensation

        with mock.patch("isaaclab.sim.find_matching_prim_paths", return_value=["/World/Robot"]):
            set_gravity_compensation(
                env,
                None,
                _make_asset_cfg(),
                body_gravity_compensation_scale={"Body1": 1.0, "Body2": 0.3},
            )

        assert body1.GetAttribute("mjc:gravcomp").Get() == pytest.approx(1.0)
        assert body2.GetAttribute("mjc:gravcomp").Get() == pytest.approx(0.3)

    def test_joint_gravcomp_with_regex(self):
        """Joint regex patterns set mjc:actuatorgravcomp on matching joints only."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        _, _, joint1_prim, joint2_prim = _create_two_body_articulation(stage)

        env = _make_mock_env(stage, ["/World/Robot"])

        from isaaclab.envs.mdp.events import set_gravity_compensation

        with mock.patch("isaaclab.sim.find_matching_prim_paths", return_value=["/World/Robot"]):
            set_gravity_compensation(
                env,
                None,
                _make_asset_cfg(),
                joint_gravity_compensation=["Joint1"],
            )

        assert joint1_prim.GetAttribute("mjc:actuatorgravcomp").Get() is True
        assert not joint2_prim.GetAttribute("mjc:actuatorgravcomp").IsValid()

    def test_env_ids_limits_which_envs_are_modified(self):
        """Only the environments in env_ids should be modified."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        # Create two "environments" with separate robot prims
        for env_idx in range(2):
            prefix = f"/World/Env_{env_idx}/Robot"
            root = UsdGeom.Xform.Define(stage, prefix)
            UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
            body_xf = UsdGeom.Xform.Define(stage, f"{prefix}/Body")
            body = body_xf.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body)
            mass = UsdPhysics.MassAPI.Apply(body)
            mass.GetMassAttr().Set(1.0)

        env = _make_mock_env(stage, ["/World/Env_0/Robot", "/World/Env_1/Robot"])

        from isaaclab.envs.mdp.events import set_gravity_compensation

        # Only modify env 0
        with mock.patch(
            "isaaclab.sim.find_matching_prim_paths", return_value=["/World/Env_0/Robot", "/World/Env_1/Robot"]
        ):
            set_gravity_compensation(
                env,
                torch.tensor([0]),
                _make_asset_cfg(),
                body_gravity_compensation_scale=1.0,
            )

        env0_body = stage.GetPrimAtPath("/World/Env_0/Robot/Body")
        env1_body = stage.GetPrimAtPath("/World/Env_1/Robot/Body")
        assert env0_body.GetAttribute("mjc:gravcomp").Get() == pytest.approx(1.0)
        assert not env1_body.GetAttribute("mjc:gravcomp").IsValid(), "env_id=1 should not be modified"

    def test_raises_when_sim_is_playing(self):
        """Should raise RuntimeError if called after simulation starts."""
        stage = Usd.Stage.CreateInMemory()
        sim = SimpleNamespace(is_playing=lambda: True, stage=stage)
        env = SimpleNamespace(sim=sim, scene=mock.MagicMock())

        from isaaclab.envs.mdp.events import set_gravity_compensation

        with pytest.raises(RuntimeError, match="prestartup"):
            set_gravity_compensation(env, None, _make_asset_cfg(), body_gravity_compensation_scale=1.0)

    def test_noop_when_both_params_none(self):
        """Should return immediately without touching USD when both params are None."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        body1, _, _, _ = _create_two_body_articulation(stage)

        env = _make_mock_env(stage, ["/World/Robot"])

        from isaaclab.envs.mdp.events import set_gravity_compensation

        # Neither param set — should be a no-op
        set_gravity_compensation(env, None, _make_asset_cfg())

        assert not body1.GetAttribute("mjc:gravcomp").IsValid()


def _build_newton_model(stage):
    """Build a Newton model from a USD stage with MuJoCo solver custom attributes."""
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(stage)
    return builder.finalize()


@pytest.mark.isaacsim_ci
class TestGravityCompensationNewtonModel:
    """End-to-end: set_gravity_compensation -> Newton model has correct gravcomp values."""

    def test_body_gravcomp_in_newton_model(self):
        """Body gravcomp written by the event propagates to Newton model.mujoco.gravcomp."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        body1, body2, _, _ = _create_two_body_articulation(stage)

        env = _make_mock_env(stage, ["/World/Robot"])

        from isaaclab.envs.mdp.events import set_gravity_compensation

        with mock.patch("isaaclab.sim.find_matching_prim_paths", return_value=["/World/Robot"]):
            set_gravity_compensation(
                env,
                None,
                _make_asset_cfg(),
                body_gravity_compensation_scale={"Body1": 0.75, "Body2": 0.3},
            )

        model = _build_newton_model(stage)

        assert hasattr(model.mujoco, "gravcomp"), "Newton model missing mujoco.gravcomp"
        gravcomp = model.mujoco.gravcomp.numpy()
        # body_label tells us the index order
        body_idx = {label.split("/")[-1]: i for i, label in enumerate(model.body_label)}
        assert gravcomp[body_idx["Body1"]] == pytest.approx(0.75)
        assert gravcomp[body_idx["Body2"]] == pytest.approx(0.3)

    def test_joint_actgravcomp_in_newton_model(self):
        """Joint actuatorgravcomp written by the event propagates to Newton model.mujoco.jnt_actgravcomp."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        _, _, joint1_prim, joint2_prim = _create_two_body_articulation(stage)

        env = _make_mock_env(stage, ["/World/Robot"])

        from isaaclab.envs.mdp.events import set_gravity_compensation

        with mock.patch("isaaclab.sim.find_matching_prim_paths", return_value=["/World/Robot"]):
            set_gravity_compensation(
                env,
                None,
                _make_asset_cfg(),
                body_gravity_compensation_scale=1.0,
                joint_gravity_compensation=["Joint1"],
            )

        model = _build_newton_model(stage)

        assert hasattr(model.mujoco, "jnt_actgravcomp"), "Newton model missing mujoco.jnt_actgravcomp"
        jnt_actgravcomp = model.mujoco.jnt_actgravcomp.numpy()
        joint_idx = {label.split("/")[-1]: i for i, label in enumerate(model.joint_label)}
        assert jnt_actgravcomp[joint_idx["Joint1"]] is np.True_
        assert jnt_actgravcomp[joint_idx["Joint2"]] is np.False_

    def test_no_gravcomp_in_newton_model_when_not_set(self):
        """Newton model has zero gravcomp when the event is not called."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        _create_two_body_articulation(stage)

        model = _build_newton_model(stage)

        if hasattr(model.mujoco, "gravcomp"):
            assert np.allclose(model.mujoco.gravcomp.numpy(), 0.0)
