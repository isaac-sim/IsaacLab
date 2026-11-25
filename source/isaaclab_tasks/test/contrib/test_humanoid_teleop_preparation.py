# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend preparation of the humanoid teleoperation tasks, through registered construction and config copies.

The public physics selectors for these tasks are covered in ``test/core/test_preset_cli.py``.
"""

from __future__ import annotations

from copy import deepcopy

import pytest


@pytest.mark.parametrize(
    "task_name",
    [
        "IsaacContrib-PickPlace-GR1T2-Abs",
        "IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs",
        "IsaacContrib-PickPlace-Locomanipulation-G1-Abs",
        "IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs",
    ],
)
@pytest.mark.parametrize("physics", [None, "isaacsim_physx", "ovphysx", "newton_mjwarp"])
def test_registered_construction_prepares_only_standalone_backends(monkeypatch, task_name, physics):
    """Registered humanoid construction prepares a standalone backend and leaves Kit physics unprepared."""
    import gymnasium as gym

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils import instantiate

    import isaaclab_tasks  # noqa: F401 -- registers the public task IDs
    from isaaclab_tasks.utils import parse_env_cfg

    overrides = [] if physics is None else [f"physics={physics}"]
    cfg = parse_env_cfg(task_name, device="cpu", num_envs=1, overrides=overrides)

    class ConstructionReached(Exception):
        pass

    def inspect_before_simulation(self, cfg, *args, **kwargs):
        self._is_closed = True
        selected_physics = "ovphysx" if physics is None else physics
        expected = selected_physics if selected_physics in ("ovphysx", "newton_mjwarp") else None
        assert cfg._prepared_physics_backend == expected
        raise ConstructionReached

    monkeypatch.setattr(ManagerBasedRLEnv, "__init__", inspect_before_simulation)
    # Each constructor receives its own unprepared copy, so each must prepare it.
    for construct in (lambda env_cfg: gym.make(task_name, cfg=env_cfg), instantiate):
        with pytest.raises(ConstructionReached):
            construct(deepcopy(cfg))


@pytest.mark.parametrize(
    "task_name",
    [
        "IsaacContrib-PickPlace-GR1T2-Abs",
        "IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs",
        "IsaacContrib-PickPlace-Locomanipulation-G1-Abs",
        "IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs",
    ],
)
@pytest.mark.parametrize("physics", ["ovphysx", "newton_mjwarp"])
def test_humanoid_preparation_preserves_caller_changes_and_rejects_backend_switch(monkeypatch, task_name, physics):
    """Teleop preparation, copies and registered construction share one completed preparation."""
    import gymnasium as gym
    from isaaclab_isaacsim.physics import PhysxCfg

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.string import string_to_callable

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(task_name, device="cpu", num_envs=1, overrides=[f"physics={physics}"])
    controlled_joints = list(cfg.actions.upper_body_ik.pink_controlled_joint_names)
    string_to_callable(cfg.backend_preparation[physics])(cfg)
    cfg.isaac_teleop.require_initial_bimanual_hand_tracking = (
        not cfg.isaac_teleop.require_initial_bimanual_hand_tracking
    )
    policy = {
        name: getattr(cfg.isaac_teleop, name)
        for name in ("require_initial_bimanual_hand_tracking", "action_conditioner", "action_validator")
    }
    haptics = cfg.haptic_feedback.to_dict()
    images = getattr(cfg, "image_obs_list", None)
    if images is not None:
        images.append("caller_camera")
        images = list(images)
    cfg.sim.dt = 0.003
    actuator_name = next(iter(cfg.scene.robot.actuators))
    cfg.scene.robot.actuators[actuator_name].stiffness = 123.0
    if physics == "newton_mjwarp":
        cfg.sim.physics.solver_cfg.nconmax = 5123
    if "GR1T2" in task_name:
        cfg.actions.upper_body_ik.controller.urdf_path = "caller-selected.urdf"

    class ConstructionReached(Exception):
        pass

    def inspect_before_simulation(self, cfg, *args, **kwargs):
        self._is_closed = True
        assert cfg.actions.upper_body_ik.pink_controlled_joint_names == controlled_joints
        assert all(getattr(cfg.isaac_teleop, name) == value for name, value in policy.items())
        assert cfg.haptic_feedback.to_dict() == haptics
        assert getattr(cfg, "image_obs_list", None) == images
        assert cfg.sim.dt == 0.003
        assert cfg.scene.robot.actuators[actuator_name].stiffness == 123.0
        if physics == "newton_mjwarp":
            assert cfg.sim.physics.solver_cfg.nconmax == 5123
        if "GR1T2" in task_name:
            assert cfg.actions.upper_body_ik.controller.urdf_path == "caller-selected.urdf"
        raise ConstructionReached

    monkeypatch.setattr(ManagerBasedRLEnv, "__init__", inspect_before_simulation)
    for prepared in (cfg, cfg.copy()):
        string_to_callable(prepared.backend_preparation[physics])(prepared)
        serialized = prepared.to_dict()
        assert serialized["haptic_feedback"] == haptics
        if "GR1T2" in task_name:
            assert serialized["scene"]["robot"]["spawn"]["func"].endswith(":_spawn_and_condition")
        with pytest.raises(ConstructionReached):
            gym.make(task_name, cfg=prepared)
        prepared.sim.physics = PhysxCfg()
        with pytest.raises(ValueError, match="resolve a fresh task configuration"):
            gym.make(task_name, cfg=prepared)


def test_humanoid_preparation_resolves_directly_built_configs():
    """Direct construction resolves preset defaults and launches with compatible RTX rendering."""
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_ov.renderers import OVRTXRendererCfg

    from isaaclab.app import validate_launch

    from isaaclab_tasks.contrib.locomanip_pick_place.fixed_base_upper_body_ik_g1_env_cfg import (
        FixedBaseUpperBodyIKG1EnvCfg,
    )
    from isaaclab_tasks.contrib.teleop_physics_cfg import prepare_humanoid_teleop_environment

    cfg = FixedBaseUpperBodyIKG1EnvCfg()
    prepare_humanoid_teleop_environment(cfg)
    assert type(cfg.sim.physics) is OvPhysxCfg
    validate_launch(cfg, {"visualizer": []})
    assert type(cfg.scene.robot_pov_cam.renderer_cfg) is OVRTXRendererCfg


@pytest.mark.parametrize("kit", [False, True])
def test_humanoid_preparation_resolves_the_physx_preset_like_the_simulation(monkeypatch, kit):
    """``physics=physx`` prepares OvPhysX without Kit, as SimulationContext would later resolve it."""
    from isaaclab_isaacsim.physics import PhysxCfg
    from isaaclab_ov.physics import OvPhysxCfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.contrib import teleop_physics_cfg
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(
        "IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs", device="cpu", num_envs=1, overrides=["physics=physx"]
    )
    monkeypatch.setattr(teleop_physics_cfg, "has_kit", lambda: kit)
    teleop_physics_cfg.prepare_humanoid_teleop_environment(cfg)
    assert type(cfg.sim.physics) is (PhysxCfg if kit else OvPhysxCfg)
    assert cfg._prepared_physics_backend == (None if kit else "ovphysx")


def test_humanoid_preparation_rejects_an_unsupported_backend():
    """A backend without task preparation fails instead of being treated as Kit PhysX."""
    from isaaclab_newton.physics import NewtonCfg, XPBDSolverCfg

    from isaaclab_tasks.contrib.locomanip_pick_place.fixed_base_upper_body_ik_g1_env_cfg import (
        FixedBaseUpperBodyIKG1EnvCfg,
    )
    from isaaclab_tasks.contrib.teleop_physics_cfg import prepare_humanoid_teleop_environment

    cfg = FixedBaseUpperBodyIKG1EnvCfg()
    cfg.sim.physics = NewtonCfg(solver_cfg=XPBDSolverCfg())
    with pytest.raises(ValueError, match="supports Kit PhysX, OvPhysX and Newton with MJWarp"):
        prepare_humanoid_teleop_environment(cfg)


def _spawn_task_roundtrip_fixture(path, _cfg, **_kwargs):
    """Module-level source spawner, preserving the public callable serialization contract."""
    from pxr import UsdGeom, UsdPhysics

    from isaaclab.sim import get_current_stage

    stage = get_current_stage()
    root = stage.DefinePrim(path, "Xform")
    joint = UsdPhysics.RevoluteJoint.Define(stage, path + "/R_thumb_distal_joint")
    joint.CreateLowerLimitAttr().Set(-15.0)
    joint.CreateUpperLimitAttr().Set(80.0)
    table = stage.DefinePrim(path + "/Table", "Xform")
    UsdPhysics.CollisionAPI.Apply(table)
    UsdPhysics.MeshCollisionAPI.Apply(table).CreateApproximationAttr().Set("boundingCube")
    UsdGeom.Mesh.Define(stage, path + "/Table/Mesh")
    for name in ("sm_steeringwheel_a01_wheel_rim_01", "hub_detail"):
        mesh = UsdGeom.Mesh.Define(stage, path + "/" + name).GetPrim()
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr().Set("convexDecomposition")
    return root


@pytest.mark.parametrize(
    "task_name",
    [
        "IsaacContrib-PickPlace-GR1T2-Abs",
        "IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs",
        "IsaacContrib-PickPlace-Locomanipulation-G1-Abs",
        "IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs",
    ],
)
@pytest.mark.parametrize("physics", ["ovphysx", "newton_mjwarp"])
@pytest.mark.parametrize("restore_target", ["copy", "fresh_prepared"])
def test_prepared_task_spawn_callbacks_survive_public_config_roundtrip(monkeypatch, task_name, physics, restore_target):
    """Restored prepared configurations reach registered construction with resolvable spawners."""
    import gymnasium as gym

    from pxr import Usd, UsdPhysics

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Asset, AssetBaseCfg
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils import to_dict, update_from_dict
    from isaaclab.utils.string import string_to_callable

    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(task_name, device="cpu", num_envs=1, overrides=[f"physics={physics}"])
    for asset in (cfg.scene.robot, cfg.scene.packing_table, cfg.scene.object):
        asset.spawn.func = _spawn_task_roundtrip_fixture
    string_to_callable(cfg.backend_preparation[physics])(cfg)
    cfg.sim.dt = 0.003
    serialized = to_dict(cfg)
    if "GR1T2" in task_name:
        cfg.actions.upper_body_ik.hand_target_margin_rad = 0.123
    if restore_target == "copy":
        restored = cfg.copy()
    else:
        restored = parse_env_cfg(task_name, device="cpu", num_envs=1, overrides=[f"physics={physics}"])
        string_to_callable(restored.backend_preparation[physics])(restored)
    update_from_dict(restored, serialized)
    if "GR1T2" in task_name:
        assert cfg.actions.upper_body_ik.hand_target_margin_rad == 0.123

    class ConstructionReached(Exception):
        pass

    def inspect_before_simulation(self, cfg, *args, **kwargs):
        self._is_closed = True
        assert cfg.sim.dt == 0.003
        assets = [cfg.scene.packing_table]
        if physics == "newton_mjwarp":
            assets.append(cfg.scene.object)
        if "GR1T2" in task_name:
            assets.append(cfg.scene.robot)
        stage = Usd.Stage.CreateInMemory()
        for index, asset in enumerate(assets):
            callback = to_dict(asset.spawn)["func"]
            assert callable(string_to_callable(callback))
            path = f"/World/Asset_{index}"
            with sim_utils.use_stage(stage):
                authored = Asset(AssetBaseCfg(prim_path=path, spawn=asset.spawn))
            root = authored.prim
            assert root == stage.GetPrimAtPath(path)
            if asset is cfg.scene.packing_table:
                assert not stage.GetPrimAtPath(path + "/Table").HasAPI(UsdPhysics.CollisionAPI)
                assert stage.GetPrimAtPath(path + "/Table/Mesh").HasAPI(UsdPhysics.CollisionAPI)
            elif asset is cfg.scene.object:
                assert (
                    stage.GetPrimAtPath(path + "/hub_detail").GetAttribute("physics:approximation").Get()
                    == "convexHull"
                )
                assert (
                    stage.GetPrimAtPath(path + "/sm_steeringwheel_a01_wheel_rim_01")
                    .GetAttribute("physics:approximation")
                    .Get()
                    == "convexDecomposition"
                )
            else:
                limits = authored.cfg.spawn.hand_joint_limits["R_thumb_distal_joint"]
                assert limits == pytest.approx((-0.2617993878, 1.3962634016))
        raise ConstructionReached

    monkeypatch.setattr(sim_utils, "find_matching_prim_paths", lambda path: [path])
    monkeypatch.setattr(ManagerBasedRLEnv, "__init__", inspect_before_simulation)
    with pytest.raises(ConstructionReached):
        gym.make(task_name, cfg=restored)
