# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab.utils.string import resolve_matching_names

from isaaclab_tasks.core.velocity.mdp.symmetry.g1_29dof import compute_symmetric_states
from isaaclab_tasks.utils import load_cfg_from_registry, resolve_presets


@pytest.mark.parametrize("backend", ["newton_mjwarp", "ovphysx"])
@pytest.mark.parametrize("surface,linear,vertical,clip", [("Rough", 2.0, 0.0, 10.0), ("Flat", 1.0, -2.0, None)])
def test_g1_default_tasks_use_validated_29dof_contract(backend, surface, linear, vertical, clip):
    task = f"Isaac-Velocity-{surface}-G1"
    env_cfg = resolve_presets(load_cfg_from_registry(task, "env_cfg_entry_point"), selected=(backend,))
    agent_cfg = resolve_presets(load_cfg_from_registry(task, "rsl_rl_cfg_entry_point"), selected=(backend,))
    assert env_cfg.rewards.track_lin_vel_xy_exp.weight == linear
    assert env_cfg.rewards.track_ang_vel_z_exp.weight == 2.0
    assert env_cfg.rewards.joint_deviation_arms.weight == (-0.1 if surface == "Flat" else -0.8)
    assert env_cfg.rewards.lin_vel_z_l2.weight == vertical
    assert agent_cfg.clip_actions == clip
    assert agent_cfg.max_iterations == (1500 if surface == "Flat" else 6000)
    assert agent_cfg.save_interval == 500
    assert agent_cfg.actor.hidden_dims == [512, 256, 128]
    if surface == "Flat":
        assert agent_cfg.algorithm.symmetry_cfg is None
        assert env_cfg.commands.base_velocity.ranges.lin_vel_y == (-0.5, 0.5)
        assert env_cfg.rewards.feet_air_time_variance.weight == -24.0
        assert env_cfg.rewards.feet_air_time.weight == 1.5
    else:
        assert agent_cfg.algorithm.symmetry_cfg.use_data_augmentation
        assert getattr(env_cfg.rewards, "feet_air_time_variance", None) is None
    assert env_cfg.scene.robot.actuators["hands"].stiffness == 0.0
    assert env_cfg.scene.robot.actuators["hands"].damping == 0.1
    assert env_cfg.scene.height_scanner is not None
    assert env_cfg.observations.policy.height_scan is not None
    assert env_cfg.scene.terrain.terrain_type == ("plane" if surface == "Flat" else "generator")
    if surface == "Flat":
        assert env_cfg.rewards.feet_flight.weight == -2.0
    else:
        assert getattr(env_cfg.rewards, "feet_flight", None) is None


def test_g1_flat_flight_penalty_requires_both_selected_feet_airborne():
    """Single-foot swings and double support remain unpenalized; other bodies are irrelevant."""
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_tasks.core.velocity.mdp import feet_flight

    air_time = torch.tensor([[0.0, 0.5, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.2], [0.2, 0.0, 0.3]])
    sensor = SimpleNamespace(data=SimpleNamespace(current_air_time=SimpleNamespace(torch=air_time)))
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces": sensor}))
    selection = SceneEntityCfg("contact_forces", body_ids=[0, 2])
    torch.testing.assert_close(feet_flight(env, selection), torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_g1_height_terms_use_median_terrain_and_global_warmup():
    """Elevated terrain and a high ray outlier must not change standing clearance."""
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_tasks.core.velocity.mdp import (
        pelvis_below_terrain_clearance_after_warmup,
        pelvis_height_deficit_l2,
    )

    root = torch.tensor([[0.0, 0.0, 1.7], [0.0, 0.0, -0.7]])
    hits = torch.zeros(2, 3, 3)
    hits[:, :, 2] = torch.tensor([[1.0, 1.0, 10.0], [-1.0, -1.0, float("nan")]])
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        common_step_counter=11_999,
        scene={
            "robot": SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=root))),
            "height_scanner": SimpleNamespace(data=SimpleNamespace(ray_hits_w=SimpleNamespace(torch=hits))),
        },
    )
    selection = {"asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("height_scanner")}
    torch.testing.assert_close(
        pelvis_height_deficit_l2(env, target_height=0.686, **selection), torch.tensor([0.0, 0.386**2])
    )
    assert not pelvis_below_terrain_clearance_after_warmup(env, 0.4, 12_000, **selection).any()
    env.common_step_counter = 12_000
    torch.testing.assert_close(
        pelvis_below_terrain_clearance_after_warmup(env, 0.4, 12_000, **selection), torch.tensor([False, True])
    )


def test_g1_symmetry_reflects_velocity_and_height_scan():
    """A left-right reflection flips lateral scan rows and the appropriate vector axes."""
    env = _symmetry_env(list(range(6)), list(range(6)), list(range(6)))
    env.observation_manager.active_terms["policy"] = ["base_lin_vel", "base_ang_vel", "height_scan"]
    env.observation_manager.group_obs_term_dim["policy"] = [(3,), (3,), (187,)]
    scan = torch.arange(187.0).reshape(1, 11, 17)
    obs = torch.cat([torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]), scan.flatten(1)], dim=1)
    augmented, _ = compute_symmetric_states(env, TensorDict({"policy": obs}, batch_size=[1]))
    torch.testing.assert_close(augmented["policy"][1, :6], torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0]))
    torch.testing.assert_close(augmented["policy"][1, 6:].reshape(11, 17), scan[0].flip(0))


def _symmetry_env(action_ids, position_ids, velocity_ids):
    names = [
        "left_knee_joint",
        "left_hip_roll_joint",
        "left_hand_thumb_0_joint",
        "right_knee_joint",
        "right_hip_roll_joint",
        "right_hand_thumb_0_joint",
    ]
    robot = SimpleNamespace(
        joint_names=names,
        data=SimpleNamespace(
            joint_pos_limits=torch.tensor([[[-1.0, 1.0]] * len(names)]),
            default_joint_pos=torch.zeros(1, len(names)),
        ),
        find_joints=lambda keys, preserve_order=False: resolve_matching_names(keys, names, preserve_order),
    )
    action = SimpleNamespace(
        cfg=SimpleNamespace(
            joint_names=[names[i] for i in action_ids],
            preserve_order=True,
        )
    )
    group = SimpleNamespace(
        joint_pos=SimpleNamespace(params={"asset_cfg": SimpleNamespace(name="robot", joint_ids=position_ids)}),
        joint_vel=SimpleNamespace(params={"asset_cfg": SimpleNamespace(name="robot", joint_ids=velocity_ids)}),
        actions=SimpleNamespace(params={}),
    )
    env = SimpleNamespace(
        device="cpu",
        scene={"robot": robot},
        action_manager=SimpleNamespace(active_terms=["joint_pos"], get_term=lambda name: action),
        observation_manager=SimpleNamespace(
            cfg=SimpleNamespace(policy=group),
            active_terms={"policy": ["joint_pos", "joint_vel", "actions"]},
            group_obs_term_dim={"policy": [(len(position_ids) * 2,), (len(velocity_ids),), (len(action_ids),)]},
        ),
    )
    env.unwrapped = env
    return env


def test_g1_symmetry_mirrors_selected_joints_in_each_terms_order():
    """Uncontrolled fingers must not shift the body joints' mirrors, including history."""
    env = _symmetry_env([4, 0, 1, 3], [0, 1, 3, 4], [4, 3, 1, 0])
    obs = TensorDict({"policy": torch.arange(1, 17, dtype=torch.float32).reshape(1, -1)}, batch_size=[1])
    actions = torch.tensor([[17.0, 18.0, 19.0, 20.0]])
    mirrored_obs, mirrored_actions = compute_symmetric_states(env, obs, actions)
    expected = torch.tensor([[3, -4, 1, -2, 7, -8, 5, -6, -11, 12, -9, 10, -15, 16, -13, 14.0]])
    torch.testing.assert_close(mirrored_obs["policy"][:1], obs["policy"])
    torch.testing.assert_close(mirrored_obs["policy"][1:], expected)
    torch.testing.assert_close(mirrored_actions[1:], torch.tensor([[-19.0, 20.0, -17.0, 18.0]]))
    twice_obs, twice_actions = compute_symmetric_states(env, mirrored_obs[1:], mirrored_actions[1:])
    torch.testing.assert_close(twice_obs["policy"][1:], obs["policy"])
    torch.testing.assert_close(twice_actions[1:], actions)


def test_g1_symmetry_preserves_full_joint_action_contract():
    """Full joint policies retain the original left/right swap and axis signs."""
    env = _symmetry_env(list(range(6)), list(range(6)), list(range(6)))
    env.observation_manager.cfg.policy.joint_pos.params = {}
    env.observation_manager.cfg.policy.joint_vel.params = {}
    actions = torch.tensor([[1, 2, 3, 4, 5, 6.0]])
    obs = TensorDict({"policy": actions.repeat(1, 4)}, batch_size=[1])
    augmented_obs, augmented = compute_symmetric_states(env, obs, actions)
    torch.testing.assert_close(augmented, torch.tensor([[1, 2, 3, 4, 5, 6.0], [4, -5, 6, 1, -2, 3.0]]))
    torch.testing.assert_close(augmented_obs["policy"][1:], torch.tensor([[4, -5, 6, 1, -2, 3.0]]).repeat(1, 4))


def test_g1_symmetry_rejects_selection_without_its_counterpart():
    """Mirroring a unilateral action cannot invent a missing actuator."""
    env = _symmetry_env([0], [0], [0])
    with pytest.raises(ValueError, match="closed under reflection"):
        compute_symmetric_states(env, actions=torch.ones(1, 1))


def test_g1_symmetry_validates_each_articulations_default_pose():
    """A map from another environment must not bypass this robot's pose validation."""
    first = _symmetry_env(list(range(6)), list(range(6)), list(range(6)))
    compute_symmetric_states(first, actions=torch.ones(1, 6))
    second = _symmetry_env(list(range(6)), list(range(6)), list(range(6)))
    second.scene["robot"].data.default_joint_pos[0, 0] = 0.2
    with pytest.raises(RuntimeError, match="default pose"):
        compute_symmetric_states(second, actions=torch.ones(1, 6))


def test_g1_sole_plate_spawn_clones_geometry_and_preserves_contact_settings(tmp_path):
    """Every clone gets the trained sole box without rewriting contact offsets."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    import isaaclab.sim as sim_utils

    from isaaclab_assets.robots.unitree import spawn_g1_with_sole_plates

    asset = Usd.Stage.CreateNew(str(tmp_path / "robot.usda"))
    root = UsdGeom.Xform.Define(asset, "/Robot").GetPrim()
    asset.SetDefaultPrim(root)
    for side in ("left", "right"):
        foot = UsdGeom.Xform.Define(asset, f"/Robot/{side}_ankle_roll_link").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(foot)
        sphere = UsdGeom.Sphere.Define(asset, foot.GetPath().AppendPath("collisions/sphere")).GetPrim()
        UsdPhysics.CollisionAPI.Apply(sphere)
        sphere.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.003)
        sphere.CreateAttribute("newton:margin", Sdf.ValueTypeNames.Float).Set(0.004)
    asset.GetRootLayer().Save()

    stage = Usd.Stage.CreateInMemory()
    for index in range(2):
        UsdGeom.Xform.Define(stage, f"/World/env_{index}")
    cfg = sim_utils.UsdFileCfg(usd_path=str(tmp_path / "robot.usda"))
    with sim_utils.use_stage(stage):
        spawn_g1_with_sole_plates("/World/env_[0-1]/Robot", cfg)
        # A second spawn must not add duplicate transform operations or colliders.
        spawn_g1_with_sole_plates("/World/env_[0-1]/Robot", cfg)
    for index in range(2):
        for side in ("left", "right"):
            path = f"/World/env_{index}/Robot/{side}_ankle_roll_link"
            assert not stage.GetPrimAtPath(path + "/collisions").IsActive()
            plate = UsdGeom.Cube(stage.GetPrimAtPath(path + "/foot_plate"))
            assert plate.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
            transform = plate.GetLocalTransformation()
            low = transform.Transform(Gf.Vec3d(-0.5))
            high = transform.Transform(Gf.Vec3d(0.5))
            # Bounds of the independently measured a1 sole used for candidate training [m].
            assert tuple(low) == pytest.approx((-0.06563756, -0.03273462, -0.03442401), abs=1e-8)
            assert tuple(high) == pytest.approx((0.13747166, 0.03273462, -0.01591613), abs=1e-8)
            assert not plate.GetPrim().GetAttribute("physxCollision:restOffset").HasAuthoredValueOpinion()
            assert not plate.GetPrim().GetAttribute("newton:margin").HasAuthoredValueOpinion()
            # Inactive source colliders keep their authored values unchanged.
            stage.GetPrimAtPath(path + "/collisions").SetActive(True)
            sphere = stage.GetPrimAtPath(path + "/collisions/sphere")
            assert sphere.GetAttribute("physxCollision:restOffset").Get() == pytest.approx(0.003)
            assert sphere.GetAttribute("newton:margin").Get() == pytest.approx(0.004)


def test_feet_timing_variance_uses_selected_completed_phases_and_command_gate():
    """Micro-contact timer overwrites cannot corrupt complete phases or leak across resets."""
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg

    from isaaclab_tasks.core.velocity.mdp import feet_air_time_variance

    air = torch.zeros(3, 3)
    sensor = SimpleNamespace(data=SimpleNamespace(current_air_time=SimpleNamespace(torch=air)))
    command = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        step_dt=0.1,
        scene=SimpleNamespace(sensors={"contact_forces": sensor}),
        command_manager=SimpleNamespace(get_command=lambda _: command),
    )
    selection = SceneEntityCfg("contact_forces", body_ids=[0, 2])
    cfg = RewardTermCfg(func=feet_air_time_variance, weight=-1.0, params={"sensor_cfg": selection})
    term = feet_air_time_variance(cfg, env)
    single_selection = SceneEntityCfg("contact_forces", body_ids=[0])
    single = feet_air_time_variance(cfg.replace(params={"sensor_cfg": single_selection}), env)
    # At 0.1 s per sample, completed swings are 0.2/0.4 s and stances are 0.3/0.5 s.
    left = [True, False, False, True, True, True, False, False, True, True, True]
    right = [True, False, False, False, False, True, True, True, True, True, False]
    for i, (left_contact, right_contact) in enumerate(zip(left, right)):
        air[:, 0] = 0.0 if left_contact else 0.01
        air[:, 1] = i  # Unselected body must not affect the reward.
        air[:, 2] = 0.0 if right_contact else 0.01
        result = term(env, "base_velocity", selection)
        torch.testing.assert_close(single(env, "base_velocity", single_selection), torch.zeros(3))
        if i == 0:
            torch.testing.assert_close(result, torch.zeros(3))
        if i == 6:
            torch.testing.assert_close(result, torch.tensor([0.01, 0.01, 0.0]))
    torch.testing.assert_close(result, torch.tensor([0.02, 0.02, 0.0]))
    with torch.inference_mode():
        bounded = term(env, "base_velocity", selection, max_time=0.25)
    torch.testing.assert_close(bounded, torch.tensor([0.000625, 0.000625, 0.0]))
    term.reset([0])
    torch.testing.assert_close(term(env, "base_velocity", selection), torch.tensor([0.0, 0.02, 0.0]))
    term.reset()
    torch.testing.assert_close(term(env, "base_velocity", selection), torch.zeros(3))
