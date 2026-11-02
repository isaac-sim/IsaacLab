# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The apple task's terms follow the evaluated GR00T policy's inputs and outputs."""

import json
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from isaaclab_experimental.cosmos import CosmosTransferProcessorCfg, read_prompt_files

from isaaclab.utils.string import resolve_matching_names, resolve_matching_names_values

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.apple_pick_place import mdp
from isaaclab_tasks.contrib.apple_pick_place.apple_pick_place_g1_dex3_env_cfg import (
    START_STATES_FILE,
    G1ApplePickPlaceCosmosEnvCfg,
    G1ApplePickPlaceEnvCfg,
)
from isaaclab_tasks.utils import parse_env_cfg


def test_apple_task_is_registered():
    spec = gym.spec("IsaacContrib-PickPlace-Apple-G1-Dex3")
    assert spec.kwargs["env_cfg_entry_point"].endswith(":G1ApplePickPlaceEnvPresetCfg")


def test_observations_and_actions_follow_the_policy_contract():
    cfg = G1ApplePickPlaceEnvCfg()
    policy = cfg.observations.policy
    # One term per policy input, not concatenated, so a pipeline can read each by name.
    terms = [name for name in vars(policy) if hasattr(getattr(policy, name), "func")]
    assert terms == ["ego_view", "left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]
    assert not policy.concatenate_terms
    state = [name for term in terms[1:] for name in getattr(policy, term).params["asset_cfg"].joint_names]
    assert len(state) == len(set(state)) == 43
    # The policy's hand outputs list index and middle before the thumb; its hand inputs list the thumb first.
    assert policy.right_hand.params["asset_cfg"].joint_names[3] == "right_hand_index_0_joint"
    assert cfg.actions.joint_pos.joint_names[14] == "left_hand_index_0_joint"
    assert cfg.actions.joint_pos.joint_names == cfg.actions.joint_effort.joint_names
    assert len(cfg.actions.joint_pos.joint_names) == 31 and not any(
        "hip" in n for n in cfg.actions.joint_pos.joint_names
    )
    # 30 Hz policy actions on 240 Hz physics.
    assert cfg.sim.dt * cfg.decimation == pytest.approx(1 / 30)


def test_cosmos_preset_generates_the_head_camera_image_for_whole_episodes():
    cfg = parse_env_cfg("IsaacContrib-PickPlace-Apple-G1-Dex3", overrides=("presets=cosmos",))
    assert isinstance(cfg, G1ApplePickPlaceCosmosEnvCfg)
    cfg.validate()
    camera = cfg.scene.head_camera
    # Cosmos renders depth on its 4:3 canvas; the policy still reads a 640 x 480 rgb image.
    assert (camera.width, camera.height) == (736, 544)
    assert camera.tensor_processors[-1].outputs == ["rgb"]
    assert cfg.observations.policy.ego_view.params["sensor_cfg"].name == "head_camera"
    model = next(p.backend for p in camera.tensor_processors if isinstance(p, CosmosTransferProcessorCfg))
    # One frame per 30 Hz step for a 20 s episode, plus the first frame, in the service's 1 + 4*k budget.
    assert model.max_episode_frames >= 601 and (model.max_episode_frames - 1) % 4 == 0
    # A longer episode gets a larger budget.
    longer = parse_env_cfg(
        "IsaacContrib-PickPlace-Apple-G1-Dex3", overrides=("presets=cosmos", "env.episode_length_s=30.0")
    )
    longer.validate()
    longer_model = next(
        p.backend for p in longer.scene.head_camera.tensor_processors if isinstance(p, CosmosTransferProcessorCfg)
    )
    assert longer_model.max_episode_frames >= 901
    # The prompt describes the real demonstrations' appearance.
    assert "apple" in read_prompt_files([model.prompt_files])[0]
    # The default task keeps the rendered image.
    assert not G1ApplePickPlaceEnvCfg().scene.head_camera.tensor_processors


def test_saved_starts_cover_every_robot_joint_and_reset_each_environment():
    cfg = G1ApplePickPlaceEnvCfg()
    assert cfg.events.reset_to_start_states.params["states_file"] == START_STATES_FILE
    with open(START_STATES_FILE) as f:
        data = json.load(f)
    robot = cfg.scene.robot
    assert len(data["joint_names"]) == len(set(data["joint_names"])) == 43
    # Every joint the policy reads is set by each saved start.
    observed = [
        name
        for term in ("left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand")
        for name in getattr(cfg.observations.policy, term).params["asset_cfg"].joint_names
    ]
    assert sorted(data["joint_names"]) == sorted(observed)
    assert len(data["states"]) == 16
    for state in data["states"]:
        assert len(state["joint_positions"]) == 43
        # Each pose is a position and an xyzw quaternion of unit length.
        for key in ("robot_root_pose", "apple_pose", "plate_pose"):
            assert len(state[key]) == 7
            assert sum(x * x for x in state[key][3:]) == pytest.approx(1.0, abs=1e-4)
    assert robot.init_state.joint_pos  # the default start still applies to anything the saved starts omit


def test_evaluation_setup_is_native_isaac_lab():
    cfg = G1ApplePickPlaceEnvCfg()
    # The legs are not commanded by the policy; gravity compensation holds them up without adding policy actions.
    legs = cfg.actions.leg_gravity.joint_names
    assert len(legs) == 12 and all(("hip" in n or "knee" in n or "ankle" in n) for n in legs)
    assert not set(legs) & set(cfg.actions.joint_pos.joint_names)
    # The head camera is held still, so it is not a child of the robot, and it is placed after the robot's start.
    assert "/Robot/" not in cfg.scene.head_camera.prim_path
    events = [name for name, term in vars(cfg.events).items() if hasattr(term, "func")]
    assert events.index("reset_head_camera") > events.index("reset_to_start_states")
    # The table's colliders are shrink-wrapped, and it does not slide when touched.
    assert cfg.scene.table.spawn.mesh_collision_props.default.shrink_wrap is True
    assert cfg.scene.table.spawn.rigid_props.kinematic_enabled is True
    # Environments share one warehouse; a copy per environment would overlap its neighbors.
    assert not cfg.scene.warehouse.prim_path.startswith("{ENV_REGEX_NS}")
    # The first image after a reset shows the new start.
    assert cfg.num_rerenders_on_reset >= 1


def test_reset_holds_the_start_pose_with_the_joint_targets():
    """No action drives the legs, so the reset must set their PD targets; they start at zero otherwise."""

    class Recorder:
        def __init__(self):
            self.calls = {}

        def __getattr__(self, name):
            return lambda **kwargs: self.calls.setdefault(name, kwargs)

    with open(START_STATES_FILE) as f:
        joint_names = json.load(f)["joint_names"]
    robot = Recorder()
    robot.find_joints = lambda names, preserve_order: (list(range(len(names))), names)
    robot.actuators = SimpleNamespace(target_command=Recorder())
    assets = {"robot": robot, "apple": Recorder(), "plate": Recorder()}
    scene = type("Scene", (), {"env_origins": torch.zeros(2, 3), "__getitem__": lambda _, name: assets[name]})()
    env = SimpleNamespace(device="cpu", num_envs=2, scene=scene)

    mdp.reset_to_start_states(env, torch.tensor([0, 1]), START_STATES_FILE)

    written = robot.calls["write_joint_position_to_sim_index"]["position"]
    targets = robot.actuators.target_command.calls
    assert torch.equal(targets["set_position_index"]["value"], written)
    assert targets["set_position_index"]["joint_ids"] == list(range(len(joint_names)))
    assert not targets["set_velocity_index"]["value"].any() and not targets["set_effort_index"]["value"].any()
    # The knees start bent; a zero target would straighten the legs under the robot.
    assert written[0, joint_names.index("left_knee_joint")] > 0.5


@pytest.mark.parametrize(
    "overrides, physics, renderer",
    (
        ((), "PhysxCfg", "IsaacRtxRendererCfg"),
        (("physics=ovphysx", "renderer=ovrtx"), "OvPhysxCfg", "OVRTXRendererCfg"),
        (("physics=ovphysx", "renderer=newton_renderer"), "OvPhysxCfg", "NewtonWarpRendererCfg"),
        (("physics=ovphysx", "renderer=ovrtx", "presets=cosmos"), "OvPhysxCfg", "OVRTXRendererCfg"),
        (("physics=newton_mjwarp", "renderer=newton_renderer"), "NewtonCfg", "NewtonWarpRendererCfg"),
        (
            ("physics=newton_mjwarp", "renderer=newton_renderer", "presets=cosmos"),
            "NewtonCfg",
            "NewtonWarpRendererCfg",
        ),
    ),
)
def test_backends_are_selected_with_presets(overrides, physics, renderer):
    cfg = parse_env_cfg("IsaacContrib-PickPlace-Apple-G1-Dex3", overrides=overrides)
    assert type(cfg.sim.physics).__name__ == physics
    assert type(cfg.scene.head_camera.renderer_cfg).__name__ == renderer


def test_newton_uses_convex_hulls_and_physx_keeps_the_authored_colliders():
    """Newton would decompose the apple, plate and table meshes on the CPU at startup, for minutes per mesh."""
    physx = parse_env_cfg("IsaacContrib-PickPlace-Apple-G1-Dex3", overrides=("physics=ovphysx",))
    newton = parse_env_cfg("IsaacContrib-PickPlace-Apple-G1-Dex3", overrides=("physics=newton_mjwarp",))
    assert type(newton.sim.physics).__name__ == "NewtonCfg"
    for name in ("apple", "plate"):
        assert getattr(newton.scene, name).spawn.mesh_collision_props.mesh_approximation_name == "convexHull"
    # A hull around the table's legs would fill the space under it.
    assert type(newton.scene.table.spawn.mesh_collision_props).__name__ == "NewtonSDFCollisionCfg"
    assert physx.scene.apple.spawn.mesh_collision_props is None
    assert physx.scene.table.spawn.mesh_collision_props.shrink_wrap is True


def test_every_actuator_sets_armature():
    """Explicit PD on the light wrist and finger links is unstable at 240 Hz without joint armature."""
    for name, actuator in G1ApplePickPlaceEnvCfg().scene.robot.actuators.items():
        assert actuator.armature is not None, name


def test_each_actuator_value_pattern_matches_a_joint_once():
    """Per-joint actuator values resolve without overlapping patterns, as the actuators do at spawn."""
    with open(START_STATES_FILE) as f:
        joint_names = json.load(f)["joint_names"]
    for name, actuator in G1ApplePickPlaceEnvCfg().scene.robot.actuators.items():
        _, names = resolve_matching_names(actuator.joint_names_expr, joint_names)
        assert names, name
        for field, value in vars(actuator).items():
            if isinstance(value, dict):
                resolve_matching_names_values(value, names)
