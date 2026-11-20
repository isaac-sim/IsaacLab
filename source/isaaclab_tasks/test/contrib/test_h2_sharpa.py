# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H2 policy action ordering and task progress across episode boundaries."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.managers import ActionManager, ObservationManager
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.warp import ProxyArray

import isaaclab_tasks
from isaaclab_tasks.contrib.h2_sharpa import POLICY_JOINT_NAMES, phase_reward
from isaaclab_tasks.contrib.pack_agx.pack_agx_env_cfg import PackAgxOrinEnvCfg
from isaaclab_tasks.contrib.pick_and_place_apple.pick_and_place_apple_env_cfg import PickAndPlaceAppleEnvCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


@pytest.mark.parametrize(
    "task, package", [("Pick-And-Place-Apple", "pick_and_place_apple"), ("Pack-AGX-Orin", "pack_agx")]
)
def test_rlinf_task_registration(task, package):
    """The train and play IDs select the same task-local environment and RLinf recipe."""
    task_id = f"IsaacContrib-{task}-H2-Sharpa"
    root = Path(isaaclab_tasks.__file__).parent
    task_dir = root / "contrib" / package
    for mode, env_id in (("train", task_id), ("eval", task_id + "-Eval")):
        cfg = load_cfg_from_registry(env_id, "env_cfg_entry_point")
        assert type(cfg).__module__ == f"isaaclab_tasks.contrib.{package}.{package}_env_cfg"
        recipe = load_cfg_from_registry(env_id, "rlinf_cfg_entry_point")
        assert recipe["env"][mode]["init_params"]["id"] == env_id

    # RLinf searches by basename. Copies outside agents/ make its selection ambiguous.
    yaml_name = f"rlinf_gr00t_{package}_ppo_cfg.yaml"
    assert list(root.rglob(yaml_name)) == [task_dir / "agents" / yaml_name]
    assert not (task_dir / "config" / "env_config.py").is_file()
    assert not list((task_dir / "config").glob("*.yaml"))


def test_policy_actions_preserve_uncontrolled_targets():
    """The 58 policy outputs command arm torques and hand positions in dataset order."""
    cfg = PickAndPlaceAppleEnvCfg()
    names = sorted(cfg.scene.robot.init_state.joint_pos)
    defaults = torch.tensor([cfg.scene.robot.init_state.joint_pos[name] for name in names]).repeat(2, 1)
    targets = defaults.clone()
    efforts = torch.zeros_like(defaults)
    velocity = torch.linspace(-0.5, 0.5, defaults.numel()).reshape_as(defaults)
    gravity = torch.linspace(-2.0, 2.0, defaults.numel()).reshape_as(defaults)

    def find_joints(patterns, preserve_order=False, as_proxy=False):
        indices, selected = resolve_matching_names(patterns, names, preserve_order=preserve_order)
        return (ProxyArray(wp.array(indices, dtype=wp.int32, device="cpu")) if as_proxy else indices), selected

    def set_position(target, joint_ids):
        targets[:, joint_ids] = target

    def set_effort(*, value, joint_ids):
        efforts[:, joint_ids] = value

    robot = SimpleNamespace(
        num_joints=len(names),
        num_base_dofs=0,
        find_joints=find_joints,
        data=SimpleNamespace(
            joint_pos=ProxyArray(wp.from_torch(defaults)),
            joint_vel=ProxyArray(wp.from_torch(velocity)),
            gravity_compensation_forces=ProxyArray(wp.from_torch(gravity)),
        ),
        set_joint_position_target_index=set_position,
        actuators=SimpleNamespace(target_command=SimpleNamespace(set_effort_index=set_effort)),
    )
    env = SimpleNamespace(
        scene={"robot": robot}, num_envs=2, device="cpu", sim=SimpleNamespace(is_playing=lambda: True)
    )
    manager = ActionManager(cfg.actions, env)
    commands = torch.linspace(-0.4, 0.4, 116).reshape(2, 58)
    manager.process_action(commands)
    manager.apply_action()

    policy_indices = [names.index(name) for name in POLICY_JOINT_NAMES]
    arms, hands = policy_indices[:14], policy_indices[14:]
    other = [i for i in range(len(names)) if i not in policy_indices]
    kp = torch.tensor(([150] * 4 + [50] * 3) * 2)
    kd = torch.tensor(([10] * 4 + [3] * 3) * 2)
    torch.testing.assert_close(
        efforts[:, arms], kp * (commands[:, :14] - defaults[:, arms]) - kd * velocity[:, arms] + gravity[:, arms]
    )
    torch.testing.assert_close(targets[:, hands], commands[:, 14:])
    torch.testing.assert_close(targets[:, other], defaults[:, other])
    assert torch.count_nonzero(efforts[:, hands + other]) == 0
    assert cfg.scene.robot.actuators["arms"].stiffness == cfg.scene.robot.actuators["arms"].damping == 0.0
    assert targets[0, names.index("head_pitch_joint")] == pytest.approx(0.6)


@pytest.mark.parametrize(
    "cfg_type, expected",
    [
        (PickAndPlaceAppleEnvCfg, [[123, 120, 117]] * 3),
        (PackAgxOrinEnvCfg, [[161, 149, 151], [134, 135, 131], [149, 147, 147]]),
    ],
)
def test_camera_group_returns_calibrated_rgb(cfg_type, expected):
    """The policy receives calibrated uint8 RGB from each configured sensor through the manager."""
    camera_names = ("front_camera", "left_wrist_camera", "right_wrist_camera")
    frame = torch.full((1, 1, 1, 3), 128, dtype=torch.uint8)

    class Scene(dict):
        @property
        def sensors(self):
            return self

    sensor = SimpleNamespace(data=SimpleNamespace(output={"rgb": SimpleNamespace(torch=frame)}))
    scene = Scene.fromkeys(camera_names, sensor)
    env = SimpleNamespace(num_envs=1, device="cpu", scene=scene, sim=SimpleNamespace(is_playing=lambda: True))
    cfg = cfg_type()
    cfg.to_dict()  # Train and play serialize the complete environment configuration.
    manager = ObservationManager({"camera_images": cfg.observations.camera_images}, env)
    images = manager.compute_group("camera_images")
    for name, rgb in zip(camera_names, expected, strict=True):
        torch.testing.assert_close(images[name], torch.tensor(rgb, dtype=torch.uint8).reshape(1, 1, 1, 3))


@pytest.mark.parametrize("cfg_type", [PickAndPlaceAppleEnvCfg, PackAgxOrinEnvCfg])
def test_progress_without_rewards_and_partial_reset(cfg_type):
    """Success progresses without evaluating rewards, and only the selected episode resets."""
    cfg = cfg_type().terminations.task_success
    apple = cfg_type is PickAndPlaceAppleEnvCfg
    object_name, target_name = ("apple", "plate") if apple else ("agx_orin", "protective_box")
    position = torch.zeros(2, 3)
    wrist = torch.zeros(2, 2, 3)
    body_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    joint_names = [
        "right_thumb_CMC_FE",
        "right_thumb_CMC_AA",
        "right_thumb_MCP_AA",
        "right_thumb_IP",
        "right_thumb_MCP_FE",
    ]
    joints = torch.zeros(2, len(joint_names))
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=ProxyArray(wp.from_torch(wrist, dtype=wp.vec3f)), joint_pos=ProxyArray(wp.from_torch(joints))
        ),
        body_names=body_names,
        num_bodies=len(body_names),
        joint_names=joint_names,
        num_joints=len(joint_names),
        find_bodies=lambda names, **kwargs: resolve_matching_names(names, body_names, **kwargs),
        find_joints=lambda names, **kwargs: resolve_matching_names(names, joint_names, **kwargs),
    )
    obj = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=ProxyArray(wp.from_torch(position, dtype=wp.vec3f)),
            root_quat_w=ProxyArray(wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2), dtype=wp.quatf)),
        )
    )
    target = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(2, 3)))
    env = SimpleNamespace(
        scene={object_name: obj, target_name: target, "robot": robot}, num_envs=2, device="cpu", step_dt=0.1
    )
    cfg.params["robot_cfg"].resolve(env.scene)
    cfg.params["robot_cfg"].finalize(env.device)
    term = cfg.func(cfg, env)
    env.termination_manager = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=term))
    term.reset()
    hold_steps = cfg.params["hold_steps" if apple else "lift_hold_steps"]
    for phase in range(4):
        position[:, 2] = (0.2 if phase < 2 else 0.05) if apple else (0.1 if phase < 2 else 0.0)
        wrist[:, 0, 0] = 0.0 if phase == 0 else 1.0
        wrist[:, 1, 0] = 0.0 if phase < 3 else 0.5
        for _ in range(hold_steps):
            done = term(env, **cfg.params)
        assert done.tolist() == [phase == 3, phase == 3]
        torch.testing.assert_close(phase_reward(env, phase), torch.full((2,), 10.0))
        assert not phase_reward(env, (phase + 1) % 4).any()
    # Remaining in the success state must not grant another transition reward.
    term(env, **cfg.params)
    assert not phase_reward(env, 3).any()
    position[0, 2] = 2.0
    term.reset(torch.tensor([0]))
    assert term(env, **cfg.params).tolist() == [False, True]
