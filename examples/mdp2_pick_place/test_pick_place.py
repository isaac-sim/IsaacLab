# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MDP2 Franka pick-and-place example on Newton, with its cameras and episode budget."""

import pytest
import torch
from isaaclab_experimental.envs import MDP2RLEnv
from pick_place_env_cfg import CUBE_SIZE, PLATE_SIZE, PickPlaceEnvCfg

pytestmark = [pytest.mark.kitless, pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")]

NUM_ENVS = 2
BUDGET = 3


def test_episodes_follow_the_droid_contract_and_the_budget():
    """The policy sees DROID's observations and actions, success ends an episode once the cube was lifted and then
    placed, and the episode budget records each episode and stops the worlds once it is spent."""
    cfg = PickPlaceEnvCfg()
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    cfg.episode_budget = BUDGET
    env = MDP2RLEnv(cfg)
    try:
        obs, _ = env.reset()
        policy, cameras = obs["policy"], obs["camera_obs"]
        assert {name: tuple(value.shape[1:]) for name, value in policy.items()} == {
            "actions": (8,),
            "robot_joint_pos": (9,),
            "joint_pos": (7,),
            "gripper_pos": (1,),
        }
        for name in ("external_camera_rgb", "wrist_camera_rgb"):
            image = cameras[name]
            assert image.shape == (NUM_ENVS, 180, 320, 3) and image.dtype == torch.uint8
            # Rendered, not blank: the table, the robot, and the objects give the image contrast.
            assert image.float().std() > 10.0

        robot, cube, plate = env.scene["robot"], env.scene["cube"], env.scene["plate"]
        actions = torch.zeros(NUM_ENVS, 8, device=env.device)
        actions[:, :7] = robot.data.default_joint_pos.torch[:, :7]
        actions[:, 7] = 1.0  # Close the gripper
        env.step(actions)
        finger = robot.joint_names.index("panda_finger_joint1")
        assert (robot.data.joint_pos_target.torch[:, finger] == 0.0).all()

        def teleport(heights: torch.Tensor, on_plate: bool) -> None:
            """Move the cube of every world above its own position, or onto the plate, at rest."""
            pose = cube.data.root_link_pose_w.torch.clone()
            if on_plate:
                pose[:, :2] = plate.data.root_link_pose_w.torch[:, :2]
            pose[:, 2] = heights
            pose[:, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
            cube.write_root_pose_to_sim_index(root_pose=pose)
            cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros(NUM_ENVS, 6, device=env.device))

        # Only world 0 completes the task: it lifts the cube, then places it. World 1 places it without lifting it.
        lifted = torch.tensor([0.3, CUBE_SIZE / 2], device=env.device)
        teleport(lifted, on_plate=False)
        _, _, terminated, _, _ = env.step(actions)
        assert not terminated.any()
        # The second step replays the whole step, the cameras included, as one graph.
        assert env.is_captured and env.num_graphs == 1

        resting = PLATE_SIZE[2] + CUBE_SIZE / 2 + 0.001
        teleport(torch.full((NUM_ENVS,), resting, device=env.device), on_plate=True)
        terminated_steps = []
        for _ in range(10):  # Until the cube settles on the plate
            _, _, terminated, truncated, _ = env.step(actions)
            terminated_steps.append(terminated.tolist())
            if terminated[0]:
                break
        assert terminated_steps[-1] == [True, False], terminated_steps
        assert env.episodes_completed == 1

        records = env.episode_records()
        assert records["world"].tolist() == [0.0] and records["episode"].tolist() == [0.0]
        assert records["termination/success"].tolist() == [1.0] and records["progress/score"].tolist() == [1.0]
        # Lifted at the first step after the gripper closed, placed once the cube settled on the plate.
        assert 0 < records["progress/lift"].item() < records["progress/place"].item() == records["progress/task"].item()
        assert records["terminated"].tolist() == [1.0] and records["truncated"].tolist() == [0.0]

        # Drop the cubes to end two more episodes: the budget of three is then spent, and the worlds stop.
        teleport(torch.full((NUM_ENVS,), -1.0, device=env.device), on_plate=False)
        _, _, terminated, _, _ = env.step(actions)
        assert terminated.tolist() == [True, True] and env.episodes_completed == BUDGET
        assert env.episode_records()["world"].tolist() == [0.0, 0.0, 1.0]
        teleport(torch.full((NUM_ENVS,), -1.0, device=env.device), on_plate=False)
        _, _, terminated, truncated, _ = env.step(actions)
        assert not (terminated | truncated).any() and env.episodes_completed == BUDGET
    finally:
        env.close()
