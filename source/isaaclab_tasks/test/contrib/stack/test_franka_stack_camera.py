# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera randomization and teacher checkpoint behavior for Franka stacking."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from pxr import Usd

from isaaclab.sensors import CameraCfg
from isaaclab.utils import instantiate
from isaaclab.utils.noise import UniformNoiseCfg

from isaaclab_tasks.contrib.stack.mdp.camera import randomize_camera_calibration
from isaaclab_tasks.contrib.stack.mdp.camera_noise_cfg import EpisodeCameraNoiseCfg
from isaaclab_tasks.utils import load_cfg_from_registry

pytestmark = pytest.mark.kitless


def test_episode_camera_noise_resamples_only_reset_environments() -> None:
    """An environment reset preserves episode photometric parameters in other environments."""
    cfg = EpisodeCameraNoiseCfg(
        noise_cfg=UniformNoiseCfg(n_min=0.0, n_max=0.0),
        exposure_range=(0.5, 1.5),
        contrast_range=(1.0, 1.0),
        white_balance_range=(1.0, 1.0),
        brightness_range=(0.0, 0.0),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        noise = instantiate(cfg, num_envs=4, device="cpu")
        image = torch.ones(4, 3, 2, 2)
        before = noise(image)
        noise.reset(torch.tensor([1, 3]))
        after = noise(image)

    torch.testing.assert_close(after[[0, 2]], before[[0, 2]], rtol=0.0, atol=0.0)
    assert torch.all((after[[1, 3]] != before[[1, 3]]).any(dim=(1, 2, 3)))


def test_camera_calibration_keeps_the_configured_pose_without_jitter() -> None:
    """An edited camera offset controls the nominal pose, including roll, in the selected environment."""
    camera = SimpleNamespace(
        cfg=CameraCfg(offset=CameraCfg.OffsetCfg(pos=(1.0, 2.0, 3.0), rot=(0.5, 0.5, 0.5, 0.5), convention="opengl")),
        stage=Usd.Stage.CreateInMemory(),
        set_world_poses=Mock(),
    )
    env = SimpleNamespace(
        device="cpu",
        num_envs=2,
        scene=SimpleNamespace(sensors={"base_camera": camera}, env_origins=torch.tensor([[0.0, 0, 0], [2.5, 0, 0]])),
    )

    randomize_camera_calibration(env, [1], 0.7993122, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    camera.set_world_poses.assert_called_once()
    (position, orientation), kwargs = camera.set_world_poses.call_args
    torch.testing.assert_close(position, torch.tensor([[3.5, 2.0, 3.0]]))
    torch.testing.assert_close(orientation * orientation[:, 3:].sign(), torch.full((1, 4), 0.5))
    torch.testing.assert_close(kwargs["env_ids"], torch.tensor([1]))
    assert kwargs["convention"] == "opengl"


def test_camera_student_loads_the_ppo_actor_and_uses_clipped_teacher_labels() -> None:
    """A native PPO actor checkpoint is a teacher, while the student runs on RGB and proprioception."""
    pytest.importorskip("rsl_rl")
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    from isaaclab_tasks.contrib.stack.config.franka.agents.distillation import ClippedTeacherDistillation

    state_task = "IsaacContrib-Stack-Cube-Franka-RL"
    student_task = "IsaacContrib-Stack-Cube-Franka-RL-Camera-Distillation"
    state_cfg = load_cfg_from_registry(state_task, "rsl_rl_cfg_entry_point")
    distillation_cfg = load_cfg_from_registry(student_task, "rsl_rl_cfg_entry_point").to_dict()

    state = torch.zeros(1, 100)
    state_obs = TensorDict({"policy": state}, [1])
    state_actor_cfg = state_cfg.actor.to_dict()
    state_actor_cfg.pop("class_name")
    state_actor = MLPModel(state_obs, state_cfg.obs_groups, "actor", 8, **state_actor_cfg)
    output = torch.tensor([[-2.0, 2.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]])
    linear_layers = [module for module in state_actor.modules() if isinstance(module, torch.nn.Linear)]
    last_linear = linear_layers[-1]
    with torch.no_grad():
        last_linear.weight.zero_()
        last_linear.bias.copy_(output[0])
        torch.testing.assert_close(state_actor(state_obs), output)

    visual_obs = TensorDict({"policy": torch.zeros(1, 33), "base_image": torch.zeros(1, 3, 128, 128)}, [1])
    camera_obs = visual_obs.clone().set("privileged", state)
    distillation_cfg["multi_gpu"] = None
    env_spec = SimpleNamespace(num_actions=8, num_envs=1, clip_actions=0.5)
    algorithm = ClippedTeacherDistillation.construct_algorithm(camera_obs, env_spec, distillation_cfg, device="cpu")
    algorithm.load({"actor_state_dict": state_actor.state_dict()}, load_cfg=None, strict=True)

    with torch.no_grad():
        assert algorithm.get_policy()(visual_obs).shape == (1, 8)
        algorithm.act(camera_obs)

    expected_labels = torch.tensor([[-0.5, 0.5, 0.25, 0, 0, 0, 0, 0]])
    torch.testing.assert_close(algorithm.transition.privileged_actions, expected_labels)
