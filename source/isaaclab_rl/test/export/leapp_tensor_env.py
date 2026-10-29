# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tensor-backed scene for exercising Isaac Lab's real export managers without simulation."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import warp as wp

from isaaclab.assets.articulation import BaseArticulation, BaseArticulationData
from isaaclab.envs import ManagerBasedRLEnv, mdp
from isaaclab.managers import ActionManager, ObservationGroupCfg, ObservationManager, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors.camera import CameraData
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.tensor_processors import TensorProcessorCfg, bias
from isaaclab.utils.warp import ProxyArray


class _JointDataSemantics:
    joint_pos = BaseArticulationData.joint_pos
    joint_vel = BaseArticulationData.joint_vel
    default_joint_pos = BaseArticulationData.default_joint_pos


class _JointData(_JointDataSemantics):
    def __init__(self, width: int):
        self.positions = torch.full((1, width), 0.25)
        self.velocities = torch.full((1, width), 0.125)
        self._defaults = torch.zeros(1, width)

    @property
    def joint_pos(self) -> ProxyArray:
        return ProxyArray(wp.from_torch(self.positions))

    @property
    def joint_vel(self) -> ProxyArray:
        return ProxyArray(wp.from_torch(self.velocities))

    @property
    def default_joint_pos(self) -> ProxyArray:
        return ProxyArray(wp.from_torch(self._defaults))


class _TensorArticulation(BaseArticulation):
    """Expose only the articulation API used by observation and joint action terms."""

    def __init__(self, width: int):
        self._width = width
        self._data = _JointData(width)
        self._joint_names = [f"joint_{index}" for index in range(width)]
        self.targets = []
        self.actuators = SimpleNamespace(
            target_command=SimpleNamespace(set_position_index=lambda **kwargs: self.targets.append(kwargs))
        )

    @property
    def data(self) -> _JointData:
        return self._data

    @property
    def num_joints(self) -> int:
        return self._width

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    def find_joints(self, name_keys, joint_subset=None, preserve_order=False, *, as_proxy=False):
        if joint_subset is not None:
            raise NotImplementedError("The tensor scene only supports whole-articulation joint selection")
        indices, names = resolve_matching_names(name_keys, self.joint_names, preserve_order)
        if as_proxy:
            return ProxyArray(wp.from_torch(torch.tensor(indices, dtype=torch.long))), names
        return indices, names


# BaseArticulation's other abstract methods belong to the physics backend and are never called here.
_TensorArticulation.__abstractmethods__ = frozenset()


class _TensorScene(dict):
    def __init__(self, width: int):
        robot = _TensorArticulation(width)
        camera_data = CameraData()
        self.pixels = torch.randint(
            0, 256, (1, 4, 4, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(13)
        )
        camera_data._output = {"rgb": ProxyArray(wp.from_torch(self.pixels))}
        camera = SimpleNamespace(data=camera_data)
        super().__init__(robot=robot, camera=camera)
        self.articulations = {"robot": robot}
        self.sensors = {"camera": camera}


class _ActionCfg(dict):
    """ActionManager expects attribute access for its aggregate debug flag."""


class TensorManagerEnv(ManagerBasedRLEnv):
    """Run the real managers against owned tensor buffers instead of a physics scene."""

    def __init__(self, cfg, *, task: str, backend: str, width: int, render_mode=None):
        self.cfg = cfg
        self.spec = SimpleNamespace(id=task)
        self.env = self  # ExportPatcher accepts a Gym wrapper whose inner env is unwrapped.
        self.render_mode = render_mode
        self.sim = SimpleNamespace(is_playing=lambda: True)
        self.scene = _TensorScene(width)
        self._generator = torch.Generator().manual_seed(17)

        policy = {
            "joint_pos": ObservationTermCfg(
                func=mdp.joint_pos_rel,
                tensor_processors=[TensorProcessorCfg(func=bias, params={"value": 0.1})],
                clip=(-1.0, 1.0),
                scale=2.0,
            ),
            "joint_vel": ObservationTermCfg(func=mdp.joint_vel, history_length=2 if backend != "sb3" else 0),
            "previous": ObservationTermCfg(func=mdp.last_action),
        }
        policy_group = ObservationGroupCfg(concatenate_terms=backend != "sb3")
        for name, term in policy.items():
            setattr(policy_group, name, term)
        proprio_group = ObservationGroupCfg()
        proprio_group.joint_pos = ObservationTermCfg(
            func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_ids=[0, 1])}
        )
        vision_group = ObservationGroupCfg()
        vision_group.rgb = ObservationTermCfg(
            func=mdp.image_rgb, params={"sensor_cfg": SceneEntityCfg("camera"), "channel_first": True}
        )
        critic_group = ObservationGroupCfg()
        critic_group.joint_pos = ObservationTermCfg(
            func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_ids=[0, 1])}
        )
        observations = {"policy": policy_group, "proprio": proprio_group, "vision": vision_group}
        if width == 10:
            observations["critic"] = critic_group
        action_cfg = _ActionCfg(
            actions=mdp.JointPositionActionCfg(
                joint_names=[".*"],
                asset_name="robot",
                use_default_offset=False,
                scale=0.5,
                offset=0.1,
                clip={"joint_.*": (-0.5, 0.5)},
            )
        )
        self.action_manager = ActionManager(action_cfg, self)
        self.observation_manager = ObservationManager(observations, self)
        self.single_action_space = gym.spaces.Box(-np.inf, np.inf, (width,), dtype=np.float32)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, 1)
        spaces = {}
        for group, value in self.observation_manager.compute().items():
            if isinstance(value, dict):
                spaces[group] = gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(-np.inf, np.inf, tuple(tensor.shape[1:]), dtype=np.float32)
                        for name, tensor in value.items()
                    }
                )
            else:
                spaces[group] = gym.spaces.Box(-np.inf, np.inf, tuple(value.shape[1:]), dtype=np.float32)
        self.single_observation_space = gym.spaces.Dict(spaces)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, 1)
        self.obs_buf = self.observation_manager.compute()

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def num_envs(self) -> int:
        return 1

    @property
    def max_episode_length(self) -> int:
        return 100

    def seed(self, seed: int = -1) -> int:
        return seed

    def reset(self, *, seed=None, options=None):
        self.scene.pixels.random_(0, 256, generator=self._generator)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, {}

    def step(self, actions):
        self.action_manager.process_action(actions)
        self.action_manager.apply_action()
        self.scene.pixels.random_(0, 256, generator=self._generator)
        self.scene["robot"].data.positions.add_(0.01)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {}

    def close(self):
        pass
