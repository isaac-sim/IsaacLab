# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import os
import torch

import pytorch_kinematics as pk
from pytorch_kinematics.transforms import Transform3d

from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.box_pushing.box_pushing_env_cfg import BoxPushingEnvCfg

# from timeit import default_timer as timer


class BoxPushingEnv(ManagerBasedRLEnv):

    def __init__(self, cfg: BoxPushingEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """
        # initialize the base class to setup the scene.
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.step_dt  # TODO find better solution

        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,))

        # batch the spaces for vectorized environments
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        print("Precomputing IK Solutions for the box sampling space")

        # initializing kinematic chain from urdf for IK
        script_dir = os.path.dirname(__file__)
        urdf = os.path.abspath(os.path.join(script_dir, "assets", "franka_ik.urdf"))

        with open(urdf, "rb") as urdf_file:
            urdf_data = urdf_file.read()
        self.chain = pk.build_serial_chain_from_urdf(urdf_data, "panda_hand")
        self.chain = self.chain.to(device=self.device)

        self._push_offset = torch.tensor([0.0, 0.0, 0.27], device=self.device)
        self._ik_cache_targets = None
        self._ik_cache_joints = None

        robot: Articulation = self.scene["robot"]
        self._ik_solver = pk.PseudoInverseIK(
            self.chain,
            retry_configs=robot.data.joint_pos[0, :7].unsqueeze(0),
            joint_limits=robot.data.joint_limits[0, :7],
            lr=0.2,
        )

        if self.cfg.use_cached_ik and self.cfg.use_ik_reset:
            self._precompute_ik_cache(self.cfg.pose_sampling_range)

    def _precompute_ik_cache(self, pose_range: dict[str, tuple[float, float]]):
        """Precompute IK targets for sampled box poses."""
        num_samples = self.cfg.ik_cache_num_samples
        device = self.device
        ranges = torch.tensor(
            [
                pose_range.get("x", (0.0, 0.0)),
                pose_range.get("y", (0.0, 0.0)),
                pose_range.get("z", (0.0, 0.0)),
            ],
            dtype=torch.float32,
            device=device,
        )
        offsets = torch.rand(num_samples, 3, device=device) * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]

        robot = self.scene["robot"]
        box = self.scene["object"]
        base_offset = (box.data.default_root_state[0, :3] - robot.data.default_root_state[0, :3]).to(device)
        target_positions = base_offset.unsqueeze(0) + offsets + self._push_offset

        target_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).repeat(num_samples, 1)
        transforms = Transform3d(pos=target_positions, rot=target_quat, device=device)
        sol = self._ik_solver.solve(transforms)
        joint_pos_des = sol.solutions[:, 0]

        self._ik_cache_targets = target_positions
        self._ik_cache_joints = joint_pos_des

    def get_cached_ik_solutions(self, relative_box_positions: torch.Tensor) -> torch.Tensor:
        """Return cached IK solutions for the provided relative box positions."""
        if self._ik_cache_targets is None or self._ik_cache_joints is None:
            raise RuntimeError("IK cache not initialized.")
        target_positions = relative_box_positions + self._push_offset
        distances = torch.cdist(target_positions, self._ik_cache_targets)
        indices = torch.argmin(distances, dim=1)
        return self._ik_cache_joints[indices]
