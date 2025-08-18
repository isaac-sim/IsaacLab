# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING
from pink.tasks import FrameTask

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from isaaclab.controllers.utils import load_torchscript_model
from isaaclab.controllers.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

"""
Upper Body IK Action
"""

G1_UPPER_BODY_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="pelvis",
    num_hand_joints=14,
    show_ik_warnings=False,
    variable_input_tasks=[
        FrameTask(
            "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
            position_cost=1.0,    # [cost] / [m]
            orientation_cost=1.0,    # [cost] / [rad]
            lm_damping=10,    # dampening for solver for step jumps
            gain=0.1,
        ),
        FrameTask(
            "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
            position_cost=1.0,    # [cost] / [m]
            orientation_cost=1.0,    # [cost] / [rad]
            lm_damping=10,    # dampening for solver for step jumps
            gain=0.1,
        ),
    ],
    fixed_input_tasks=[],
)

G1_UPPER_BODY_IK_ACTION_CFG = PinkInverseKinematicsActionCfg(
    pink_controlled_joint_names=[
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_roll_joint",
        ".*_wrist_yaw_joint",
    ],
    # Fixed joints for IK (pelvis, legs, and hands are fixed)
    # ik_urdf_fixed_joint_names=[
    #     ".*_hip_yaw_joint",
    #     ".*_hip_roll_joint",
    #     ".*_hip_pitch_joint",
    #     ".*waist.*",
    #     ".*_knee_joint",
    #     ".*_ankle_pitch_joint",
    #     ".*_ankle_roll_joint",
    #     ".*_index_.*",
    #     ".*_middle_.*",
    #     ".*_thumb_.*",
    # ],
    hand_joint_names=[
        ".*_index_.*",
        ".*_middle_.*",
        ".*_thumb_.*",
    ],
    target_eef_link_names={
        "left_wrist": "left_wrist_yaw_link",
        "right_wrist": "right_wrist_yaw_link",
    },
    # the robot in the sim scene we are controlling
    asset_name="robot",
    # Configuration for the IK controller
    # The frames names are the ones present in the URDF file
    # The urdf has to be generated from the USD that is being used in the scene
    controller=G1_UPPER_BODY_IK_CONTROLLER_CFG,
)

"""
Lower Body Action
"""

class LowerBodyAction(ActionTerm):
    cfg: LowerBodyActionCfg
    """The configuration of the action term."""

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: LowerBodyActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Save the observation config from cfg
        self._observation_cfg = env.cfg.observations
        self._obs_group_name = cfg.obs_group_name

        # Load policy here if needed
        self._policy = load_torchscript_model(cfg.policy_path, device=env.device)
        self._env = env

        # Find joint ids for the lower body joints
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)

        # Get the scale and offset from the configuration
        self._scale = torch.tensor(cfg.scale, device=env.device)
        self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        # Create tensors to store raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """ Lower Body Action: [vx, vy, wz, hip_height]"""
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Process the input actions using the locomotion policy.

        Args:
            actions: The lower body commands.
        """

        # Extract base command from the action tensor
        # Assuming the base command [vx, vy, wz, hip_height]
        base_command = actions  # Shape: [num_envs, 4]
        
        obs_tensor = self._env.obs_buf[self._obs_group_name]

        # Concatenate actions repeated by history length
        history_length = getattr(self._observation_cfg, self._obs_group_name).history_length
        # Default to 1 if history_length is None (no history, just current observation)
        if history_length is None:
            history_length = 1
        repeated_commands = base_command.unsqueeze(1).repeat(1, history_length, 1).reshape(base_command.shape[0], -1)
        policy_input = torch.cat([repeated_commands, obs_tensor], dim=-1)

        joint_actions = self._policy.forward(policy_input)

        # Store the raw actions or joint targets (used for last_action and history of actions)
        self._raw_actions[:] = joint_actions

        # Apply scaling and offset to the raw actions from the policy
        self._processed_actions = joint_actions * self._scale + self._offset

        # Clip actions if configured
        if self.cfg.clip is not None:
            import pdb; pdb.set_trace()
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


    def apply_actions(self):
        """Apply the actions to the environment. """
        # Store the raw actions
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

@configclass
class LowerBodyActionCfg(ActionTermCfg):
    """Configuration for the lower body action term."""

    class_type: type[ActionTerm] = LowerBodyAction
    """The class type for the lower body action term."""

    policy_path: str = MISSING
    """The path to the policy model."""

    joint_names: list[str] = MISSING
    """The names of the joints to control."""

    scale: float = 1.0
    """The scale of the action."""

    obs_group_name: str = MISSING
    """The name of the observation group to use."""
    
    offset: float = 0.0
    """The offset of the action."""