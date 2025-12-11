# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_mul, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class DifferentialInverseKinematicsAction(ActionTerm):
    """Differential inverse kinematics action term.

    This action term uses a differential IK controller to compute joint position commands to achieve
    a desired end-effector pose. The action is a delta pose (position + orientation) command.

    Key design:
    - `process_actions()`: Scales input actions and sets command in IK controller
    - `apply_actions()`: Computes Jacobian, runs IK solver, and applies joint position targets
    - `_processed_actions`: Stores **scaled end-effector commands**, NOT joint positions
    """

    cfg: actions_cfg.DifferentialInverseKinematicsActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _body_idx: int
    """The index of the end-effector body."""

    def __init__(self, cfg: actions_cfg.DifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # parse the body index
        body_ids, body_names = self._asset.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected a single body match for '{cfg.body_name}', got {len(body_ids)}: {body_names}"
            )
        self._body_idx = body_ids[0]

        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=cfg.controller,
            num_envs=self.num_envs,
            device=self.device
        )

        # create tensors for raw and processed actions
        # CRITICAL: _processed_actions stores SCALED ACTIONS (end-effector commands), NOT joint positions!
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # convert body offset to tensor if specified
        if cfg.body_offset_pos is not None:
            self._offset_pos = torch.tensor(cfg.body_offset_pos, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos = None

        if cfg.body_offset_rot is not None:
            self._offset_rot = torch.tensor(cfg.body_offset_rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_rot = None

        # scaling for actions
        if isinstance(cfg.scale, (float, int)):
            self._scale = torch.full((self.num_envs, self.action_dim), float(cfg.scale), device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Only float is supported.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return self._ik_controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the action term."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed (scaled) actions - these are end-effector commands, NOT joint positions."""
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Process actions: scale and set command in IK controller.
        
        Does NOT compute joint positions - that happens in apply_actions().
        
        Args:
            actions: The input actions (end-effector pose commands).
        """
        # store and scale the raw actions
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._scale

        # get current end-effector pose (with offset applied)
        ee_pos_w, ee_quat_w = self._compute_frame_pose()

        # set the command for the IK controller
        self._ik_controller.set_command(self._processed_actions, ee_pos=ee_pos_w, ee_quat=ee_quat_w)

    def apply_actions(self):
        """Apply actions: compute Jacobian, solve IK, and apply joint position targets."""
        # get current end-effector pose and joint positions
        ee_pos_w, ee_quat_w = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]

        # compute the Jacobian for the end-effector
        jacobian = self._compute_frame_jacobian()

        # compute the desired joint positions using IK
        joint_pos_des = self._ik_controller.compute(
            ee_pos=ee_pos_w,
            ee_quat=ee_quat_w,
            jacobian=jacobian,
            joint_pos=joint_pos
        )

        # apply the joint position targets
        self._asset.set_joint_position_target(joint_pos_des, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the action term.

        Args:
            env_ids: The environment indices to reset. If None, then all environments are reset.
        """
        # Zero out actions
        if env_ids is None:
            self._raw_actions[:] = 0.0
            # Reset IK controller for all environments
            self._ik_controller.reset(torch.arange(self.num_envs, device=self.device))
        else:
            self._raw_actions[env_ids] = 0.0
            # Reset IK controller for specific environments
            self._ik_controller.reset(env_ids)

    """
    Helper methods.
    """

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the end-effector pose in the world frame with offset applied.

        Returns:
            A tuple of (position, orientation) tensors.
            - position: (num_envs, 3) in world frame
            - orientation: (num_envs, 4) as quaternion (w, x, y, z) in world frame
        """
        # get the body pose in world frame
        body_pos_w = self._asset.data.body_pos_w[:, self._body_idx, :]
        body_quat_w = self._asset.data.body_quat_w[:, self._body_idx, :]

        # apply offset if specified
        if self._offset_pos is not None or self._offset_rot is not None:
            offset_pos = self._offset_pos if self._offset_pos is not None else torch.zeros_like(body_pos_w)
            offset_rot = self._offset_rot if self._offset_rot is not None else torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            
            # combine transforms: body_T_world * offset_T_body = ee_T_world
            ee_pos_w, ee_quat_w = combine_frame_transforms(body_pos_w, body_quat_w, offset_pos, offset_rot)
        else:
            ee_pos_w, ee_quat_w = body_pos_w, body_quat_w

        return ee_pos_w, ee_quat_w

    def _compute_frame_jacobian(self) -> torch.Tensor:
        """Compute the geometric Jacobian using Pinocchio with the robot's URDF.

        This uses Pinocchio's analytical Jacobian computation which is accurate and efficient.
        The Jacobian is computed using the robot's kinematic model loaded from URDF.

        Returns:
            The geometric Jacobian matrix (num_envs, 6, num_joints).
        """
        # Lazy import and initialize Pinocchio model
        if not hasattr(self, "_pinocchio_model"):
            try:
                self._initialize_pinocchio()
            except (ImportError, FileNotFoundError) as e:
                raise RuntimeError(
                    f"Failed to initialize Pinocchio for Jacobian computation: {e}\n"
                    "Please install Pinocchio: pip install pin"
                ) from e
        
        import pinocchio as pin
        import numpy as np
        
        num_envs = self.num_envs
        num_joints = self._num_joints
        
        # Initialize Jacobian tensor
        jacobian = torch.zeros(num_envs, 6, num_joints, device=self.device)
        
        # Get current joint positions
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids].cpu().numpy()
        
        # Compute Jacobian for each environment
        for env_idx in range(num_envs):
            # Update Pinocchio model with current joint configuration
            q = joint_pos[env_idx]
            
            # Compute forward kinematics
            pin.forwardKinematics(self._pinocchio_model, self._pinocchio_data, q)
            pin.updateFramePlacements(self._pinocchio_model, self._pinocchio_data)
            
            # Compute Jacobian for the end-effector frame
            # getFrameJacobian returns Jacobian in the LOCAL frame, so we need WORLD frame
            J = pin.computeFrameJacobian(
                self._pinocchio_model,
                self._pinocchio_data,
                q,
                self._ee_frame_id,
                pin.ReferenceFrame.WORLD
            )
            
            # Convert to torch and store (Pinocchio gives (6, njoints))
            jacobian[env_idx] = torch.from_numpy(J[:, :num_joints]).to(device=self.device, dtype=torch.float32)
        
        return jacobian
    
    def _initialize_pinocchio(self):
        """Initialize Pinocchio model from URDF for Jacobian computation."""
        import pinocchio as pin
        import os
        
        # Path to Franka URDF
        urdf_path = os.path.join(
            os.path.dirname(__file__),
            "../../controllers/config/data/lula_franka_gen.urdf"
        )
        
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(
                f"Franka URDF not found at {urdf_path}. "
                "Cannot compute Jacobian without kinematic model."
            )
        
        # Load the URDF model
        self._pinocchio_model = pin.buildModelFromUrdf(urdf_path)
        self._pinocchio_data = self._pinocchio_model.createData()
        
        # Find the end-effector frame ID
        # The frame name should match the body name from the config
        if hasattr(self._asset, 'body_names') and len(self._asset.body_names) > 0:
            ee_frame_name = self._asset.body_names[self._body_idx]
        else:
            # Default to common Franka end-effector frame names
            ee_frame_name = "panda_hand"  # or "panda_link8"
        
        # Find frame ID in Pinocchio model
        try:
            self._ee_frame_id = self._pinocchio_model.getFrameId(ee_frame_name)
        except:
            # Try alternative names
            for name in ["panda_hand", "panda_link8", "panda_link7", "end_effector"]:
                try:
                    self._ee_frame_id = self._pinocchio_model.getFrameId(name)
                    print(f"[INFO] Using end-effector frame: {name}")
                    break
                except:
                    continue
            else:
                raise RuntimeError(
                    f"Could not find end-effector frame '{ee_frame_name}' in Pinocchio model. "
                    f"Available frames: {[self._pinocchio_model.frames[i].name for i in range(self._pinocchio_model.nframes)]}"
                )


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion inverse (conjugate for unit quaternions).
    
    Args:
        q: Quaternion tensor (N, 4) in (w, x, y, z) format.
    
    Returns:
        Inverse quaternion (N, 4).
    """
    return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=-1)
