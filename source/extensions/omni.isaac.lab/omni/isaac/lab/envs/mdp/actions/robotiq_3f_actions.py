
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from . import actions_cfg
    
    

class Robotiq3FingerAction(ActionTerm):
    """
    Class for the Robotiq 3-finger gripper action term.
    The commands are sent as two values from zero to one, one for the finger opening and one for the scissor opening.
    """
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    
    def __init__(self, cfg: actions_cfg.Robotiq3FingerActionCfg, env: ManagerBasedEnv) ->None:
        # initialize the action term
        super().__init__(cfg, env)
        
        self.side = cfg.side
        joint_names = []
        for finger_name in ["a", "b", "c"]:
            for joint in range(1, 4):
                joint_name = f"victor_{self.side}_finger_{finger_name}_joint_{joint}"
                joint_names.append(joint_name)
        joint_names.append(f"victor_{self.side}_palm_finger_b_joint")
        joint_names.append(f"victor_{self.side}_palm_finger_c_joint")
        self._joint_ids, self._joint_names = self._asset.find_joints(joint_names, preserve_order=True)
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        
        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        print("Robotiq3FingerAction initialized")
    
    """
    Properties.
    """
    
    @property
    def action_dim(self) -> int:
        return 2
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    """
    Operations
    """
    def process_actions(self, actions:torch.Tensor):
        """ Compute joint angles based on opening and scissor values """
        self._raw_actions[:] = actions
        joint_pos_curr = self._asset.read_joint_state_from_sim(joint_ids=self._joint_ids)["position"]
        openness_curr = inverse_compute_finger_angles_jit(joint_pos_curr)
        scissor_curr = inverse_compute_scissor_angle_jit(joint_pos_curr[:, -2:])
        if self.cfg.use_relative_mode:
            actions[:, 0] += openness_curr[:, 0]
            actions[:, 1] += scissor_curr[:, 0]
        
        compute_finger_angles_jit(actions[:, 0], self._processed_actions)
        
        # # debugging
        # inverse_angles = inverse_compute_finger_angles_jit(self._processed_actions)
        # if not torch.allclose(actions[:, 0], inverse_angles[:, 0]):
        #     omni.log.warn(f"Failed to compute finger angles for {self.side} finger a")
        
        # copy the joint angles to finger b and c
        self._processed_actions[:, 3:6] = self._processed_actions[:, 0:3]
        self._processed_actions[:, 6:9] = self._processed_actions[:, 0:3]
        
        # compute scissors angle
        compute_scissor_angle_jit(actions[:, 1], self._processed_actions[:, 9:11])
        # inverse_scissor_action = inverse_compute_scissor_angle_jit(self._processed_actions[:, 9:11])
        # if not torch.allclose(actions[:, 1], inverse_scissor_action[:, 0]):
        #     omni.log.warn(f"Failed to compute scissor angle for {self.side} finger")
        
    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
    

    
# @torch.jit.script
def compute_finger_angles_jit(control, finger_angles): 
    """Compute joint angles based on opening and scissor values 

    Args:
        control (torch.Tesnor): B x 2, representing the opening and scissor values. 0 corresponds to fully open, 1 is fully closed.
        
    """
    # Convert control input to g (range from 0 to 255)
    g_batch = control * 255

    max_angle = torch.tensor([70.0, 90.0, 43.0], device=control.device, dtype=control.dtype)
    min_3 = -55.0
    m1 = max_angle[0] / 140.0
    m2 = max_angle[1] / 100.0

    # Conditions based on g_batch values
    cond1 = g_batch <= 110.0
    cond2 = (g_batch > 110.0) & (g_batch <= 140.0)
    cond3 = (g_batch > 140.0) & (g_batch <= 240.0)
    cond4 = g_batch > 240.0

    # Calculate angles for each phase and store them directly into the tensor
    finger_angles[cond1, 0] = m1 * g_batch[cond1]  # theta1
    finger_angles[cond1, 1] = 0  # theta2
    finger_angles[cond1, 2] = -m1 * g_batch[cond1]  # theta3

    finger_angles[cond2, 0] = m1 * g_batch[cond2]  # theta1
    finger_angles[cond2, 1] = 0  # theta2
    finger_angles[cond2, 2] = min_3  # theta3

    finger_angles[cond3, 0] = max_angle[0]  # theta1
    finger_angles[cond3, 1] = m2 * (g_batch[cond3] - 140)  # theta2
    finger_angles[cond3, 2] = min_3  # theta3

    finger_angles[cond4, 0] = max_angle[0]  # theta1
    finger_angles[cond4, 1] = max_angle[1]  # theta2
    finger_angles[cond4, 2] = min_3  # theta3

    # Convert angles to radians
    torch.deg2rad(finger_angles, out=finger_angles)


def inverse_compute_finger_angles_jit(finger_angles):
    """Inverse operation to compute control values based on joint angles.

    Args:
        finger_angles (torch.Tensor): B x 3, representing the joint angles (theta1, theta2, theta3) in radians.
        
    Returns:
        control (torch.Tensor): B x 2, representing the opening and scissor values (0 corresponds to fully open, 1 is fully closed).
    """
    # Convert angles from radians to degrees
    finger_angles = torch.rad2deg(finger_angles)

    # Initialize control values
    control = torch.zeros((finger_angles.size(0), 2), device=finger_angles.device, dtype=finger_angles.dtype)

    max_angle = torch.tensor([70.0, 90.0, 43.0], device=finger_angles.device, dtype=finger_angles.dtype)
    min_3 = -55.0
    m1 = max_angle[0] / 140.0
    m2 = max_angle[1] / 100.0

    # Reverse conditions to compute g_batch
    cond1 = finger_angles[:, 2] > min_3  # For cond1 and cond2 (theta3 not at min_3)
    cond2 = (finger_angles[:, 2] == min_3) & (finger_angles[:, 1] == 0)  # For cond2
    cond3 = (finger_angles[:, 2] == min_3) & (finger_angles[:, 1] > 0) & (finger_angles[:, 1] < max_angle[1])  # For cond3
    cond4 = (finger_angles[:, 2] == min_3) & (finger_angles[:, 1] == max_angle[1])  # For cond4

    # Compute g_batch based on conditions
    g_batch = torch.zeros_like(finger_angles[:, 0])

    g_batch[cond1] = finger_angles[cond1, 0] / m1  # Phase 1
    g_batch[cond2] = finger_angles[cond2, 0] / m1  # Phase 2
    g_batch[cond3] = 140 + (finger_angles[cond3, 1] / m2)  # Phase 3
    g_batch[cond4] = 240 + (finger_angles[cond4, 1] / 0)  # Phase 4, upper bound

    # Convert g_batch back to control values (range from 0 to 1)
    control[:, 0] = g_batch / 255

    return control
    
# @torch.jit.script
def compute_scissor_angle_jit(control, scissor_angle):
    # 0 corresponds to fully open at -16 degrees, 1 is fully closed with at +10 degrees
    torch.deg2rad(16 - 26.0 * control, out=scissor_angle[:, 0])
    scissor_angle[:, 1] = -scissor_angle[:, 0]

def inverse_compute_scissor_angle_jit(scissor_angle):
    scissor_control = torch.zeros((scissor_angle.size(0), 1), device=scissor_angle.device, dtype=scissor_angle.dtype)
    scissor_control[:, 0] = (16 - torch.rad2deg(scissor_angle[:, 0])) / 26.0
    return scissor_control