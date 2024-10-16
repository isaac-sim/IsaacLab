# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Kuka R800 robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""
from math import radians
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.assets.articulation import Articulation

import omni.log
import torch

##
# Configuration
##
##
# Configuration
##

KUKA_VICTOR_LEFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path="assets/victor/victor_left_arm_with_gripper.usd",
        usd_path="assets/victor/victor_left_arm_with_approx_gripper/victor_left_arm_with_approx_gripper.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.001)
    ),
    
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        joint_pos={
            # arm states
            "victor_left_arm_joint_1": 1.3661269501533881,
            "victor_left_arm_joint_2": -0.5341374194622199,
            "victor_left_arm_joint_3": 2.383251686578518,
            "victor_left_arm_joint_4": 1.6179420456098288,
            "victor_left_arm_joint_5": -2.204557118713759,
            "victor_left_arm_joint_6": 1.1547660552023602,
            "victor_left_arm_joint_7": 0.5469460457579646,
            # gripper finger states
            "victor_left_finger_a_joint_1": 0.890168571428571,
            "victor_left_finger_a_joint_2": 0,
            "victor_left_finger_a_joint_3": -0.8901685714285714,
            "victor_left_finger_b_joint_1": 0.890168571428571,
            "victor_left_finger_b_joint_2": 0,
            "victor_left_finger_b_joint_3": -0.8901685714285714,
            "victor_left_finger_c_joint_1": 0.890168571428571,
            "victor_left_finger_c_joint_2": 0,
            "victor_left_finger_c_joint_3": -0.8901685714285714,
            # gripper scissors states
            "victor_left_palm_finger_b_joint": 0.115940392156862,
            "victor_left_palm_finger_c_joint": -0.11594039215686275,
        },
    ),
    actuators={
        "victor_left_arm": ImplicitActuatorCfg(
            joint_names_expr=["victor_left_arm_joint.*"],
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "victor_left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["victor_left.*finger.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        )
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Kuka iiwa robot."""

KUKA_VICTOR_LEFT_HIGH_PD_CFG = KUKA_VICTOR_LEFT_CFG.copy()
KUKA_VICTOR_LEFT_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
KUKA_VICTOR_LEFT_HIGH_PD_CFG.actuators["victor_left_arm"].stiffness = 400.0
KUKA_VICTOR_LEFT_HIGH_PD_CFG.actuators["victor_left_arm"].damping = 80.0
"""Configuration of Kuka iiwa with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""


KUKA_VICTOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path="assets/victor/victor_approx_gripper/victor_approx_gripper.usd",
        usd_path="assets/victor/victor_full_gripper/victor_full_gripper.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.5,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.001, rest_offset=0)
    ),
    
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(-0.4, -0.35, -0.8),
        pos=(0, 0, 0),
        joint_pos={
            # arm states
            # "victor_left_arm_joint_1": -0.694,
            # "victor_left_arm_joint_2": 0.140,
            # "victor_left_arm_joint_3": -0.229,
            # "victor_left_arm_joint_4": -1.110,
            # "victor_left_arm_joint_5": -0.512,
            # "victor_left_arm_joint_6": 1.272,
            # "victor_left_arm_joint_7": 0.077,
            # "victor_right_arm_joint_1": 0.724,
            # "victor_right_arm_joint_2": 0.451,
            # "victor_right_arm_joint_3": 0.940,
            # "victor_right_arm_joint_4": -1.425,
            # "victor_right_arm_joint_5": 0.472,
            # "victor_right_arm_joint_6": 0.777,
            # "victor_right_arm_joint_7": -0.809,
            
            "victor_left_arm_joint_1": 1.3661269501533881,
            "victor_left_arm_joint_2": -0.5341374194622199,
            "victor_left_arm_joint_3": 2.383251686578518,
            "victor_left_arm_joint_4": 1.6179420456098288,
            "victor_left_arm_joint_5": -2.204557118713759,
            "victor_left_arm_joint_6": 1.1547660552023602,
            "victor_left_arm_joint_7": 0.5469460457579646,
            "victor_right_arm_joint_1": 0.724,
            "victor_right_arm_joint_2": 0.451,
            "victor_right_arm_joint_3": 0.940,
            "victor_right_arm_joint_4": -1.425,
            "victor_right_arm_joint_5": 0.472,
            "victor_right_arm_joint_6": 0.777,
            "victor_right_arm_joint_7": -0.809,
            
            # gripper finger states
            "victor_left_finger_a_joint_1": 0.890168571428571,
            "victor_left_finger_a_joint_2": 0,
            "victor_left_finger_a_joint_3": -0.8901685714285714,
            "victor_left_finger_b_joint_1": 0.890168571428571,
            "victor_left_finger_b_joint_2": 0,
            "victor_left_finger_b_joint_3": -0.8901685714285714,
            "victor_left_finger_c_joint_1": 0.890168571428571,
            "victor_left_finger_c_joint_2": 0,
            "victor_left_finger_c_joint_3": -0.8901685714285714,
            
            "victor_right_finger_a_joint_1": 0.890168571428571,
            "victor_right_finger_a_joint_2": 0,
            "victor_right_finger_a_joint_3": -0.8901685714285714,
            "victor_right_finger_b_joint_1": 0.890168571428571,
            "victor_right_finger_b_joint_2": 0,
            "victor_right_finger_b_joint_3": -0.8901685714285714,
            "victor_right_finger_c_joint_1": 0.890168571428571,
            "victor_right_finger_c_joint_2": 0,
            "victor_right_finger_c_joint_3": -0.8901685714285714,
            
            # gripper scissors states
            "victor_left_palm_finger_b_joint": 0.115940392156862,
            "victor_left_palm_finger_c_joint": -0.11594039215686275,
            "victor_right_palm_finger_b_joint": 0.115940392156862,
            "victor_right_palm_finger_c_joint": -0.11594039215686275,
        },
    ),
    actuators={
        "victor_left_arm": ImplicitActuatorCfg(
            joint_names_expr=["victor_left_arm_joint.*"],
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "victor_right_arm": ImplicitActuatorCfg(
            joint_names_expr=["victor_right_arm_joint.*"],
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "victor_left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["victor_left.*finger.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        "victor_right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["victor_right.*finger.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Kuka iiwa robot."""

KUKA_VICTOR_HIGH_PD_CFG = KUKA_VICTOR_CFG.copy()
KUKA_VICTOR_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
KUKA_VICTOR_HIGH_PD_CFG.actuators["victor_left_arm"].stiffness = 400.0
KUKA_VICTOR_HIGH_PD_CFG.actuators["victor_left_arm"].damping = 80.0
KUKA_VICTOR_HIGH_PD_CFG.actuators["victor_right_arm"].stiffness = 400.0
KUKA_VICTOR_HIGH_PD_CFG.actuators["victor_right_arm"].damping = 80.0
"""Configuration of Kuka iiwa with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
from omni.isaac.lab.utils import configclass
from typing import Literal
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv
    
@configclass
class Robotiq3FingerActionCfg(ActionTermCfg):
    """Configuration for the Robotiq 3-finger gripper action term
    """
    side: Literal["left", "right"] = "left"


class Robotiq3FingerAction(ActionTerm):
    """
    Class for the Robotiq 3-finger gripper action term.
    The commands are sent as two values from zero to one, one for the finger opening and one for the scissor opening.
    """
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    
    def __init__(self, cfg: Robotiq3FingerActionCfg, env: ManagerBasedEnv) ->None:
        # initialize the action term
        super().__init__(cfg, env)
        
        side = "left"
        joint_names = []
        for finger_name in ["a", "b", "c"]:
            for joint in range(1, 4):
                joint_name = f"victor_{side}_finger_{finger_name}_joint_{joint}"
                joint_names.append(joint_name)
        joint_names.append(f"victor_{side}_palm_finger_b_joint")
        joint_names.append(f"victor_{side}_palm_finger_c_joint")
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        
        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        # "victor_left_finger_a_joint_1": 0.890168571428571,
        # "victor_left_finger_a_joint_2": 0,
        # "victor_left_finger_a_joint_3": -0.8901685714285714,
        # "victor_left_finger_b_joint_1": 0.890168571428571,
        # "victor_left_finger_b_joint_2": 0,
        # "victor_left_finger_b_joint_3": -0.8901685714285714,
        # "victor_left_finger_c_joint_1": 0.890168571428571,
        # "victor_left_finger_c_joint_2": 0,
        # "victor_left_finger_c_joint_3": -0.8901685714285714,
        # # gripper scissors states
        # "victor_left_palm_finger_b_joint": 0.115940392156862,
        # "victor_left_palm_finger_c_joint": -0.11594039215686275,
    
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
        compute_finger_angles_jit(actions, self._processed_actions)
        # copy the joint angles to finger b and c
        self._processed_actions[:, 3:6] = self._processed_actions[:, 0:3]
        self._processed_actions[:, 6:9] = self._processed_actions[:, 0:3]
        
        # compute scissors angle
        compute_scissor_angle_jit(actions[:, 1], self._processed_actions[:, 9:11])
        
    
    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
    

@torch.jit.script
def compute_finger_angles_jit(control, finger_angles): 
    """Compute joint angles based on opening and scissor values 

    Args:
        control (torch.Tesnor): B x 2, representing the opening and scissor values. 0 corresponds to fully open, 1 is fully closed.
        
    """
    # Convert control input to g (range from 0 to 255)
    g_batch = control * 255

    max_angle = torch.tensor([70.0, 90.0, 43.0])
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
    finger_angles = torch.deg2rad(finger_angles)

    return finger_angles

@torch.jit.script
def compute_scissor_angle_jit(control, scissor_angle):
    # 0 corresponds to fully open at -16 degrees, 1 is fully closed with at +10 degrees
    scissor_angle[:, 0] = torch.deg2rad(16 - 26.0 * control)
    scissor_angle[:, 1] = -scissor_angle[:, 0]
