from dataclasses import MISSING

from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from .joint_actions import DelayedJointPositionAction

@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = DelayedJointPositionAction
    delay_update_global_steps: int  =  24 * 8000
    history_length: int = 8
    action_delay_steps: list[int]| int = [1, 1]
    use_delay: bool = False 
