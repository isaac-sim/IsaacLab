# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the conveyor-to-conveyor Franka transfer task."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .actions import ConveyorRelativeJointPositionAction, ResetBufferedGripperAction
from .actions_cfg import ConveyorRelativeJointPositionActionCfg, ResetBufferedGripperActionCfg
from .curriculums import ConveyorResetCurriculum
from .observations import (
    active_transfer_features,
    cube_conveyor_state,
    end_effector_axes,
    end_effector_velocity,
    gripper_joint_positions,
    target_cube_one_hot,
    target_side_one_hot,
    transfer_object_observation,
)
from .reset_events import (
    BELT_DEPLOYMENT_VARIANT,
    ConveyorResetRecipe,
    ConveyorResetStateTable,
    advance_conveyor_transfer_goal,
    build_reset_rows,
    select_next_transfer_cube,
)
from .rewards import (
    ConveyorTransferProgressReward,
    action_term_l2,
    finite_joint_velocity_l2,
    physical_cube_acquisition_mask,
    terminal_failure,
    transfer_success_reward,
)
from .state import ConveyorTransferState
from .terminations import (
    ConveyorResetLearningProgress,
    StableConveyorTransfer,
    cube_out_of_workspace,
    nonfinite_scene_state,
    subgoal_time_out,
    transfer_sequence_time_out,
)
