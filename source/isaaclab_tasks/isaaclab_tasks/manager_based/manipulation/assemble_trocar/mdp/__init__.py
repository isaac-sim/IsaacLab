# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MDP utilities for the assemble_trocar task."""

from isaaclab.envs.mdp import JointPositionActionCfg, time_out

from .events import reset_robot_to_default_joint_positions, reset_task_stage, reset_tray_with_random_rotation
from .observations import get_robot_body_joint_states, get_robot_dex3_joint_states
from .rewards import (
    lift_trocars_reward,
    trocar_insertion_reward,
    trocar_placement_reward,
    trocar_tip_alignment_reward,
    update_task_stage,
)
from .terminations import object_drop_termination, task_success_termination

__all__ = [
    "JointPositionActionCfg",
    "time_out",
    "get_robot_body_joint_states",
    "get_robot_dex3_joint_states",
    "reset_tray_with_random_rotation",
    "reset_robot_to_default_joint_positions",
    "reset_task_stage",
    "update_task_stage",
    "lift_trocars_reward",
    "trocar_tip_alignment_reward",
    "trocar_insertion_reward",
    "trocar_placement_reward",
    "task_success_termination",
    "object_drop_termination",
]
