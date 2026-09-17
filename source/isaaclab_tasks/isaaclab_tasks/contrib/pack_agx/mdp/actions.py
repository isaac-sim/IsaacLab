# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for the H2 AGX Orin packing task."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class H2GravityCompensatedJointPositionAction(JointPositionAction):
    """Absolute joint-position action with PhysX gravity feed-forward on both arms."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._gravity_joint_ids, _ = self._asset.find_joints(
            [
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ]
        )
        self._gravity_columns = (
            torch.as_tensor(self._gravity_joint_ids, device=self.device, dtype=torch.long) + self._asset.num_base_dofs
        )

    def apply_actions(self) -> None:
        if not self._asset.cfg.spawn.rigid_props.disable_gravity:
            gravity = self._asset.data.gravity_compensation_forces.torch[:, self._gravity_columns]
            self._asset.set_joint_effort_target_index(
                target=gravity,
                joint_ids=self._gravity_joint_ids,
            )
        super().apply_actions()
