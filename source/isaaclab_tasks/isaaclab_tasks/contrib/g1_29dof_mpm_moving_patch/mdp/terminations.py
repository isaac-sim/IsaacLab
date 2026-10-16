# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms specific to the granular-bed locomotion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ..mpm_env import G1MovingPatchEnv


def root_outside_workspace(
    env: G1MovingPatchEnv, margin: float = 0.3, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate outside terrain XY bounds inset by boundary width plus margin [m]."""
    bounds = env.scene.terrain.background_mesh.bounds  # type: ignore
    root = env.scene[asset_cfg.name].data.root_pos_w.torch[:, :2]
    pad = env.cfg.scene.terrain.moving_patch_terrain.boundary_terrain_size + margin
    lo = torch.tensor(bounds[0, :2] + pad, device=env.device)
    hi = torch.tensor(bounds[1, :2] - pad, device=env.device)
    return ((root < lo) | (root > hi)).any(dim=1)


def root_state_infinite(env: G1MovingPatchEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the coupled solve left the base state non-finite.

    A diverging granular solve turns a whole world's particle and body state into ``NaN``. Every
    other termination term compares against a threshold, and comparisons with ``NaN`` are false, so
    without this term the world stays broken and keeps feeding ``NaN`` observations until the
    episode times out.

    Args:
        env: Environment instance.
        asset_cfg: Configuration of the tracked articulation.

    Returns:
        Whether the base state is non-finite, shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return ~torch.isfinite(asset.data.root_state_w.torch).all(dim=-1)


def root_height_below_minimum_adaptive(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    min_foot_height = (asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]).min(dim=1).values

    return asset.data.root_pos_w.torch[:, 2] - min_foot_height < minimum_height


class ExtremeJointPositionAction(ManagerTermBase):
    """Terminate when the policy's commanded position target would saturate a joint's
    PD torque well past its effort limit if the joint were sitting at that limit.

    For a position-controlled joint with PD gain ``kp``, joint range ``[q_min, q_max]``,
    and effort limit ``tau_max``, the spring torque applied if the joint is at ``q_max``
    and the target is ``q_des`` is ``kp * (q_des - q_max)``. The action is considered
    "too extreme into the upper limit" when this would exceed ``torque_fraction * tau_max``:

        q_des > q_max + torque_fraction * tau_max / kp

    and symmetrically for the lower limit:

        q_des < q_min - torque_fraction * tau_max / kp
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        action_name: str = cfg.params["action_name"]  # type: ignore
        torque_fraction: float = cfg.params.get("torque_fraction", 10.0)  # type: ignore

        self.action_term = env.action_manager.get_term(action_name)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.joint_ids = cfg.params["asset_cfg"].joint_ids
        self.torque_fraction = torque_fraction

        # Joint ids in articulation space, one per action dimension.
        action_joint_ids = self._resolve_joint_ids(self.joint_ids, self.asset.num_joints)
        action_dim = len(action_joint_ids)

        # Snapshot kp and tau_max from the owning actuators. PD gains and effort limits
        # are configured at startup and not currently randomized; if that changes, this
        # cache will need to be refreshed.
        stiffness = torch.zeros(self.num_envs, action_dim, device=self.device)
        effort_limit = torch.zeros_like(stiffness)
        action_covered = [False] * action_dim
        for actuator in self.asset.actuators.values():
            actuator_joint_ids = self._resolve_joint_ids(actuator.joint_indices, self.asset.num_joints)
            actuator_by_joint = {joint_id: actuator_id for actuator_id, joint_id in enumerate(actuator_joint_ids)}
            # Create a list of the actuator joint ids for the current actuator that correspond to an field in the action
            action_ids: list[int] = []
            actuator_ids: list[int] = []
            for action_id, joint_id in enumerate(action_joint_ids):
                if joint_id in actuator_by_joint:
                    action_ids.append(action_id)
                    actuator_ids.append(actuator_by_joint[joint_id])
                    action_covered[action_id] = True
            if action_ids:
                action_ids_tensor = torch.tensor(action_ids, device=self.device, dtype=torch.long)
                actuator_ids_tensor = torch.tensor(actuator_ids, device=self.device, dtype=torch.long)
                stiffness[:, action_ids_tensor] = actuator.stiffness[:, actuator_ids_tensor]
                effort_limit[:, action_ids_tensor] = actuator.effort_limit[:, actuator_ids_tensor]

        if not all(action_covered):
            missing = [
                self.asset.joint_names[action_joint_ids[action_id]]
                for action_id, covered in enumerate(action_covered)
                if not covered
            ]
            raise RuntimeError(
                f"ExtremeJointPositionAction: action joints {missing} have no owning actuator on"
                f" asset '{self.asset.cfg.prim_path}'."
            )

        active = stiffness > 0
        margin = self.torque_fraction * effort_limit / torch.where(active, stiffness, torch.ones_like(stiffness))
        self._margin = torch.where(active, margin, torch.full_like(margin, float("inf")))

    @staticmethod
    def _resolve_joint_ids(joint_ids, total: int) -> list[int]:
        if isinstance(joint_ids, slice):
            return list(range(total))[joint_ids]
        if isinstance(joint_ids, torch.Tensor):
            return joint_ids.tolist()
        return list(joint_ids)

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, action_name: str, torque_fraction: float = 10.0
    ) -> torch.Tensor:
        target = self.action_term.processed_actions
        pos_limits = self.asset.data.joint_pos_limits[:, self.joint_ids]
        q_min = pos_limits[..., 0]
        q_max = pos_limits[..., 1]

        too_high = target > q_max + self._margin
        too_low = target < q_min - self._margin
        return (too_high | too_low).any(dim=1)
