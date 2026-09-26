# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-specific action terms for LEAPP export workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
    OperationalSpaceControllerAction,
)
from isaaclab.utils.leapp.leapp_semantics import POSE6_ELEMENT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import (
        DeployDifferentialInverseKinematicsActionCfg,
        DeployOperationalSpaceControllerActionCfg,
        DeployRelativeJointPositionActionCfg,
    )


_LEAPP_TRACED_OBSERVATION_INPUTS = "_leapp_traced_observation_inputs"
_LEAPP_CONSUMED_OBSERVATION_INPUTS = "_leapp_consumed_observation_inputs"


def _leapp_real_env(env):
    real_env = object.__getattribute__(env, "_real_env") if type(env).__name__ == "_EnvProxy" else env
    return real_env


def _tensor_data_to_torch(data):
    """Return a torch tensor view for Isaac Lab data stored as torch or Warp-backed data."""
    return data.torch if hasattr(data, "torch") else wp.to_torch(data)


def _get_observation_term_from_buffer(env, group_name: str, term_name: str):
    """Return a term slice from the cached observation buffer."""
    obs_buffer = getattr(env, "obs_buf", None)
    if obs_buffer is None:
        obs_buffer = getattr(getattr(env, "observation_manager", None), "_obs_buffer", None)
    if not obs_buffer or group_name not in obs_buffer:
        return None

    group_obs = obs_buffer[group_name]
    if isinstance(group_obs, dict):
        return group_obs.get(term_name)

    obs_manager = getattr(env, "observation_manager", None)
    if obs_manager is None:
        return None

    term_names = obs_manager.active_terms.get(group_name, [])
    if term_name not in term_names:
        return None

    term_index = term_names.index(term_name)
    term_dims = obs_manager.group_obs_term_dim[group_name]
    concat_dim = obs_manager._group_obs_concatenate_dim[group_name]
    if concat_dim > 0:
        concat_dim -= 1

    start = sum(dim[concat_dim] for dim in term_dims[:term_index])
    length = term_dims[term_index][concat_dim]
    return group_obs.narrow(dim=concat_dim, start=start, length=length)


def _pop_leapp_traced_observation_input(env, name: str, *, group_name: str, term_name: str):
    """Consume one traced observation tensor for the current LEAPP action trace."""
    real_env = _leapp_real_env(env)
    consumed_inputs = getattr(real_env, _LEAPP_CONSUMED_OBSERVATION_INPUTS, None)
    if consumed_inputs is None:
        consumed_inputs = set()
        setattr(real_env, _LEAPP_CONSUMED_OBSERVATION_INPUTS, consumed_inputs)

    if name in consumed_inputs:
        return None

    traced_inputs = getattr(real_env, _LEAPP_TRACED_OBSERVATION_INPUTS, {})
    traced_tensor = traced_inputs.pop(name, None)
    if traced_tensor is None:
        traced_tensor = _get_observation_term_from_buffer(real_env, group_name, term_name)

    if traced_tensor is not None:
        consumed_inputs.add(name)
    return traced_tensor


def _is_leapp_observation_input_consumed(env, name: str) -> bool:
    real_env = _leapp_real_env(env)
    return name in getattr(real_env, _LEAPP_CONSUMED_OBSERVATION_INPUTS, set())


def _pose_rel_action_extra(*, position_scale: float, orientation_scale: float, target_types: Sequence[str]):
    return {
        "isaaclab_connection": "action:arm_action:pose_rel",
        "target_types": list(target_types),
        "position_scale": float(position_scale),
        "orientation_scale": float(orientation_scale),
    }


def _diffik_pose_rel_action_extra(scale):
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32).flatten()
    if scale_tensor.numel() == 1:
        scale_values = [float(scale_tensor.item())] * 6
    else:
        scale_values = [float(value) for value in scale_tensor.tolist()]

    extra = {
        "isaaclab_connection": "action:arm_action:pose_rel",
        "target_types": ["pose_rel"],
        "scale": scale_values,
    }
    if len(scale_values) >= 6:
        if len({round(value, 12) for value in scale_values[:3]}) == 1:
            extra["position_scale"] = scale_values[0]
        if len({round(value, 12) for value in scale_values[3:6]}) == 1:
            extra["orientation_scale"] = scale_values[3]
    return extra


class DeployRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative joint action that reuses traced current joint observations during LEAPP export."""

    def __init__(self, cfg: DeployRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        asset = self._asset
        if type(asset).__name__ == "_ArticulationWriteProxy":
            observation_input_name = f"{self.cfg.asset_name}_joint_pos"
            current_joint_pos = _pop_leapp_traced_observation_input(
                self._env,
                observation_input_name,
                group_name="policy",
                term_name="joint_pos",
            )
            if current_joint_pos is None:
                if not _is_leapp_observation_input_consumed(self._env, observation_input_name):
                    raise RuntimeError(
                        "DeployRelativeJointPositionAction requires the traced "
                        f"'{self.cfg.asset_name}_joint_pos' observation during LEAPP export."
                    )
                real_asset = object.__getattribute__(asset, "_real_asset")
                current_joint_pos = _tensor_data_to_torch(real_asset.data.joint_pos)[:, self._joint_ids]
        else:
            current_joint_pos = _tensor_data_to_torch(asset.data.joint_pos)[:, self._joint_ids]

        current_actions = self.processed_actions + current_joint_pos
        self._asset.set_joint_position_target_index(target=current_actions, joint_ids=self._joint_ids)


class DeployOperationalSpaceControllerAction(OperationalSpaceControllerAction):
    """OSC action that exports scaled pose_rel deltas during LEAPP export."""

    def __init__(self, cfg: DeployOperationalSpaceControllerActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._leapp_export_processed_actions = True
        self._leapp_processed_action_element_names = list(POSE6_ELEMENT_NAMES)
        self._leapp_processed_action_kind = None
        self._leapp_processed_action_extra = _pose_rel_action_extra(
            position_scale=cfg.position_scale,
            orientation_scale=cfg.orientation_scale,
            target_types=cfg.controller_cfg.target_types,
        )

    def apply_actions(self):
        asset = self._asset
        if type(asset).__name__ != "_ArticulationWriteProxy":
            super().apply_actions()
            return

        real_asset = object.__getattribute__(asset, "_real_asset")
        self._asset = real_asset
        try:
            super().apply_actions()
        finally:
            self._asset = asset


class DeployDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """DiffIK action that exports scaled relative TCP pose deltas during LEAPP export."""

    def __init__(self, cfg: DeployDifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._leapp_export_processed_actions = True
        self._leapp_processed_action_element_names = list(POSE6_ELEMENT_NAMES)
        self._leapp_processed_action_kind = None
        self._leapp_processed_action_extra = _diffik_pose_rel_action_extra(cfg.scale)

    def apply_actions(self):
        asset = self._asset
        if type(asset).__name__ != "_ArticulationWriteProxy":
            super().apply_actions()
            return

        real_asset = object.__getattribute__(asset, "_real_asset")
        self._asset = real_asset
        try:
            super().apply_actions()
        finally:
            self._asset = asset
