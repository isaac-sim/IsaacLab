# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the fourbar-pole swing-up environments.

This module holds both observation helpers (``joint_pos_cos`` / ``joint_pos_sin``)
that encode an angle without the ``+-pi`` wrap discontinuity, and the swing-up
reward / success-metric terms.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_apply

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_cos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Cosine of the selected joint positions.

    Encodes the angle without the wrap-around discontinuity at ``+-pi`` so the
    policy sees a smooth signal as the pole swings through the bottom.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.cos(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])


def joint_pos_sin(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sine of the selected joint positions (companion to :func:`joint_pos_cos`)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sin(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])


def pole_upright(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Uprightness reward for the pole, maximal (``+1``) when fully upright.

    The pole joint is zero when upright and ``+-pi`` when hanging, so ``cos`` gives
    a dense shaping signal in ``[-1, 1]`` that drives the swing-up.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.cos(asset.data.joint_pos.torch[:, asset_cfg.joint_ids]), dim=1)


class UprightSuccessRateCommand(CommandTerm):
    """Command term that tracks pole-upright terminal success as a metric."""

    cfg: UprightSuccessRateCommandCfg

    def __init__(self, cfg: UprightSuccessRateCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset_cfg = cfg.asset_cfg
        self._asset_cfg.resolve(env.scene)
        self._asset: Articulation = env.scene[self._asset_cfg.name]

        self._command = torch.zeros((self.num_envs, 1), device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        pole_pos = self._asset.data.joint_pos.torch[:, self._asset_cfg.joint_ids]
        self.metrics["success_rate"] = (torch.cos(pole_pos) > self.cfg.threshold).all(dim=1).float()

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass


@configclass
class UprightSuccessRateCommandCfg(CommandTermCfg):
    """Configuration for :class:`UprightSuccessRateCommand`."""

    class_type: type[UprightSuccessRateCommand] | str = "{DIR}.rewards:UprightSuccessRateCommand"
    resampling_time_range: tuple[float, float] = (1e6, 1e6)

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["coupler_to_pole"])
    threshold: float = 0.95


class LoopClosureErrorCommand(CommandTerm):
    """Command term that monitors four-bar kinematic consistency.

    Loop closure: the four-bar articulation is an open tree whose loop is closed by the
    ``rocker_to_ground`` joint (excluded from the articulation). That joint's two
    anchors must coincide in world space:

    - rocker anchor (``localPos0``) on body ``rocker``
    - ground anchor (``localPos1``) on body ``ground_link``

    Ground link pose: ``ground_link`` is fixed to the world via ``root_joint`` and should
    remain at the articulation spawn pose (``expected_pos`` / ``expected_quat`` relative to
    each env origin). Position and orientation errors are reported separately.
    """

    cfg: LoopClosureErrorCommandCfg

    def __init__(self, cfg: LoopClosureErrorCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset_cfg = cfg.asset_cfg
        self._asset_cfg.resolve(env.scene)
        self._asset: Articulation = env.scene[self._asset_cfg.name]

        # Resolve body indices from the Newton model's physical body order
        # (``model.body_label``), which is what ``body_link_pos_w`` is indexed by.
        # The view's ``body_names`` is joint-child ordered and contains a duplicate
        # for the closed-loop joint (``rocker_to_ground``), so ``body_names.index()``
        # would point at the spurious phantom row instead of the real body.
        model_body_labels = [str(label).split("/")[-1] for label in self._asset.root_view.model.body_label]
        self._rocker_body_id = model_body_labels.index(cfg.rocker_body_name)
        self._ground_body_id = model_body_labels.index(cfg.ground_body_name)

        self._rocker_anchor = torch.tensor(cfg.rocker_anchor, device=self.device).expand(self.num_envs, 3)
        self._ground_anchor = torch.tensor(cfg.ground_anchor, device=self.device).expand(self.num_envs, 3)

        self._command = torch.zeros((self.num_envs, 1), device=self.device)
        self.metrics["loop_closure_pos_error_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["loop_closure_pos_error_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ground_link_pos_error_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ground_link_pos_error_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ground_link_ori_error_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ground_link_ori_error_max"] = torch.zeros(self.num_envs, device=self.device)
        self._kinematic_metric_step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self._expected_pos_b = torch.tensor(cfg.expected_pos, device=self.device).expand(self.num_envs, 3)
        self._expected_quat_b = torch.tensor(cfg.expected_quat, device=self.device).expand(self.num_envs, 4)
        self._env_origin_quat_w = torch.tensor((0.0, 0.0, 0.0, 1.0), device=self.device).expand(self.num_envs, 4)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        pos_w = self._asset.data.body_link_pos_w.torch
        quat_w = self._asset.data.body_link_quat_w.torch

        rocker_world = pos_w[:, self._rocker_body_id] + quat_apply(quat_w[:, self._rocker_body_id], self._rocker_anchor)
        ground_world = pos_w[:, self._ground_body_id] + quat_apply(quat_w[:, self._ground_body_id], self._ground_anchor)
        loop_err = torch.linalg.norm(rocker_world - ground_world, dim=-1)

        expected_pos_w, expected_quat_w = combine_frame_transforms(
            self._env.scene.env_origins,
            self._env_origin_quat_w,
            self._expected_pos_b,
            self._expected_quat_b,
        )
        ground_pos_err, ground_rot_err = compute_pose_error(
            pos_w[:, self._ground_body_id],
            quat_w[:, self._ground_body_id],
            expected_pos_w,
            expected_quat_w,
        )
        ground_pos_err = torch.linalg.norm(ground_pos_err, dim=-1)
        ground_ori_err = torch.linalg.norm(ground_rot_err, dim=-1)

        self._kinematic_metric_step_count += 1
        n = self._kinematic_metric_step_count.float()
        self.metrics["loop_closure_pos_error_mean"] += (loop_err - self.metrics["loop_closure_pos_error_mean"]) / n
        self.metrics["loop_closure_pos_error_max"] = torch.maximum(self.metrics["loop_closure_pos_error_max"], loop_err)
        self.metrics["ground_link_pos_error_mean"] += (
            ground_pos_err - self.metrics["ground_link_pos_error_mean"]
        ) / n
        self.metrics["ground_link_pos_error_max"] = torch.maximum(
            self.metrics["ground_link_pos_error_max"], ground_pos_err
        )
        self.metrics["ground_link_ori_error_mean"] += (
            ground_ori_err - self.metrics["ground_link_ori_error_mean"]
        ) / n
        self.metrics["ground_link_ori_error_max"] = torch.maximum(
            self.metrics["ground_link_ori_error_max"], ground_ori_err
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            self._kinematic_metric_step_count.zero_()
        else:
            self._kinematic_metric_step_count[env_ids] = 0
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass


@configclass
class LoopClosureErrorCommandCfg(CommandTermCfg):
    """Configuration for :class:`LoopClosureErrorCommand`."""

    class_type: type[LoopClosureErrorCommand] | str = "{DIR}.rewards:LoopClosureErrorCommand"
    resampling_time_range: tuple[float, float] = (1e6, 1e6)

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["rocker", "ground_link"])
    """Asset config resolving the loop-joint bodies, in order ``[rocker, ground_link]``."""

    rocker_body_name: str = "rocker"
    """Name of the ``rocker`` body carrying the loop-joint ``localPos0`` anchor."""

    ground_body_name: str = "ground_link"
    """Name of the ``ground_link`` body carrying the loop-joint ``localPos1`` anchor."""

    rocker_anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Loop-joint anchor on the ``rocker`` body (USD ``localPos0``)."""

    ground_anchor: tuple[float, float, float] = (0.0, 0.2, 0.0)
    """Loop-joint anchor on the ``ground_link`` body (USD ``localPos1``)."""

    expected_pos: tuple[float, float, float] = (0.0, 0.0, 1.5)
    """Expected ``ground_link`` position relative to each env origin (articulation spawn pose)."""

    expected_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Expected ``ground_link`` orientation ``(x, y, z, w)`` relative to each env origin."""
