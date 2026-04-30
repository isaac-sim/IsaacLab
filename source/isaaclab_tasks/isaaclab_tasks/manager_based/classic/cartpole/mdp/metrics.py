# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def log_pole_upright_success_rate(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.5,
) -> None:
    """Log ``Metrics/success_rate`` as the fraction of envs whose pole is within ``threshold`` rad of upright.

    Intended as an interval-mode event term so the metric lands in ``env.extras["log"]`` every step.
    The tag matches the universal convention used by the benchmark success-metric pipeline.
    ``env_ids`` is ignored on purpose — the metric is a global mean across all envs.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pole_angle = wrap_to_pi(asset.data.joint_pos.torch[:, asset_cfg.joint_ids[0]])
    env.extras["log"]["Metrics/success_rate"] = (pole_angle.abs() < threshold).float().mean()
