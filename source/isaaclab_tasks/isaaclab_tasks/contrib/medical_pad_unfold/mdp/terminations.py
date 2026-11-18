# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terminations of the medical-pad task: the classifier judges the episode a success or a failure."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .classifier_tracker import get_tracker

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def classifier_success(
    env: ManagerBasedRLEnv, classifier: dict, sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera")
) -> torch.Tensor:
    """Environments whose episode succeeded in the chunk that completed in this step.

    The classifier judges frame by frame, but the result is reported when the chunk completes, which is where
    environments can reset, so this term fires only on chunk boundaries.
    """
    return get_tracker(env, classifier, sensor_cfg).success


def classifier_failure(
    env: ManagerBasedRLEnv, classifier: dict, sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera")
) -> torch.Tensor:
    """Environments whose episode failed in the chunk that completed in this step. See :func:`classifier_success`."""
    return get_tracker(env, classifier, sensor_cfg).failure
