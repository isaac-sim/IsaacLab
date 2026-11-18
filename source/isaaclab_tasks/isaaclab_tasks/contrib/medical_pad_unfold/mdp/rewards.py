# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward of the medical-pad task: progress of a frame classifier over the generated frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .classifier_tracker import get_tracker

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def classifier_progress(
    env: ManagerBasedRLEnv, classifier: dict, sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera")
) -> torch.Tensor:
    """Sum of the per-frame increases of the classifier's expected reward over a generated chunk.

    The world model produces a chunk of frames at once, so the reward is paid in the step that receives them and is
    zero in the steps in between. Frames after the one at which the episode succeeds or fails are not counted. Within
    a chunk the increments telescope to the last counted frame's score minus the previous chunk's last score (zero at
    a reset). The reward manager multiplies a term by the step duration; give the term a weight of ``1 / env.step_dt``
    so the total return equals the sum of the classifier increments.

    Args:
        env: The environment.
        classifier: Arguments of :class:`ClassifierTracker`; the same for the reward and the terminations.
        sensor_cfg: The camera whose frame history is scored.
    """
    return get_tracker(env, classifier, sensor_cfg).reward
