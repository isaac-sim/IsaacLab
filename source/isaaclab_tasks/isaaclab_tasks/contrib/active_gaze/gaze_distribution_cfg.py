# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of :class:`~isaaclab_tasks.contrib.active_gaze.gaze_distribution.GazeGaussianDistribution`."""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class GazeGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Gaussian action head whose gaze actions, the last two, have a fixed standard deviation.

    ``init_std``, ``std_type`` and ``std_range`` apply to the other actions.
    """

    class_name: str = "isaaclab_tasks.contrib.active_gaze.gaze_distribution:GazeGaussianDistribution"
    gaze_std: float = 0.1
    """Fixed standard deviation of the gaze actions. One action moves the gaze by the gaze action's ``step_size``."""
