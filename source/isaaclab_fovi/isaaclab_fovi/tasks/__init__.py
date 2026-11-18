# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera tasks of Isaac Lab with a foveated camera whose gaze the policy moves.

Each task keeps its stock counterpart's scene, rewards, vision model and PPO settings. The camera becomes a foveated
camera at the same mount, with the same optics and resolution, and the policy gets two gaze actions. A teacher's gaze
labels supervise those actions during training.
"""

from . import cartpole, kuka_allegro  # noqa: F401 (registers the tasks)
