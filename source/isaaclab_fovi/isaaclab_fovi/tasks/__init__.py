# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera tasks of Isaac Lab with a foveated camera whose gaze the policy moves.

Each task keeps its stock counterpart's scene, vision model and PPO settings. The camera becomes a foveated camera at
the same mount, with the same optics and, except for Cartpole, the same resolution: Cartpole's retina is 32 samples
across instead of 96 pixels. The policy gets two gaze actions, whose noise is fixed, and a teacher's gaze labels
supervise them during training. The Kuka tasks' action penalty leaves out the gaze, so moving it is free.
"""

from . import cartpole, kuka_allegro  # noqa: F401 (registers the tasks)
