# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural dynamics model definitions, independent optimizer settings and Newton solver integration.

NeRDCfg describes a model. NeRDTrainingCfg describes optimization. Recording environments declare their own
exploration actions, reset events and RecorderManager terms. The recording and training scripts own task composition
and import the optional newton-neural library only when executed; configuration imports remain independent of it.
"""

import importlib.metadata

from isaaclab.utils.module import lazy_export

try:
    __version__ = importlib.metadata.version("isaaclab_neural")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

lazy_export()
