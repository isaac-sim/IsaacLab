# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing randomizations that are applied at the time of spawning prims.
Note that this is different from the randomizations that are applied at the time of simulation reset, and allows
to modify properties such as scale, joint offsets, etc. at the time of spawning the prims.
"""

from .randomizations import *  # noqa: F401, F403
from .randomizations_cfg import *  # noqa: F401, F403
