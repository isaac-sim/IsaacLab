# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with implementation of manager terms.

The functions can be provided to different managers that are responsible for the
different aspects of the MDP. These include the observation, reward, termination,
actions, events and curriculum managers.

The terms are defined under the ``envs`` module because they are used to define
the environment. However, they are not part of the environment directly, but
are used to define the environment through their managers.

"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .actions import *  # noqa: F403
    from .commands import *  # noqa: F403
    from .curriculums import *  # noqa: F403
    from .events import *  # noqa: F403
    from .observations import *  # noqa: F403
    from .recorders import *  # noqa: F403
    from .rewards import *  # noqa: F403
    from .terminations import *  # noqa: F403

from isaaclab.utils.module import cascading_export

cascading_export(
    submodules=[
        "actions",
        "commands",
        "curriculums",
        "events",
        "observations",
        "recorders",
        "rewards",
        "terminations",
    ],
)
