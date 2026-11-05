# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for an input source activated by interactive playback."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .source import TeleopSource


@configclass
class TeleopSourceCfg:
    """Construction of an input source, independent of action processing."""

    class_type: type[TeleopSource] | str = MISSING
    """Source constructor, called with ``(cfg, env, *, env_ids)``.

    Use a lazy ``"package.module:Class"`` reference to keep device dependencies
    out of task configuration and training imports. Capture sources additionally
    accept the playback owner's optional ``xr_session``.
    """
