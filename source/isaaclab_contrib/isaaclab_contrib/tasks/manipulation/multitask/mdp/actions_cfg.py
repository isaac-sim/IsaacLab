# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action term configurations for multi-task environments."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .actions import ScatteredActionTerm


@configclass
class ScatteredActionTermCfg(ActionTermCfg):
    """Groups multiple :class:`ActionTermCfg` terms that share the same action dimension.

    The policy outputs one set of actions and the grouped term broadcasts
    them to every sub-term.  Each sub-term applies to its own asset/group.

    All terms **must** have the same ``action_dim``;
    unequal dimensions raise :class:`ValueError` at init time.
    """

    class_type: type[ScatteredActionTerm] | str = "{DIR}.actions:ScatteredActionTerm"

    asset_name: str = "__grouped__"

    terms: list[ActionTermCfg] = MISSING
    """Action term configs that share a dimension.  Each term carries its own ``asset_name``."""

    dim: int | None = None
    """Fallback action dimension when all sub-terms are disabled.

    When set and ``terms`` is empty (or every sub-term's asset is
    unregistered), the term still reports this many action dimensions
    but acts as a no-op.  Leave ``None`` to require at least one active
    sub-term.
    """
