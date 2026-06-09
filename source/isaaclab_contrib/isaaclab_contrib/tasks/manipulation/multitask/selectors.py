# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selector functions for multitask manipulation scene configs."""

from __future__ import annotations


def asset_names(asset_cfgs: dict[str, object], names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Select scene asset names from an explicit list."""
    return tuple(name for name in names if name in asset_cfgs)
