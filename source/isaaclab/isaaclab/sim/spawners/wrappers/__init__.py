# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for wrapping spawner configurations.

Unlike the other spawner modules, this module provides a way to wrap multiple spawner configurations
into a single configuration. This is useful when the user wants to spawn multiple assets based on
different configurations.
"""

from .wrappers import spawn_multi_asset, spawn_multi_usd_file
from .wrappers_cfg import MultiAssetSpawnerCfg, MultiUsdFileCfg
