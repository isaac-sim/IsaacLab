# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the OvPhysX backend."""

from __future__ import annotations

import os
import tempfile
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .ovphysx_backend import OvPhysxBackend

# POSIX temp roots are shared between users; Windows already gives each user a private one.
_CACHE_DIR_NAME = f"ovphysx_derived_data_cache_{os.getuid()}" if hasattr(os, "getuid") else "ovphysx_derived_data_cache"

DEFAULT_COOKED_COLLIDER_CACHE_DIR: str = os.path.join(tempfile.gettempdir(), _CACHE_DIR_NAME)
"""Per-user cooked-collider cache shared by physics and native runtime configuration defaults."""


@configclass
class OvPhysxBackendCfg(BackendCfg):
    """Native runtime construction settings, separate from physics scene policy."""

    class_type: type[OvPhysxBackend] | str = "{DIR}.ovphysx_backend:OvPhysxBackend"

    device: str = MISSING
    """Normalized device: ``cpu`` or ``cuda:<index>``."""

    cooked_collider_cache_dir: str | None = DEFAULT_COOKED_COLLIDER_CACHE_DIR
    """Cooked-collider cache directory; every backend in a process must use the same one."""
