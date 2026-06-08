# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for loading optional OvPhysX runtime modules."""

from __future__ import annotations

import importlib
from types import ModuleType

_OVPHYSX_INSTALL_MESSAGE = (
    "The OvPhysX backend requires the optional 'ovphysx' runtime wheel, which is not installed. "
    "Install it with: ./isaaclab.sh -i 'ov[ovphysx]' "
    "(or, manually: pip install --extra-index-url https://pypi.nvidia.com "
    "-e 'source/isaaclab_ovphysx[ovphysx]')."
)


def import_ovphysx(module_name: str = "ovphysx") -> ModuleType:
    """Import an optional ``ovphysx`` runtime module with an actionable install error.

    Args:
        module_name: Name of the ``ovphysx`` module to import.

    Returns:
        The imported runtime module.

    Raises:
        ModuleNotFoundError: If the optional ``ovphysx`` runtime wheel is not installed.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != "ovphysx":
            raise
        raise ModuleNotFoundError(_OVPHYSX_INSTALL_MESSAGE, name="ovphysx") from exc
