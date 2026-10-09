# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for loading optional OvPhysX runtime modules and schema resources."""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path
from types import ModuleType

_OVPHYSX_INSTALL_MESSAGE = (
    "The OvPhysX backend requires the optional 'ovphysx' runtime wheel, which is not installed. "
    "Run your command with: uv run --extra ovphysx <command> "
    "(or, manually: python -m pip install --extra-index-url https://pypi.nvidia.com ovphysx)."
)


def physx_schema_paths() -> list[Path]:
    """Find the external PhysX schema resources without importing USD or the provider.

    Raises:
        FileNotFoundError: If ``physx-usd-schemas`` or one of its resources is missing.
    """
    try:
        package = importlib.metadata.distribution("physx-usd-schemas")
    except importlib.metadata.PackageNotFoundError as exc:
        raise FileNotFoundError(
            "PhysX schemas are not installed. Install the 'ovphysx' extra or physx-usd-schemas."
        ) from exc
    paths = [
        Path(package.locate_file(f"physx_usd_schemas/{module}/resources"))
        for module in ("PhysxSchema", "OmniUsdPhysicsDeformableSchema")
    ]
    for path in paths:
        for filename in ("plugInfo.json", "generatedSchema.usda"):
            if not (path / filename).is_file():
                raise FileNotFoundError(
                    f"Incomplete physx-usd-schemas: missing {path / filename}. Reinstall the package."
                )
    return paths


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
