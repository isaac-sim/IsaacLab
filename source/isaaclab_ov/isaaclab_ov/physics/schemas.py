# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""External PhysX schema resources without importing their USD plugin provider."""

import importlib.metadata
from pathlib import Path


def physx_schema_paths() -> list[Path]:
    """Locate the external PhysX schemas without importing the external USD plugin provider.

    Returns:
        Resource directories for ``PhysxSchema`` and ``OmniUsdPhysicsDeformableSchema``, in that order.

    Raises:
        importlib.metadata.PackageNotFoundError: If ``physx-usd-schemas`` is not installed.
        FileNotFoundError: If a resource directory lacks ``plugInfo.json`` or ``generatedSchema.usda``.
    """
    package = importlib.metadata.distribution("physx-usd-schemas")
    paths = [
        Path(package.locate_file(f"physx_usd_schemas/{module}/resources"))
        for module in ("PhysxSchema", "OmniUsdPhysicsDeformableSchema")
    ]
    for path in paths:
        for name in ("plugInfo.json", "generatedSchema.usda"):
            if not (path / name).is_file():
                raise FileNotFoundError(f"Incomplete physx-usd-schemas: missing {path / name}. Reinstall the provider.")
    return paths
