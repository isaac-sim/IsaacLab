# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Executed in a fresh process to reject accidental research/runtime imports."""

import importlib
import importlib.abc
import sys
from types import ModuleType

BLOCKED = (
    "ffcv",
    "wandb",
    "lightning",
    "pytorch_lightning",
    "timm",
    "fovi.models",
    "fovi.training",
    "isaacsim",
    "omni",
)


class BlockResearchImports(importlib.abc.MetaPathFinder):
    """Fail at the import boundary even when research packages are installed."""

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None = None,
        target: ModuleType | None = None,
    ) -> None:
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in BLOCKED):
            raise AssertionError(f"Standalone geometry imported forbidden dependency {fullname}")
        return None


sys.meta_path.insert(0, BlockResearchImports())
for module in (
    "isaaclab_fovi",
    "isaaclab_fovi.sensors",
    "isaaclab_fovi.sensors.fixed_software_fov",
    "isaaclab_fovi.sensors.active_camera_geometry",
    "isaaclab_fovi.sensors.foveated_rays",
    "isaaclab_fovi.sensors.pooling",
    "fovi.sensing.coords",
    "fovi.arch.knn",
):
    importlib.import_module(module)
print("Standalone import boundary passed")
