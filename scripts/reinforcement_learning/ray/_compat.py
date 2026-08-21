# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility support for the deprecated Ray script entry points."""

import importlib
import runpy
import warnings


def forward(module_name: str, namespace: dict[str, object]) -> None:
    """Forward a legacy script or module import to its contributed replacement.

    Args:
        module_name: Fully qualified replacement module name.
        namespace: Legacy module globals to populate when imported.
    """
    legacy_path = namespace.get("__file__", "The legacy Ray script")
    warnings.warn(
        f"{legacy_path} is deprecated; use `python -m {module_name}` instead.",
        FutureWarning,
        stacklevel=2,
    )
    if namespace.get("__name__") == "__main__":
        runpy.run_module(module_name, run_name="__main__")
        return

    module = importlib.import_module(module_name)
    exported_names = getattr(module, "__all__", [name for name in dir(module) if not name.startswith("_")])
    namespace.update({name: getattr(module, name) for name in exported_names})
