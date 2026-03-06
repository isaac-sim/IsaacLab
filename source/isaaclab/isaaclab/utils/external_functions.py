# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for utilities for working with external functions."""

from importlib import import_module
from typing import Any


def call_externally_defined_function(function_path: str, *args: Any, **kwargs: Any) -> Any:
    # Dynamically import the function from the path and call it.
    module_path, fn_name = function_path.rsplit(".", 1)
    try:
        module = import_module(module_path)
        fn = getattr(module, fn_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Failed to find the requested externally defined module {module_path}.") from e
    except AttributeError as e:
        raise AttributeError(
            f"Failed to find the requested externally defined function {fn_name} in module {module_path}."
        ) from e
    return fn(*args, **kwargs)
