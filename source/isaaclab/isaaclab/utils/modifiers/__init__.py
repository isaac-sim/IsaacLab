# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated import path for tensor processors."""

import sys
import warnings
from types import ModuleType

from .. import tensor_processors

__all__ = [
    "ModifierCfg",
    "DigitalFilterCfg",
    "IntegratorCfg",
    "ModifierBase",
    "DigitalFilter",
    "Integrator",
    "bias",
    "clip",
    "scale",
]


def _install_aliases(module_name: str, processors: ModuleType) -> None:
    """Preserve legacy package and submodule imports without loading processor implementations."""
    module = sys.modules[module_name]
    renamed = {"ModifierBase": "TensorProcessor", "ModifierCfg": "TensorProcessorCfg"}

    def __getattr__(name: str):
        if name not in module.__all__:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        return getattr(processors, renamed.get(name, name))

    module.__getattr__ = __getattr__
    module.__dir__ = lambda: sorted(set(module.__dict__) | set(module.__all__))
    for name in ("modifier", "modifier_base", "modifier_cfg"):
        sys.modules[f"{module_name}.{name}"] = module
        setattr(module, name, module)


warnings.warn(
    "isaaclab.utils.modifiers, isaaclab_experimental.utils.modifiers, and ObservationTermCfg.modifiers are deprecated."
    " Use the corresponding tensor_processors APIs instead.",
    FutureWarning,
    stacklevel=2,
)

_install_aliases(__name__, tensor_processors)
