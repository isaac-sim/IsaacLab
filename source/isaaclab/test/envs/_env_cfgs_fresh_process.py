# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fresh-process validation helper for shared environment configurations."""

from __future__ import annotations

import sys

from isaaclab.test.env_cfgs import (
    make_empty_direct_marl_env_cfg,
    make_empty_manager_based_env_cfg,
    make_empty_manager_based_rl_env_cfg,
)

_ENV_CFG_FACTORIES = (
    make_empty_manager_based_env_cfg,
    make_empty_manager_based_rl_env_cfg,
    make_empty_direct_marl_env_cfg,
)

_FORBIDDEN_ROOTS = {
    "carb",
    "isaaclab_assets",
    "isaaclab_tasks",
    "isaaclab_tasks_experimental",
    "isaacsim",
    "omni",
    "pxr",
    "scipy",
}

_FORBIDDEN_PREFIXES = (
    "isaaclab.actuators.actuator_base_cfg",
    "isaaclab.actuators.actuator_pd_cfg",
    "isaaclab.assets",
    "isaaclab.sim.spawners.from_files",
    "isaaclab.sim.spawners.shapes",
    "isaaclab.utils.assets",
    "isaaclab_physx.sim.schemas",
)


def main() -> None:
    """Construct and validate every shared configuration without runtime imports."""
    for factory in _ENV_CFG_FACTORIES:
        cfg = factory(device="cpu", num_envs=2, env_spacing=1.5)
        validate = getattr(cfg, "validate")
        validate()
        assert cfg.sim.device == "cpu"
        assert cfg.scene.num_envs == 2
        assert cfg.scene.env_spacing == 1.5

    forbidden_modules = sorted(
        module_name
        for module_name in sys.modules
        if module_name.partition(".")[0] in _FORBIDDEN_ROOTS
        or any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in _FORBIDDEN_PREFIXES
        )
    )
    assert not forbidden_modules, f"Forbidden modules loaded: {forbidden_modules}"


if __name__ == "__main__":
    main()
