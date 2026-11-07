# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the custom coupling environment configuration."""

import pytest
from isaaclab_newton.physics.coupling import CouplerProxyCfg

import isaaclab_contrib.custom_coupling.tasks  # noqa: F401

from isaaclab_tasks.utils import resolve_task_config

pytestmark = pytest.mark.kitless


def test_example_default_and_legacy_presets_use_proxy_coupling() -> None:
    """The legacy task warns and resolves both preset names to the maintained proxy coupler."""
    with pytest.warns(FutureWarning, match="deprecated"):
        env_cfg, _ = resolve_task_config("IsaacContrib-Lift-Soft-Franka-Custom-Coupling", "", overrides=())

    assert isinstance(env_cfg.sim.physics.solver_cfg, CouplerProxyCfg)

    # Keep the legacy preset selectable while routing it through generic coupling.
    env_cfg, _ = resolve_task_config(
        "IsaacContrib-Lift-Soft-Franka-Custom-Coupling", "", overrides=("physics=newton_mjwarp_vbd",)
    )

    assert isinstance(env_cfg.sim.physics.solver_cfg, CouplerProxyCfg)


def test_core_task_rejects_the_removed_preset_name() -> None:
    """Selecting the moved name on a core task must fail loudly, not fall back."""
    with pytest.raises(ValueError, match="newton_mjwarp_vbd"):
        resolve_task_config(
            "Isaac-Lift-Soft-Franka", "rsl_rl_cfg_entry_point", overrides=("physics=newton_mjwarp_vbd",)
        )


def test_proxy_preset_selectable_on_example() -> None:
    """The example still resolves the inherited core proxy preset."""
    env_cfg, _ = resolve_task_config(
        "IsaacContrib-Lift-Soft-Franka-Custom-Coupling", "", overrides=("physics=newton_mjwarp_vbd_proxy",)
    )

    assert isinstance(env_cfg.sim.physics.solver_cfg, CouplerProxyCfg)


def test_cloth_camera_task_uses_proxy_coupling() -> None:
    """The cloth camera task uses the core proxy coupling by default."""
    env_cfg, _ = resolve_task_config("Isaac-Lift-Soft-Franka-Camera", "", overrides=())

    assert isinstance(env_cfg.sim.physics.solver_cfg, CouplerProxyCfg)
