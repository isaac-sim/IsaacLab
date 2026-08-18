# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Digit velocity physics presets."""

from isaaclab_tasks.contrib.velocity.config.anymal_c.rough_env_cfg import AnymalCRoughEnvCfg
from isaaclab_tasks.contrib.velocity.config.digit.rough_env_cfg import DigitRoughEnvCfg
from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets


def _preset_names(env_cfg) -> set[str]:
    """Return every preset alternative name declared anywhere in *env_cfg*."""
    return {name for variants in collect_presets(env_cfg).values() for name in variants}


def test_digit_declares_no_newton_preset_because_it_cannot_run_newton() -> None:
    """Digit is PhysX-only, so no ``newton_`` name may remain selectable.

    ``LocomotionVelocityRoughEnvCfg`` declares ``base_com`` with a ``newton_mjwarp``
    branch that disables the randomization. On tasks offering Newton that branch rides
    along with the backend, but Digit has no Newton physics for it to attach to, so it
    would surface as a bare ``presets=newton_mjwarp`` that strips the randomization
    from a PhysX run.
    """
    assert not {name for name in _preset_names(DigitRoughEnvCfg()) if name.startswith("newton")}


def test_digit_still_randomizes_the_torso_centre_of_mass() -> None:
    """Collapsing the preset must keep the randomization, retargeted to Digit's torso."""
    env_cfg = resolve_presets(DigitRoughEnvCfg(), selected=("physx",))

    assert env_cfg.events.base_com is not None
    assert env_cfg.events.base_com.params["asset_cfg"].body_names == "torso_base"


def test_velocity_tasks_that_do_run_newton_keep_the_companion_branch() -> None:
    """The shared preset is untouched: only Digit opts out of it."""
    assert "newton_mjwarp" in collect_presets(AnymalCRoughEnvCfg())["events.base_com"]
