# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Settings of the neural robot dynamics model behind ``physics=newton_nerd`` of the ANYmal C flat task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_neural import NeRDRecipeCfg, NeRDRecordCfg, NeRDTrainCfg

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .nerd_recipe import AnymalCFlatNeRDRecipe


@configclass
class AnymalCFlatNeRDRecipeCfg(NeRDRecipeCfg):
    """FlashNeRD of the robot with the actuator network of its legs, with a ground probe on every foot.

    One frame is one control period. The robots are dropped from tilted, moving poses under random joint targets,
    and their falls stay in the data.
    """

    class_type: type[AnymalCFlatNeRDRecipe] | str = "{DIR}.nerd_recipe:AnymalCFlatNeRDRecipe"

    record: NeRDRecordCfg = NeRDRecordCfg(num_envs=4096, num_frames=2000, episode_frames=(100, 100))
    # The noise also perturbs the foot probes, which the loss computes again on the noisy states.
    train: NeRDTrainCfg = NeRDTrainCfg(
        max_steps=20_000, window=32, batch_size=2048, learning_rate=2.0e-3, noise=0.3, horizons=(1, 10, 50)
    )

    # The model stands for the nominal robot and does not see a randomized mass or centre of mass.
    unseen_events: tuple[str, ...] = ("add_base_mass", "base_com")

    foot_bodies: str = "*FOOT"
    """Bodies that carry a probe of the ground, as a glob pattern on the body names."""

    target_reach: float = 0.75
    """Largest offset of a joint target from the default pose [rad]. Under ``physics=newton_nerd`` the task clips its
    joint targets to the default of this range, so change it here and not on the command line."""

    hold_frames: tuple[int, int] = (1, 10)
    """Shortest and longest hold of a constant target [frames]."""

    drop: float = 0.2
    """Largest height above the nominal spawn height that an episode starts from [m]."""

    tilt: float = 0.5
    """Largest tilt of the base about a horizontal axis [rad]."""

    base_speed: tuple[float, float] = (1.0, 2.0)
    """Largest linear [m/s] and angular [rad/s] speed of the base."""

    joint_reach: float = 0.5
    """Offset of a joint from the default pose [rad]."""

    joint_speed: float = 2.0
    """Largest joint speed [rad/s]."""
