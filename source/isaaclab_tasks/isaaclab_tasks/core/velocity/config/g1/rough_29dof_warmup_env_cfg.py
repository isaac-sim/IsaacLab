# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Withholding the base-height termination until the policy can stand.

Roughly a third of the runs in this line finish at ``success_rate`` 0.000, and the failure has a
mechanism rather than being seed noise. Traced on ``mesh_newton/s42`` and ``hf_newton/s44``: nothing
goes non-finite, episode length ends up near the 1000-step cap, and the robot is alive but not
walking. What separates them from the runs that work is the terrain curriculum --

====================================  ==============  ==============
term (mean of the last 20 iterations)  failed (s42)    worked (s43)
====================================  ==============  ==============
``Curriculum/terrain_levels``          0.85            5.80
``Episode_Termination/base_height``    0.345           0.013
``Episode_Reward/feet_air_time``       0.010           0.096
``track_lin_vel_xy``                   0.417           0.827
====================================  ==============  ==============

-- the failed run is still on the easiest terrain after 6000 iterations, ends a third of its
episodes on the height termination, and barely lifts its feet. The curriculum promotes on distance
walked, so a policy that keeps being cut short never earns harder terrain and never leaves the
crouch it is being killed for.

The termination exists because this asset has no other signal for a robot that has stopped standing
-- without it ``success_rate`` ends at 0.010 on flat ground. But it is doing that job from step one,
against a policy that cannot stand yet, and that is what closes the door. Withholding it for a warm-
up period leaves the crouch unpunished exactly while the policy is too poor to avoid it, and turns
it on once walking is possible at all.

This is a hypothesis about *why* the bimodality exists, so it is worth five seeds rather than one:
the effect to measure is the fraction of runs that reach a working gait, and at three seeds a change
from 2/3 to 3/3 means nothing.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from .rough_29dof_env_cfg import MINIMUM_PELVIS_HEIGHT, pelvis_below_terrain_clearance
from .rough_29dof_posture_env_cfg import G129DofRoughHipL2AirTime100EnvCfg

_WARMUP_STEPS = 12_000
"""Environment steps before the base-height termination starts firing.

24 steps per environment per iteration, so this is 500 PPO iterations. Long enough to cover the
phase where the policy cannot stand -- the runs that work are still at ``success_rate`` 0.000 at
iteration 500 -- and far short of the 2000-3000 where takeoff happens, so the termination is back on
well before the gait is decided.
"""


def pelvis_below_terrain_clearance_after_warmup(
    env,
    minimum_height: float,
    warmup_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
):
    """:func:`pelvis_below_terrain_clearance`, inert for the first ``warmup_steps`` steps.

    Args:
        env: Environment the termination is evaluated for.
        minimum_height: Height above the terrain below which the episode ends [m].
        warmup_steps: Environment steps to wait before the termination starts firing.
        asset_cfg: The articulation whose root height is checked.
        sensor_cfg: The height scanner giving the terrain height under the robot.

    Returns:
        Per-environment termination flags, shape ``(num_envs,)``.
    """
    import torch  # noqa: PLC0415

    if env.common_step_counter < warmup_steps:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return pelvis_below_terrain_clearance(env, minimum_height, asset_cfg, sensor_cfg)


@configclass
class G129DofRoughAirTime100HeightWarmupEnvCfg(G129DofRoughHipL2AirTime100EnvCfg):
    """w100 with the base-height termination withheld for the first 500 iterations."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.base_height = DoneTerm(
            func=pelvis_below_terrain_clearance_after_warmup,
            params={
                "minimum_height": MINIMUM_PELVIS_HEIGHT,
                "warmup_steps": _WARMUP_STEPS,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
