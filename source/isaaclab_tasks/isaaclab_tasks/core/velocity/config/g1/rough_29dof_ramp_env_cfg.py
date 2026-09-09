# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ramping the base-height floor instead of switching it on and off.

Roughly a third of the runs on this task finish at ``success_rate`` 0.000, and the two arms of the
fork are indistinguishable at iteration 500. Measured on the waist-only sweep, Newton mesh:

============================  ======  ======  ======  ======
iteration                        500    1500    3000    5999
============================  ======  ======  ======  ======
s42 terrain level               0.00    2.41    5.91    5.84
s42 base-height terminations    0.645   0.190   0.035   0.020
s43 terrain level               0.00    0.08    0.42    1.35
s43 base-height terminations    0.589   0.574   0.134   0.120
============================  ======  ======  ======  ======

They separate between iteration 500 and 1500: one run gets its termination rate down and the
curriculum starts moving, the other stays cut short and never earns harder terrain, because the
curriculum promotes on distance walked. It is a bistability in the first fifteen hundred iterations,
not a difference in seed quality.

Withholding the termination entirely (:mod:`rough_29dof_warmup_env_cfg`) reached 4/5 rather than
5/5 and pushed takeoff to iteration 4500-5000, which does not fit a 6000-iteration budget. A window
that ends at 500 would close before the fork even opens.

So this ramps the floor rather than gating it: the minimum height starts at 0.20 m and reaches the
full 0.40 m at iteration 1500. A policy that is still learning to stand is not cut short while it
learns, so its episodes stay long enough for the curriculum to move; a policy that tries to walk in
a permanent crouch still hits a floor, and the floor it eventually meets is the one the task has
always used.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from .rough_29dof_env_cfg import MINIMUM_PELVIS_HEIGHT, pelvis_below_terrain_clearance
from .rough_29dof_waistonly_env_cfg import G129DofRoughAirTime100WaistOnlyEnvCfg

_RAMP_START_HEIGHT = 0.20
"""Floor in force at the first step [m].

Half the final value: low enough that a robot which has folded but not fallen keeps its episode,
high enough that lying down still ends it.
"""

_RAMP_STEPS = 36_000
"""Environment steps over which the floor rises to its full value.

24 steps per environment per iteration, so 1500 iterations -- the window the fork was measured to
open in. Short enough to leave four fifths of a 6000-iteration budget under the final floor.
"""


def pelvis_below_ramped_clearance(
    env,
    final_height: float,
    start_height: float,
    ramp_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
):
    """:func:`pelvis_below_terrain_clearance` with a floor that rises over the first steps.

    Args:
        env: Environment the termination is evaluated for.
        final_height: Floor once the ramp is complete [m].
        start_height: Floor at the first step [m].
        ramp_steps: Environment steps over which the floor rises.
        asset_cfg: The articulation whose root height is checked.
        sensor_cfg: The height scanner giving the terrain height under the robot.

    Returns:
        Per-environment termination flags, shape ``(num_envs,)``.
    """
    progress = min(1.0, env.common_step_counter / max(1, ramp_steps))
    height = start_height + (final_height - start_height) * progress
    return pelvis_below_terrain_clearance(env, height, asset_cfg, sensor_cfg)


@configclass
class G129DofRoughAirTime100WaistRampEnvCfg(G129DofRoughAirTime100WaistOnlyEnvCfg):
    """The waist-only arm with the base-height floor ramped over the first 1500 iterations."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.base_height = DoneTerm(
            func=pelvis_below_ramped_clearance,
            params={
                "final_height": MINIMUM_PELVIS_HEIGHT,
                "start_height": _RAMP_START_HEIGHT,
                "ramp_steps": _RAMP_STEPS,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
