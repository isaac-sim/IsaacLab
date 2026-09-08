# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""w100 with every gait fault this line has found priced, and nothing else added.

The stock task specifies "track the command and do not fall" and leaves posture and energy free, so
the solution set is enormous. Measured on Newton alone, at 6000 iterations, the same task produced
gaits spanning 127 to 237 W, waist angles from +4.5 to -26 degrees, and airborne-share ratios from
0.47 to 1.02 -- all at ``success_rate`` 0.96 to 1.00. Trained under PhysX it produced ~1000 W and
-20 to -29 degrees of recline. That spread is what an under-constrained objective looks like; the
score cannot tell those gaits apart, so nothing pushes them together.

This config adds the three terms this line established, and deliberately nothing else -- no domain
randomization, no stronger push, so the comparison is about the reward specification and not about
the training conditions:

* waist deviation L2 at -1.0, which took the recline from -26 degrees to +0.9,
* leg mechanical power at -5e-4, which took the airborne-share ratio from 0.47 to 1.02 and the
  power from 237 W to 132,
* hip-pitch deviation L2 at -0.5, because pricing power without pricing hip pitch bought the saving
  by crouching: -47.9 degrees of hip flexion and 15.7 cm of pelvis height.

The question it exists to answer is whether a well-specified task converges to the same gait in two
simulators whose actuators demonstrably differ. Read posture -- waist angle, stance width, the
airborne-share ratio -- and not power, which is not comparable across backends: the same Newton
policy replayed under PhysX draws 4439 W against 259, because PhysX removes about 55% of joint
velocity per physics step where Newton removes 2%.
"""

from isaaclab.utils.configclass import configclass

from .rough_29dof_dr_env_cfg import _WAIST_L2_WEIGHTS, _add_waist_l2
from .rough_29dof_posture_env_cfg import G129DofRoughHipL2AirTime100EnvCfg
from .rough_29dof_power_env_cfg import _HIP_PITCH_L2_WEIGHT, _POWER_WEIGHTS, _add_power_penalty


@configclass
class G129DofRoughAirTime100ConstrainedEnvCfg(G129DofRoughHipL2AirTime100EnvCfg):
    """w100 plus the waist, power and hip-pitch terms. No randomization, stock push."""

    def __post_init__(self):
        super().__post_init__()

        from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: PLC0415
        from isaaclab.managers import SceneEntityCfg  # noqa: PLC0415

        from .rough_29dof_wbc_env_cfg import joint_deviation_l2  # noqa: PLC0415

        _add_waist_l2(self, _WAIST_L2_WEIGHTS["t1"])
        _add_power_penalty(self, _POWER_WEIGHTS["p2"])
        self.rewards.joint_deviation_hip_pitch = RewTerm(
            func=joint_deviation_l2,
            weight=_HIP_PITCH_L2_WEIGHT,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint"])},
        )
