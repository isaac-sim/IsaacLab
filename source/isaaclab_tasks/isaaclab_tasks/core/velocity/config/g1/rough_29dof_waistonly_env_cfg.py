# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""w100 with one term added: the waist held upright.

The question this answers is whether a single configuration trains on both physics backends and
keeps w100's posture on both. What is known so far:

* **w100 itself trains on both** -- 2/3 seeds on Newton (heightfield and mesh alike) and 3/3 on
  PhysX -- but its PhysX gaits walk reclined 20 to 29 degrees at the waist, against -0.4 on Newton.
  Posture, not trainability, is what fails there.
* **The three-term stack does not train.** w100 plus waist L2, the power penalty and hip pitch, with
  no randomization, reached 0.087 / 0.284 / 0.203 on Newton and 0.946 / 0.000 / 0.000 on PhysX at
  6000 iterations. Constraining the gait that hard makes it unlearnable.

So the minimal candidate is the one term that targets the fault actually observed: waist deviation
repriced from the stock L1 at -0.1 to L2 at -1.0, which on the randomized Newton arm took the recline
from -26 degrees to +0.9 without costing success. Everything else is w100 untouched -- no
randomization, stock push, no power penalty, no hip-pitch term.

L2 rather than a larger L1 for the reason the hip fix needed it: a constant gradient prices two
degrees of lean the same as twenty, so the policy runs to the limit.
"""

from isaaclab.utils.configclass import configclass

from .rough_29dof_dr_env_cfg import _WAIST_L2_WEIGHTS, _add_waist_l2
from .rough_29dof_posture_env_cfg import G129DofRoughHipL2AirTime100EnvCfg


@configclass
class G129DofRoughAirTime100WaistOnlyEnvCfg(G129DofRoughHipL2AirTime100EnvCfg):
    """w100 plus waist deviation L2 at -1.0, and nothing else."""

    def __post_init__(self):
        super().__post_init__()
        _add_waist_l2(self, _WAIST_L2_WEIGHTS["t1"])


@configclass
class G129DofRoughAirTime100WaistWarmupEnvCfg(G129DofRoughAirTime100WaistOnlyEnvCfg):
    """The waist-only arm with the base-height termination withheld for the first 500 iterations.

    Waist L2 alone covers three of the four cells: PhysX mesh 3/3 at 0.99, Newton mesh 2/3 at 0.996
    and 0.962, and posture within a degree of w100's on both. The cell it does not cover is Newton
    heightfield, where all three seeds ended at 0.000 -- against 2/3 for w100 without the term.

    Those failures are the shape this line has produced all along: nothing goes non-finite, episode
    length sits near the cap, and the robot is alive without walking, with the terrain curriculum
    stuck near level 1 and a base-height termination rate an order of magnitude above the runs that
    work. The curriculum promotes on distance walked, so a policy that keeps being cut short never
    earns harder terrain. Withholding the termination while the policy cannot stand yet took the
    seed that failed outright on plain w100 to the best score of its sweep (0.000 -> 0.964).

    Combining them is therefore aimed at the failure actually observed rather than at the score.
    """

    def __post_init__(self):
        super().__post_init__()

        from isaaclab.managers import SceneEntityCfg  # noqa: PLC0415
        from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: PLC0415

        from .rough_29dof_env_cfg import MINIMUM_PELVIS_HEIGHT  # noqa: PLC0415
        from .rough_29dof_warmup_env_cfg import (  # noqa: PLC0415
            _WARMUP_STEPS,
            pelvis_below_terrain_clearance_after_warmup,
        )

        self.terminations.base_height = DoneTerm(
            func=pelvis_below_terrain_clearance_after_warmup,
            params={
                "minimum_height": MINIMUM_PELVIS_HEIGHT,
                "warmup_steps": _WARMUP_STEPS,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
