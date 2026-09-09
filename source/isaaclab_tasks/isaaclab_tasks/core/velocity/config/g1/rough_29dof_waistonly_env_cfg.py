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
