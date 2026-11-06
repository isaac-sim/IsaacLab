# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AgiBot G2 teleoperation tasks driven by AgiBot's ik_7d solver.

Run a VR session against the stock teleoperation script -- no Isaac Lab or
IsaacTeleop source is modified; ``--external_callback`` is the supported hook
for registering an out-of-tree environment:

.. code-block:: bash

    cd /path/to/IsaacLab
    export VIRTUAL_ENV=$PWD/.venv OMNI_KIT_ACCEPT_EULA=YES \
           LD_LIBRARY_PATH=/path/to/gcc13/lib:$LD_LIBRARY_PATH
    ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
        --external_callback isaaclab_contrib.tasks.agibot_g2.register \
        --task Isaac-Teleop-G2-Ik7d-v0 --num_envs 1

Controls: the grip pose of each controller drives that arm's end-effector; each
thumbstick's X axis swivels that arm's elbow about the shoulder-wrist axis
without moving the hand.

**Calibrating the first session.** Two things cannot be validated without a
headset, and both will look like the robot is behaving strangely rather than
like a configuration error:

* :attr:`XrCfg.anchor_pos` / :attr:`XrCfg.anchor_rot` place the operator's XR
  origin in the simulation world. This is absolute-pose teleoperation, so the
  operator's hands have to be physically where the robot's hands should be. If
  the arms slam to a limit the moment the session starts, the anchor is wrong,
  not the IK.
* ``Se3RetargeterConfig.target_offset_roll/pitch/yaw`` rotate the controller's
  grip frame onto the robot's end-effector frame. The values here were measured
  for this robot and grip; re-measure if either changes. Hold a controller in the
  posture the robot's hand should be in, read the commanded versus measured
  orientation, and set the offsets from that.

Note that the reachable envelope is small: end-effector displacement from home
is exact to 0.05 m in every direction and degrades beyond ~0.10 m forward and
outward. ik_7d does *not* report an unreachable target as a failure -- it
returns a valid, in-limit configuration that simply lags -- so an arm that
stops following the hand is at the edge of its envelope, not broken.
"""

import gymnasium as gym

TASK_ID = "Isaac-Teleop-G2-Ik7d-v0"
"""Gym id of the bimanual ik_7d teleoperation environment."""


def register() -> list[str]:
    """Register this package's environments with ``gymnasium``.

    Suitable as the ``--external_callback`` argument of the stock teleoperation
    and recording scripts, which call it before ``gym.make``. Idempotent.

    Returns:
        An empty list; callers treat ``None`` as "every remaining arg is unrecognized".
    """
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": f"{__name__}.teleop_ik7d_env_cfg:G2Ik7dTeleopEnvCfg"},
            disable_env_checker=True,
        )
    return []


register()
