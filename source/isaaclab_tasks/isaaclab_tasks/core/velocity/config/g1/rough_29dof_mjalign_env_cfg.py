# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``su`` with every alignable difference against the MuJoCo model closed.

The deployment evidence is that MuJoCo and the robot agree with each other and Isaac Lab does not,
so the alignment moves Isaac Lab toward ``unitree_mujoco``'s ``scene_29dof.xml`` rather than the
other way. Audited term by term on 2026-09-11 with ``scratchpad/align_audit.py``, both sides read
off the artefact that runs -- the composed USD stage and the MuJoCo-compiled model, the latter with
the deploy loop's own foot-plate override applied.

**Already identical, nothing to do:** all 29 shared joint limits (under 0.001 rad); armature 0.01;
25 of 30 shared bodies' mass and principal moments *exactly*; the PD gains, which the deploy loop
reads from the exported contract; the plate foot (0.2031 x 0.0655 x 0.0185 m on both sides, origins
apart by 0.9 mm).

**Closed here:**

===================== ==================== ==================== =========================
term                  was (Isaac Lab)      now (MuJoCo's)       how
===================== ==================== ==================== =========================
torso/waist inertials CAD export           hardware-identified  ``g1_mj_aligned.usda``
torque ceiling        300 / 20 / 50 / 300  88 / 139 / 50 / 25   ``_apply_hardware_efforts``
joint dry friction    0                    0.2 (0.1 on wrists)  ``friction``
passive damping       0                    0.05                 ``viscous_friction``
ground friction       0.8 static / 0.6 dyn 1.0                  ``physics_material``
===================== ==================== ==================== =========================

The ankle ceiling moves the wrong way if you only read the headline: Isaac Lab's 20 N*m is *below*
the hardware's 50, so this arm gives the ankle more authority, not less, while taking it away from
the knee (300 -> 139) and the arms (300 -> 25).

**Left open, and why:**

* **The hands.** The USD carries an articulated Dex3 -- 16 bodies, 14 joints, 1.020 kg -- where the
  MJCF fuses a rigid rubber hand into ``wrist_yaw_link`` (0.255 kg against the USD's 0.085). They
  are different hands, not hand against no hand, and which is right depends on what is fitted to the
  robot. Porting the MJCF's wrist mass without deleting the Dex3 bodies would double-count, so the
  generator keeps it behind ``--wrists``.
* **The collision set.** The USD collides on head, torso, knees, wrist_yaw, pelvis_contour and the
  feet; the MJCF collides on the whole leg chain (hip pitch/roll/yaw, knee, ankle pitch, foot) and
  the whole arm chain, and has no head or hands. Self-collision is off on both, so on flat ground
  this only shows up in a fall -- but it is the largest remaining difference and wants its own arm.
* **The actuator model.** Isaac Lab evaluates its PD once per 50 Hz control step and hands a target
  to the solver; the MuJoCo loop recomputes ``tau = kp(q* - q) + kd(dq* - dq)`` every 2 ms from
  fresh state. ``lockstep_compare.py`` puts the first divergence at step 7 (0.14 s) on the knee and
  shows that zeroing MuJoCo's damping and friction and matching the trunk mass barely moves it, so
  this is the leading suspect and cannot be closed by a config change.
"""

from isaaclab.utils.configclass import configclass

from .rough_29dof_dr_env_cfg import _add_randomization
from .rough_29dof_env_cfg import _apply_hardware_efforts
from .rough_29dof_standup_env_cfg import G129DofRoughStandUpEnvCfg

MJCF_FRICTIONLOSS = 0.2
"""Joint dry friction [N*m] the MJCF gives 25 of its 29 joints.

In Isaac Sim 5.0 and later ``ActuatorBaseCfg.friction`` is an effort rather than a coefficient, so
this transfers one for one. It matters most at the ankle, where the PD stiffness is 20: 0.2 N*m is a
large fraction of the torque the controller has to play with, and Isaac Lab had none of it.
"""

MJCF_WRIST_FRICTIONLOSS = 0.1
"""What the MJCF gives the four wrist pitch/yaw joints instead."""

MJCF_DAMPING = 0.05
"""Passive viscous damping [N*m*s/rad], every joint. Distinct from the PD's ``damping``, which is
the controller's kd and already matches the deployment contract on both sides."""

MJCF_GROUND_FRICTION = 1.0
"""The MJCF floor's sliding friction, against Isaac Lab's 0.8 static / 0.6 dynamic."""


def _align_to_mujoco(cfg) -> None:
    """Close every term that a config can close, in place."""
    _apply_hardware_efforts(cfg)

    actuators = cfg.scene.robot.actuators
    for name, actuator in actuators.items():
        if name == "hands":
            # The MJCF has no hand joints at all, so there is no value to port.
            continue
        actuator.viscous_friction = MJCF_DAMPING
        if name == "arms":
            # Patterns here must partition the group: the resolver rejects a joint matched twice,
            # so the catch-all is spelled out rather than written as ".*".
            actuator.friction = {
                ".*_wrist_pitch_joint": MJCF_WRIST_FRICTIONLOSS,
                ".*_wrist_yaw_joint": MJCF_WRIST_FRICTIONLOSS,
                ".*_shoulder_.*_joint": MJCF_FRICTIONLOSS,
                ".*_elbow_joint": MJCF_FRICTIONLOSS,
                ".*_wrist_roll_joint": MJCF_FRICTIONLOSS,
            }
        else:
            actuator.friction = MJCF_FRICTIONLOSS

    material = cfg.events.physics_material.params
    material["static_friction_range"] = (MJCF_GROUND_FRICTION, MJCF_GROUND_FRICTION)
    material["dynamic_friction_range"] = (MJCF_GROUND_FRICTION, MJCF_GROUND_FRICTION)


@configclass
class G129DofRoughMujocoAlignedEnvCfg(G129DofRoughStandUpEnvCfg):
    """``su`` with the MuJoCo-alignable terms closed. Pair it with ``g1_mj_aligned.usda``."""

    def __post_init__(self):
        super().__post_init__()
        _align_to_mujoco(self)


MJDR_FRICTION_RANGE = (0.5, 2.0)
"""Ground friction span for the randomized variant, as a multiple of MuJoCo's 1.0.

DR29 uses 0.4-1.6 static and 0.3-1.2 dynamic, which its own docstring describes as Isaac Lab's stock
0.8/0.6 scaled by 0.5-2x. The same intent against MuJoCo's floor -- a single sliding friction of 1.0
-- is 0.5 to 2.0 on both, so the span is recentred rather than inherited. Inheriting DR29's numbers
would centre the draw on 0.8/0.6 and quietly undo the one term this line exists to pin.
"""


@configclass
class G129DofRoughMujocoAlignedDREnvCfg(G129DofRoughMujocoAlignedEnvCfg):
    """The aligned arm with DR29's randomization spread around MuJoCo's values.

    The alignment and the randomization are not in conflict as long as the draws are centred on what
    was aligned to. DR29's joint-friction span is already ``0.0-0.3`` "to bracket the official
    MuJoCo G1's 0.2 N*m of frictionloss" -- its own words -- so that one carries over untouched and
    now sits around a nominal this arm actually uses. Armature and actuator stiffness are scaled
    0.5-2x about their nominal, which is unchanged by the alignment. Only the ground friction has to
    be respanned, because DR29's numbers are relative to the stock floor rather than MuJoCo's.

    The trade this makes is explicit: a randomized floor is no longer MuJoCo's floor, so this arm is
    the sim2real candidate and the unrandomized :class:`G129DofRoughMujocoAlignedEnvCfg` stays the
    one to lockstep against MuJoCo.
    """

    def __post_init__(self):
        super().__post_init__()
        _add_randomization(self)
        self.events.physics_material.params.update(
            {
                "static_friction_range": MJDR_FRICTION_RANGE,
                "dynamic_friction_range": MJDR_FRICTION_RANGE,
            }
        )
