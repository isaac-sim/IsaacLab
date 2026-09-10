# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""An explicit PD actuator, because the robot's motor controller is one.

Both this line and mjlab drive the joints *implicitly*: the gains are handed to the solver, which
folds the damping into the integrator. MuJoCo's ``implicitfast`` and Isaac Lab's
:class:`~isaaclab.actuators.ImplicitActuator` are both this. It is unconditionally stable in the
damping term, which is convenient in simulation and is not what the hardware does -- a G1's motor
controller computes ``tau = kp (q* - q) - kd qdot`` explicitly at its own rate and applies it as a
torque, with no knowledge of the next state.

So a policy trained implicitly can lean on damping the real controller cannot deliver, and the
symptom on hardware is high-frequency oscillation at gains that looked fine in simulation. Reported
independently by the author of the sentdex G1 video, who could not transfer at all until the
implicit actuator was replaced by an explicit one, and then could.

:class:`~isaaclab.actuators.IdealPDActuator` computes the same law in Python each control step and
writes the result as an effort, which is the hardware's arrangement. Nothing else changes: the same
stiffness, damping, armature and torque ceilings, on top of ``yms`` -- the per-joint action scale
plus mirror augmentation, which is the best configuration measured here (0.990 success at sd 0.003,
pelvis roll 0.55 degrees, left/right airborne ratio 1.03 to 1.07).

What this arm answers is whether those gains are physically realisable or an artefact of the
integrator. If it trains, the configuration is a sim2real candidate; if it does not, the gains were
being propped up by the solver and the whole line needs retuning before any of it reaches hardware.
"""

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.utils.configclass import configclass

from .rough_29dof_mjlab_env_cfg import G129DofRoughMjlabScaleEnvCfg


def _to_explicit(cfg) -> None:
    """Rebuild every actuator group as an explicit PD model, in place, keeping its numbers."""
    rebuilt = {}
    for name, group in cfg.scene.robot.actuators.items():
        rebuilt[name] = IdealPDActuatorCfg(
            joint_names_expr=list(group.joint_names_expr),
            stiffness=group.stiffness,
            damping=group.damping,
            armature=group.armature,
            effort_limit_sim=group.effort_limit_sim,
            velocity_limit_sim=getattr(group, "velocity_limit_sim", None),
        )
    cfg.scene.robot.actuators = rebuilt


@configclass
class G129DofRoughExplicitEnvCfg(G129DofRoughMjlabScaleEnvCfg):
    """``ms`` with the PD computed explicitly rather than by the solver."""

    def __post_init__(self):
        super().__post_init__()
        _to_explicit(self)
