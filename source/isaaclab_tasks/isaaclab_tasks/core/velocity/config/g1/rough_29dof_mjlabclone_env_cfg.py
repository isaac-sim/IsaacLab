# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Everything mjlab does, not just its rewards: the plant as well as the objective.

Porting mjlab's reward and termination set alone (``mf``) fails outright -- 76-85% of episodes end
on ``fell_over``, terrain level pinned at 0, ``success_rate`` flat at 0.40 through 3700 iterations.
That answered the wrong question. mjlab does not have a -200 termination penalty either, so if the
rewards were the whole story their robot would fall over too. It does not, because **it is not the
same plant**:

===============================  ===================  ==================  ==========
quantity                         mjlab                ours                
===============================  ===================  ==================  ==========
hip_pitch / hip_yaw / waist_yaw  kp 40.2, kd 2.56     kp 200, kd 5        5.0x stiff
hip_roll / knee                  kp 99.1, kd 6.31     kp 150-200, kd 5    1.5-2.0x
waist_roll / waist_pitch         kp 28.5, kd 1.81     kp 200, kd 5        7.0x
ankle                            kp 28.5, kd 1.81     kp 20, kd 2         0.7x
shoulder / elbow                 kp 14.3, kd 0.91     kp 40, kd 10        2.8x
kd/kp                            0.0637 everywhere    0.025 to 0.25
armature                         0.0036 to 0.0251     0.01 flat
===============================  ===================  ==================  ==========

mjlab's gains are not chosen, they are derived: ``kp = armature * (2*pi*10 Hz)^2`` and
``kd = 2*zeta*armature*omega`` with ``zeta = 2``, from each motor's reflected inertia through its
two-stage planetary. That fixes ``kd/kp = 2*zeta/omega = 0.0637 s`` for every joint by construction,
which is overdamped -- the actuator cannot oscillate. Ours is five to seven times stiffer on the
hips and waist and two and a half times less damped there, which is a plant that can.

The action scale follows from the gains, which is why ``ms`` was mislabelled. The rule
``0.25 * effort / stiffness`` gives mjlab 0.35-0.55 on the legs -- essentially the blanket 0.5 we
started with -- and gives 0.11-0.17 only because our stiffness is wrong.

So this config changes the robot rather than the objective:

* **Actuators** rebuilt from mjlab's motor table: armature, stiffness, damping, effort and velocity
  limits per motor type, with the ankles and waist roll/pitch as two 5020s in parallel.
* **Action scale** ``0.25 * effort / stiffness`` on those gains, so mjlab's own numbers.
* **Initial pose** ``KNEES_BENT``: hip -0.312, knee 0.669, ankle -0.363, root at 0.76 m, against our
  -0.20 / 0.42 / -0.23 at 0.793. Note this moves both the action offset and the observation zero.
* **Collision** replaced wholesale by ``scripts/make_g1_mjlab_colliders.py`` -- 31 capsules and 2
  spheres, seven capsules per sole against the shipped four spheres, and no hand colliders at all.
  Self-collision on, as mjlab's ``FULL_COLLISION`` has it.
* **Fingers out** of the action and observation spaces: mjlab's robot has 29 joints.
* **Commands** mjlab's: ``vx`` and ``vy`` both (-1, 1) -- ours never asked for lateral or backward
  motion -- ``wz`` (-0.5, 0.5), resampled every 3 to 8 seconds, 30% heading environments.
* **Push** every 1 to 3 seconds, against our 10 to 15.
* ``soft_joint_pos_limit_factor`` 0.9.

Rewards and terminations are inherited from :class:`G129DofRoughMjlabFullEnvCfg`.

**What is still ours:** the terrain generator, the observation *set* (single group for actor and
critic, including ``base_lin_vel`` and the height scan, where mjlab runs an asymmetric
actor-critic), the PPO hyperparameters, and the physics backend. If this arm walks, those are the
remaining differences; if it does not, the plant was not the answer either.
"""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .rough_29dof_mjlab_env_cfg import _LOCOMOTION_JOINTS
from .rough_29dof_mjlabfull_env_cfg import G129DofRoughMjlabFullEnvCfg

_MOTORS = {
    # name: (joint patterns, armature, stiffness, damping, effort, velocity)
    "m5020": (
        [".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint",
         ".*_wrist_roll_joint"],
        0.00360972, 14.250623, 0.907223, 25.0, 37.0,
    ),
    "m7520_14": ([".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"],
                 0.01017752, 40.179239, 2.557890, 88.0, 32.0),
    "m7520_22": ([".*_hip_roll_joint", ".*_knee_joint"], 0.02510192, 99.098428, 6.308802, 139.0, 20.0),
    "m4010": ([".*_wrist_pitch_joint", ".*_wrist_yaw_joint"], 0.00425000, 16.778327, 1.068142, 5.0, 22.0),
    # Waist roll/pitch and the ankles are four-bar linkages driven by two 5020s. mjlab assumes a
    # nominal 1:1 ratio and sums the two motors, which is what these doubled numbers are.
    "m_double5020": ([".*_ankle_pitch_joint", ".*_ankle_roll_joint", "waist_roll_joint", "waist_pitch_joint"],
                     0.00721945, 28.501246, 1.814446, 50.0, 37.0),
}
"""mjlab's motor table, evaluated. ``armature`` is the reflected inertia of the two-stage planetary;
``stiffness = armature * (2*pi*10)^2`` and ``damping = 2 * 2 * armature * (2*pi*10)``."""

_MJLAB_TRUE_SCALE = {
    ".*_shoulder_pitch_joint": 0.438577,
    ".*_shoulder_roll_joint": 0.438577,
    ".*_shoulder_yaw_joint": 0.438577,
    ".*_elbow_joint": 0.438577,
    ".*_wrist_roll_joint": 0.438577,
    ".*_hip_pitch_joint": 0.547546,
    ".*_hip_yaw_joint": 0.547546,
    "waist_yaw_joint": 0.547546,
    ".*_hip_roll_joint": 0.350661,
    ".*_knee_joint": 0.350661,
    ".*_wrist_pitch_joint": 0.074501,
    ".*_wrist_yaw_joint": 0.074501,
    ".*_ankle_pitch_joint": 0.438577,
    ".*_ankle_roll_joint": 0.438577,
    "waist_roll_joint": 0.438577,
    "waist_pitch_joint": 0.438577,
}
"""``0.25 * effort / stiffness`` on mjlab's gains -- one unit of action is a quarter of the motor's
stall torque. On the legs this is 0.35 to 0.55, near the blanket 0.5 we started with; it only came
out at 0.11 to 0.17 in ``ms`` because that used our five-times-stiffer gains."""

_KNEES_BENT = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}
_KNEES_BENT_HEIGHT = 0.76


@configclass
class G129DofRoughMjlabCloneEnvCfg(G129DofRoughMjlabFullEnvCfg):
    """mjlab's plant and mjlab's objective, on the shipped Isaac asset."""

    def __post_init__(self):
        super().__post_init__()

        robot = self.scene.robot
        robot.actuators = {
            name: ImplicitActuatorCfg(
                joint_names_expr=list(patterns),
                stiffness=stiffness,
                damping=damping,
                armature=armature,
                effort_limit_sim=effort,
                velocity_limit_sim=velocity,
            )
            for name, (patterns, armature, stiffness, damping, effort, velocity) in _MOTORS.items()
        }
        # The fingers are out of the action space, but the joints still exist and need something
        # holding them at zero. mjlab's robot has none, so any weak position actuator will do.
        robot.actuators["hands"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_hand_.*_joint"],
            stiffness=1.0,
            damping=0.1,
            armature=0.001,
            effort_limit_sim=2.45,
        )
        robot.soft_joint_pos_limit_factor = 0.9
        robot.spawn.articulation_props.enabled_self_collisions = True

        robot.init_state.pos = (0.0, 0.0, _KNEES_BENT_HEIGHT)
        robot.init_state.joint_pos = dict(_KNEES_BENT)

        self.actions.joint_pos.joint_names = list(_LOCOMOTION_JOINTS)
        self.actions.joint_pos.scale = dict(_MJLAB_TRUE_SCALE)
        for term in ("joint_pos", "joint_vel"):
            getattr(self.observations.policy, term).params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=list(_LOCOMOTION_JOINTS)
            )

        command = self.commands.base_velocity
        command.ranges.lin_vel_x = (-1.0, 1.0)
        command.ranges.lin_vel_y = (-1.0, 1.0)
        command.ranges.ang_vel_z = (-0.5, 0.5)
        command.resampling_time_range = (3.0, 8.0)
        command.heading_command = True
        command.heading_control_stiffness = 0.5
        command.rel_heading_envs = 0.3

        if getattr(self.events, "push_robot", None) is not None:
            self.events.push_robot.interval_range_s = (1.0, 3.0)


# Referenced by the workflow so the layer path lives in one place.
MJLAB_COLLIDER_LAYER = "g1_mjlab_colliders.usda"
"""Filename of the layer :mod:`scripts.make_g1_mjlab_colliders` writes."""

