# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A teacher-student pipeline built on the w100 gait rather than on the DR29 teacher.

w100 -- plate feet, hip deviation L2 at -1.0, air-time weight 1.0 -- is the best-looking gait this
line has produced, but its actor reads a terrain height scan and a base linear velocity, and a real
G1 publishes neither. Distillation is how that policy becomes deployable: the teacher keeps the
privileged observation it was trained on, and a student learns to reproduce its actions from
proprioception alone.

Two stages, two configs:

* :class:`G129DofRoughAirTime100RobustEnvCfg` is the teacher. It is the randomized w100 arm plus
  self-collision, which is the combination the DR29 line calls ``s3_robust`` -- measured there at
  ``success_rate`` 0.998 and ``fall_rate`` 0.049 against 0.962 / 0.130 for the un-randomized
  teacher. Self-collision is included because that is what was asked for, but it is the term that
  cost 17 cm of pelvis height on the DR29 line while buying no success, so read ``pelvis_height``
  on this arm before trusting it.
* :class:`G129DofRoughAirTime100DistillEnvCfg` is the student. Same physics, same randomization,
  two observation groups instead of one.

Two things must line up or the run is wrong rather than merely worse:

* **The teacher group must reproduce the observation the teacher was trained on**, term for term
  and in order. Here that is the stock policy group unchanged -- proprioception, the height scan
  and the base linear velocity, single-frame, exactly what w100's actor consumed.
* **The student group must be deployable**: no height scan, no base linear velocity, and five
  frames of proprioception in their place. A policy that cannot see the terrain has to infer it
  from how the last few steps went, which is what the history is for.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup  # noqa: F401
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.velocity_env_cfg import ObservationsCfg

from .rough_29dof_dr_env_cfg import (
    _HISTORY_LENGTH,
    _HISTORY_TERMS,
    _WAIST_L2_WEIGHTS,
    G129DofRoughAirTime100DREnvCfg,
    _add_waist_l2,
)
from .rough_29dof_mjlab_env_cfg import G129DofRoughMjlabScaleEnvCfg
from .rough_29dof_power_env_cfg import G129DofRoughWaist1Power2EnvCfg


@configclass
class G129DofRoughAirTime100RobustEnvCfg(G129DofRoughAirTime100DREnvCfg):
    """The teacher: the randomized w100 arm with self-collision enabled.

    The DR29 line's ``s3_robust`` combines both, and this is its counterpart on the shipped asset.
    One caveat that does not carry over: ``s3_robust`` ran on the trimmed collider rung, while this
    arm keeps the plate rung's full collider set, so self-collision here has considerably more pairs
    to test.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True


@configclass
class G129DofRoughAirTime100RobustWaistEnvCfg(G129DofRoughAirTime100RobustEnvCfg):
    """The teacher, with the waist held upright.

    :class:`G129DofRoughAirTime100RobustEnvCfg` was written before the recline was found, and it
    reproduces it: measured on its 6000-iteration checkpoint, ``waist_pitch`` is -24.3 degrees and
    the torso link -26.7, against -0.35 and -0.46 on w100. Nothing in that arm argues with it --
    ``joint_deviation_hip`` covers ``hip_roll`` and ``hip_yaw``, not the waist, and the waist's own
    term is the stock L1 at -0.1, which prices 24 degrees at 0.04 per step against a lean that is
    both steadier under a push and cheaper to hold.

    Distilling from a teacher that walks reclined can only produce a student that walks reclined, so
    the fix belongs here rather than in the student. The same repricing on the un-self-colliding arm
    (``t1``) took ``waist_pitch`` to +0.85 degrees with ``success_rate`` 1.000.
    """

    def __post_init__(self):
        super().__post_init__()
        _add_waist_l2(self, _WAIST_L2_WEIGHTS["t1"])


@configclass
class G129DofRoughAirTime100DistillObservationsCfg(ObservationsCfg):
    """Two groups: what the robot can publish, and what the teacher was trained on."""

    @configclass
    class TeacherCfg(ObservationsCfg.PolicyCfg):
        """The teacher's own observation, unchanged. Do not reorder -- see the module docstring."""

    teacher: TeacherCfg = TeacherCfg()


@configclass
class G129DofRoughAirTime100DistillEnvCfg(G129DofRoughAirTime100RobustWaistEnvCfg):
    """Distil the sighted w100 teacher into a proprioception-only student.

    Built on the waist-corrected teacher: the student's environment has to match the one its teacher
    was trained in, and the teacher observation group has to reproduce that teacher's input exactly.
    """

    observations: G129DofRoughAirTime100DistillObservationsCfg = G129DofRoughAirTime100DistillObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # The student's group: what a real G1 publishes, with five frames of proprioception in
        # place of the terrain it cannot see. Per term rather than group-level -- there is no
        # height scan left to stack, but the layout has to match the deployment stack's, which
        # buffers per term and concatenates afterwards.
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        for term in _HISTORY_TERMS:
            obs_term = getattr(self.observations.policy, term)
            obs_term.history_length = _HISTORY_LENGTH
            obs_term.flatten_history_dim = True

        # The teacher's group is the stock policy group, left exactly as the teacher trained on it,
        # except for the corruption: the teacher was trained on the noisy stream, but here it is an
        # oracle being copied, and its noise would only inject variance into the student's targets.
        self.observations.teacher.enable_corruption = False


@configclass
class G129DofRoughWaist1Power2BlindDistillEnvCfg(G129DofRoughWaist1Power2EnvCfg):
    """Distil ``cw`` into a proprioception-only student.

    ``cw`` -- randomization, the stronger push, waist L2 at -1.0 and the leg power penalty at
    -5e-4 -- is the arm that walks evenly (airborne-share ratio 1.02 against t1's 0.47) and it is
    the gait asked for on the robot. Like every actor in this line it reads 328 values, 190 of which
    the robot cannot produce: ``height_scan`` (187) needs a terrain map and ``base_lin_vel`` (3) a
    base-velocity estimator. Distillation is the only way that gait reaches ``g1_deploy``.

    The environment is ``cw``'s, unchanged, because a student distilled in a different environment
    from its teacher learns the wrong thing while reporting a low behaviour loss -- measured on this
    line at 0.024 loss and 0.476 success, against 0.930 once the environments matched. What changes
    is only the observation: the teacher group is ``cw``'s own input, and the student group is what
    ``g1_deploy``'s ``core.observe`` can build, with five frames of proprioception standing in for
    the terrain it cannot see.

    Note that ``cw`` is not the ``s3_robust`` teacher: self-collision is off here, since the arm was
    never trained with it.
    """

    observations: G129DofRoughAirTime100DistillObservationsCfg = G129DofRoughAirTime100DistillObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        for term in _HISTORY_TERMS:
            obs_term = getattr(self.observations.policy, term)
            obs_term.history_length = _HISTORY_LENGTH
            obs_term.flatten_history_dim = True
        self.observations.teacher.enable_corruption = False


@configclass
class G129DofRoughMjlabScaleBlindDistillEnvCfg(G129DofRoughMjlabScaleEnvCfg):
    """Distil ``yms`` into a proprioception-only student.

    ``yms`` -- the per-joint action scale plus left-right mirror augmentation -- is the best gait
    measured on this line: 0.990 success at sd 0.003, and on flat ground under a pinned straight
    command a pelvis roll of 0.55 degrees against the control's 2.8, an airborne share ratio of 1.03
    to 1.07 against 0.71, and a worst joint-pair asymmetry under one degree against three to six.

    Its actor still reads 328 values, 190 of which the robot cannot produce -- ``height_scan`` (187)
    and ``base_lin_vel`` (3) -- so it reaches ``g1_deploy`` only through a student. The environment
    is ``yms``'s own, unchanged, because distilling in a different environment scored 0.476 against
    0.930 on this line for the same behaviour loss.

    The mirror augmentation is a property of how the teacher was trained, not of the environment, so
    it does not appear here; whether the student inherits the symmetry is the thing to measure.
    """

    observations: G129DofRoughAirTime100DistillObservationsCfg = G129DofRoughAirTime100DistillObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        for term in _HISTORY_TERMS:
            obs_term = getattr(self.observations.policy, term)
            obs_term.history_length = _HISTORY_LENGTH
            obs_term.flatten_history_dim = True
        self.observations.teacher.enable_corruption = False
