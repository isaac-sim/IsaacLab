# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ANYmal-D flat-terrain teacher configuration for learned-dynamics data collection and deployment."""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg

ANYMAL_D_REMOVE_ROOT_TRANSLATION = "planar"
"""Root translation removed from ANYmal-D model inputs, so that environment origins do not reach the model."""

ANYMAL_D_CONTINUITY = "off"
"""Continuity mode of a deployed ANYmal-D artifact.

The PhysicsNeMo Robotics joint codec normalizes the free-root quaternion again when writing a prediction,
so the written state can differ from the solver's own record by one floating-point unit and ``"verify"``
would reject it. Isaac Lab reports every environment reset through the solver reset mask, so turning the
check off does not hide resets.
"""


@configclass
class AnymalDFlatTeacherEnvCfg(AnymalDFlatEnvCfg):
    """ANYmal-D velocity tracking on flat ground at a 50 Hz physics frame.

    MuJoCo Warp advances each 20 ms frame in four 5 ms substeps with its internal contacts, and one
    environment step equals one frame, so the policy keeps its 50 Hz control rate. The configuration
    removes everything a learned solver cannot reproduce:

    * the contact-force sensor, with the rewards and termination that read it;
    * events that change model properties (material, mass, center of mass) or apply external wrenches.

    A fallen robot is terminated by its base orientation instead of a base contact.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.decimation = 1
        self.sim.dt = 0.02
        self.sim.render_interval = 1
        self.episode_length_s = 10.0
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=60, nconmax=25, cone="elliptic", impratio=100.0, integrator="implicitfast"
            ),
            num_substeps=4,
            use_cuda_graph=False,
            deterministic_mode="not_guaranteed",
            default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
        )
        # sensors and terms a learned solver cannot provide
        self.scene.contact_forces = None
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.terminations.base_contact = None
        self.terminations.bad_orientation = TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": 0.5 * math.pi}
        )
        # events that change the trained model or apply external wrenches
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
