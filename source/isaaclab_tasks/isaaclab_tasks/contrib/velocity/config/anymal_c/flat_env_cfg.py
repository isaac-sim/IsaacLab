# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_neural.physics import NeRDSolverCfg

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.velocity.velocity_env_cfg import VELOCITY_SIM_CFG
from isaaclab_tasks.utils import preset

from .rough_env_cfg import AnymalCRoughEnvCfg

ANYMAL_C_NERD_TARGET_REACH = 0.75


@configclass
class AnymalCFlatEnvCfg(AnymalCRoughEnvCfg):
    decimation: int = preset(default=4, newton_nerd=1)
    sim: SimulationCfg = replace(
        VELOCITY_SIM_CFG,
        dt=preset(default=0.005, newton_nerd=0.02),
        render_interval=preset(default=4, newton_nerd=1),
        physics=replace(
            VELOCITY_SIM_CFG.physics,
            newton_nerd=replace(
                VELOCITY_SIM_CFG.physics.newton_nerd,
                solver_cfg=NeRDSolverCfg(
                    checkpoint="logs/neural/anymal_c_flat/flashnerd.pt",
                    articulation="{ENV_REGEX_NS}/Robot",
                    precision="fp16",
                ),
                # Keep contact sensors active when predicted feet hover just above the ground.
                default_shape_cfg=replace(VELOCITY_SIM_CFG.physics.newton_nerd.default_shape_cfg, margin=0.002),
            ),
        ),
    )

    def __post_init__(self):
        super().__post_init__()

        # physics
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 120
        newton_mjwarp.solver_cfg.nconmax = 15
        newton_mjwarp.solver_cfg.cone = "elliptic"
        newton_mjwarp.solver_cfg.impratio = 100.0
        self.sim.physics.default = self.sim.physics.ovphysx
        self.sim.physics.newton_steel.solver_cfg.iterations = 16
        # scene
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # observations
        self.observations.policy.height_scan = None
        # rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5
        # curriculum
        self.curriculum.terrain_levels = None

        # The joint targets pass straight on to the model. The gains [N*m/rad, N*m*s/rad] only estimate the joint
        # efforts that the torque penalty reads.
        legs = self.scene.robot.actuators["legs"]
        pass_through = ImplicitActuatorCfg(
            joint_names_expr=legs.joint_names_expr,
            stiffness=75.0,
            damping=4.0,
            actuator_effort_limit=legs.actuator_effort_limit,
            actuator_velocity_limit=legs.actuator_velocity_limit,
        )
        self.scene.robot.actuators["legs"] = preset(default=legs, newton_nerd=pass_through)
        # The joint targets [rad] stay within the range that the model was recorded on, whatever sets the actions.
        reach = ANYMAL_C_NERD_TARGET_REACH
        recorded = {
            joints: (angle - reach, angle + reach) for joints, angle in self.scene.robot.init_state.joint_pos.items()
        }
        self.actions.joint_pos.clip = preset(default=self.actions.joint_pos.clip, newton_nerd=recorded)
        # The model carries the friction it was recorded with and stands for the nominal robot, so the startup events
        # that set friction, mass and centre of mass are off.
        self.events.physics_material = preset(default=self.events.physics_material, newton_nerd=None)
        self.events.add_base_mass = preset(default=self.events.add_base_mass, newton_nerd=None)
        self.events.base_com = preset(default=self.events.base_com, newton_nerd=None)
