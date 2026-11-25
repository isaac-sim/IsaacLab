# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_neural.physics import NeRDSolverCfg

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import preset

from .rough_env_cfg import AnymalCRoughEnvCfg

ANYMAL_C_NERD_TARGET_REACH = 0.75


@configclass
class AnymalCFlatEnvCfg(AnymalCRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # physics
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 120
        newton_mjwarp.solver_cfg.nconmax = 15
        newton_mjwarp.solver_cfg.cone = "elliptic"
        newton_mjwarp.solver_cfg.impratio = 100.0
        self.sim.physics.default = self.sim.physics.isaacsim_physx
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

        # ``physics=newton_nerd`` steps the robot with a FlashNeRD model, which stands for MuJoCo Warp and for the
        # actuator network of the legs.
        nerd = replace(
            newton_mjwarp,
            solver_cfg=NeRDSolverCfg(
                checkpoint="logs/neural/anymal_c_flat/flashnerd.pt",
                articulation=self.scene.robot.prim_path,
                precision="fp16",
            ),
            num_substeps=1,
            # A margin of 2 mm keeps the contact sensors on when a predicted foot hovers just above the ground.
            default_shape_cfg=replace(newton_mjwarp.default_shape_cfg, margin=0.002),
        )
        self.sim.physics = preset(**vars(self.sim.physics), newton_nerd=nerd)
        self.sim.dt = preset(default=self.sim.dt, newton_nerd=self.sim.dt * self.decimation)
        self.sim.render_interval = preset(default=self.sim.render_interval, newton_nerd=1)
        self.decimation = preset(default=self.decimation, newton_nerd=1)
        # The presets were built again, so the default is tied to its alternative once more.
        self.sim.physics.default = self.sim.physics.isaacsim_physx
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
