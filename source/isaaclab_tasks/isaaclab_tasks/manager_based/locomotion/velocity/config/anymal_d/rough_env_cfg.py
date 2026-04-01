# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.sim import SimulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    EventsCfg,
    LocomotionVelocityRoughEnvCfg,
    StartupEventsCfg,
)
from isaaclab_tasks.utils import PresetCfg, preset

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip


@configclass
class RoughPhysicsCfg(PresetCfg):
    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    # WAR: Rough terrain requires pyramidal cone — elliptic cone + high impratio diverges
    # in float32 with many mesh contacts due to MuJoCo Warp single-precision limitations.
    # Upstream (open): https://github.com/google-deepmind/mujoco_warp/issues/1000
    # Uses Newton collision pipeline since MuJoCo's built-in collision does not support
    # mesh terrain geometry.
    newton = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=200,
            nconmax=100,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
            ccd_iterations=100,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_250_000),
        num_substeps=1,
        debug_mode=False,
    )
    physx = default


@configclass
class AnymalDPhysxEventsCfg(EventsCfg, StartupEventsCfg):
    pass

@configclass
class AnymalDNewtonEventsCfg(EventsCfg):

    collider_offsets = EventTerm(
        func=mdp.randomize_rigid_body_collider_offsets,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "contact_offset_distribution_params": (0.01, 0.01),
            "rest_offset_distribution_params": (0.01, 0.01),
        },
    )

@configclass
class AnymalDEventsCfg(PresetCfg):
    default = AnymalDPhysxEventsCfg()
    newton = AnymalDNewtonEventsCfg()
    physx = default


@configclass
class AnymalDRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=RoughPhysicsCfg())
    events: AnymalDEventsCfg = AnymalDEventsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["legs"].armature = preset(default=0.0, newton=0.05, physx=0.0)

@configclass
class AnymalDRoughEnvCfg_PLAY(AnymalDRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
