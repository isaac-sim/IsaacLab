# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_assets import BOOSTER_K1_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, DeformableObjectCfg, DeformableObject, RigidObject, RigidObjectCfg, Articulation
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import os

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


@configclass
class BoosterK1EnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 22
    observation_space = 79
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/Field",
        terrain_type="usd",
        usd_path= os.path.expanduser("~/IsaacLab-nomadz/source/isaaclab_assets/data/Environment/Field.usd"),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Ball
    ball_cfg = DeformableObjectCfg(

        prim_path="/World/envs/env_.*/Ball",

        spawn=sim_utils.UsdFileCfg(

            usd_path = os.path.expanduser(
                "~/IsaacLab-nomadz/source/isaaclab_assets/data/Environment/Ball.usd"),

            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),

            mass_props= sim_utils.MassPropertiesCfg(mass=0.044, density=-1)
        ),

        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.06)),

        debug_vis=False,

    )

    # Goal blue
    goal_blue_cfg = RigidObjectCfg(
        prim_path = "/World/envs/env_.*/Goal_Blue",

        spawn=sim_utils.UsdFileCfg(

            usd_path = os.path.expanduser(
                "~/IsaacLab-nomadz/source/isaaclab_assets/data/Environment/Goal_Blue.usd"),

            rigid_props = sim_utils.RigidBodyPropertiesCfg(),
            mass_props = sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props = sim_utils.CollisionPropertiesCfg(),
        ),

        init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0, 0.0, 0.06),
                                                  rot=(0.671592,  0.120338,  0.712431,  -0.164089)
                                                  ),
    )

    # Goal red
    goal_red_cfg = RigidObjectCfg(
        prim_path = "/World/envs/env_.*/Goal_Red",

        spawn=sim_utils.UsdFileCfg(

            usd_path = os.path.expanduser(
                "~/IsaacLab-nomadz/source/isaaclab_assets/data/Environment/Goal_Red.usd"),

            rigid_props = sim_utils.RigidBodyPropertiesCfg(),
            mass_props = sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props = sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-10.0, 0.0, 0.06),
                                                  rot=(0.523403,  0.677758, -0.513991, -0.080530)
                                                  ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # TODO Correct these values
    joint_gears: list = [
        50.0,   # AAHead_yaw (head - moderate)
        50.0,   # Head_pitch (head - moderate)
    
        40.0,   # Left_Shoulder_Pitch (arms - lower effort)
        40.0,   # Right_Shoulder_Pitch
    
        40.0,   # Left_Shoulder_Roll
        40.0,   # Right_Shoulder_Roll
    
        40.0,   # Left_Elbow_Pitch
        40.0,   # Right_Elbow_Pitch
    
        40.0,   # Left_Elbow_Yaw
        40.0,   # Right_Elbow_Yaw
    
        
        45.0,   # Left_Hip_Pitch (legs - high effort)
        45.0,   # Right_Hip_Pitch
    
        35.0,   # Left_Hip_Roll
        35.0,   # Right_Hip_Roll
    
        35.0,   # Left_Hip_Yaw
        35.0,   # Right_Hip_Yaw
    
        60.0,   # Left_Knee_Pitch (highest effort)
        60.0,   # Right_Knee_Pitch
    
        25.0,   # Left_Ankle_Pitch (feet - moderate)
        25.0,   # Right_Ankle_Pitch
    
        15.0,   # Left_Ankle_Roll (feet - lower effort)
        15.0,   # Right_Ankle_Roll
    ]


    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.8

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01


class BoosterK1Env(LocomotionEnv):
    cfg: BoosterK1EnvCfg

    def __init__(self, cfg: BoosterK1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


    def _setup_scene(self):
         # Instantiate robot and objects before cloning environments to the scene
        self.robot = Articulation(self.cfg.robot)
        self.ball = DeformableObject(self.cfg.ball_cfg)
        self.goal_blue = RigidObject(self.cfg.goal_blue_cfg)
        self.goal_red = RigidObject(self.cfg.goal_red_cfg)

        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add objects to scene
        self.scene.deformable_objects["Ball"] = self.ball
        self.scene.rigid_objects["Goal_Red"] = self.goal_red
        self.scene.rigid_objects["Goal_Blue"] = self.goal_blue
    
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
     

    def _reset_idx(self, env_ids):

        super()._reset_idx(env_ids)
        
        # Reset Balls initial position
        ball_state = self.ball.data.default_nodal_state_w
        self.ball.write_nodal_state_to_sim(ball_state, env_ids=env_ids)

        # Reset goals position  (Shouldnt be needed once floor is designed)
        goal_blue_state = self.goal_blue.data.default_root_state
        goal_red_state = self.goal_red.data.default_root_state
        self.goal_blue.write_root_state_to_sim(goal_blue_state, env_ids)
        self.goal_red.write_root_state_to_sim(goal_red_state, env_ids)



        
