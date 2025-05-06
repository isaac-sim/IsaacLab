from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity_rma.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_assets.robots.unitree import *  # isort: skip


@configclass
class Go2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        }

        # self.scene.robot.actuators = {
        #     "legs": DelayedPDActuatorCfg(
        #         joint_names_expr=[".*"],
        #         effort_limit=23.5,
        #         velocity_limit=30.0,
        #         stiffness=25, 
        #         damping=0.5, 
        #         min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
        #         max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        #     )
        # }
        
        self.scene.terrain.terrain_generator.sub_terrains["flat"].proportion = 0.7
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.3
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0

        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.15)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------Observations------------------------------
        
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------

        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------

        self.events.randomize_rigid_joint_mass = None
        self.events.randomize_rigid_body_inertia = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None

        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }

        # self.events.randomize_push_robot.params["velocity_range"] = {
        #     "x": (-3.0, 3.0), 
        #     "y": (-3.0, 3.0)
        # }

        # ------------------------------Rewards------------------------------

        # self.rewards.foot_clearance.weight = 0.0
        # self.rewards.gait.weight = 1.0
        # self.rewards.joint_pos.weight = 0.0
        # self.rewards.contact_forces.weight = -1e-3
        # self.rewards.feet_contact_without_cmd.weight = 0.25


        self.rewards.foot_clearance.weight = 0.0
        self.rewards.gait.weight = 0.0
        self.rewards.joint_pos.weight = 0.0
        # self.rewards.contact_forces.weight = -1e-3
        # self.rewards.feet_contact_without_cmd.weight = 0.25
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.base_height_l2.weight = -10.0
        self.rewards.action_smoothness.weight = 0.0
        self.rewards.air_time_variance.weight = 0.0
        self.rewards.base_motion.weight = 0.0
        self.rewards.base_orientation.weight = 0.0
        self.rewards.foot_slip.weight = 0.0

        # ------------------------------Terminations------------------------------

        self.terminations.base_contact = None

        # ------------------------------Commands------------------------------

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 2.0) 
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)

        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0) 
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        if self.__class__.__name__ == "Go2RoughEnvCfg":
            self.disable_zero_weight_rewards()
        
