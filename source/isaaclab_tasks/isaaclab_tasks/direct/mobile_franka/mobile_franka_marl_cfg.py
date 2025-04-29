from isaaclab_assets.robots.ridgeback_franka import RIDGEBACK_FRANKA_PANDA_CFG
from isaaclab_assets.robots.mobile_franka import MOBILE_FRANKA_CFG

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
# from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

@configclass
class MobileFrankaMARLCfg(DirectMARLEnvCfg):
    # Environment settings
    decimation = 2
    episode_length_s = 500 / (120 / 2)  # Adjusted for control frequency
    possible_agents = ["franka", "base"]
    action_spaces = {"franka": 7, "base": 3}
    observation_spaces = {"franka": 40, "base": 40}
    state_space = -1

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            # solver_position_iteration_count=12,
            # solver_velocity_iteration_count=6,
            # contact_offset=0.005,
            # rest_offset=0.0,
            bounce_threshold_velocity=0.2,
            # enable_sleeping=True,
            # max_depenetration_velocity=1000.0,
        ),
    )

    # Robot configuration
    mobile_franka_cfg: ArticulationCfg = MOBILE_FRANKA_CFG.replace(prim_path="/World/envs/env_.*/MobileFranka").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            # rot=(0.7071068, 0.0, 0.7071068, 0.0),
            # rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        # solver_position_iteration_count=12,
        # solver_velocity_iteration_count=1,
        # enable_self_collisions=False,
        # enable_gyroscopic_forces=True,
    )

    actuated_joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7", 
    ]

    mobile_base_names = [
        "dummy_base_prismatic_x_joint",
        "dummy_base_prismatic_y_joint",
        "dummy_base_revolute_z_joint",
    ]

    xy_base_names = [
        "dummy_base_prismatic_x_joint",
        "dummy_base_prismatic_y_joint",
    ]

    finger_joint_names = [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    finger_body_names = [
        "panda_leftfinger",
        # "panda_finger2",
    ]

    # object configuration
    # target_cube_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/object",
    #     spawn=sim_utils.SphereCfg(
    #         radius=0.1,
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=8,
    #             solver_velocity_iteration_count=0,
    #             sleep_threshold=0.005,
    #             stabilization_threshold=0.0025,
    #             max_depenetration_velocity=1000.0,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(density=500.0),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
    # )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )

    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=3.0, replicate_physics=True)

    action_scale = 7.5
    dof_velocity_scale = 0.1
    max_base_pos = 3.0

    # Reward scales
    dist_reward_scale = 20
    rot_reward_scale = 0.5
    around_handle_reward_scale = 10.0
    open_reward_scale = 7.5
    finger_dist_reward_scale = 100.0
    action_penalty_scale = 0.01
    finger_close_reward_scale = 10.0
    act_moving_average = 1.0
    # Reset noise
    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.0
    reset_dof_vel_noise = 0.0
