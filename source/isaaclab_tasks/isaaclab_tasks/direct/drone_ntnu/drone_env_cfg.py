import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class DroneEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 5.0
    decimation = 1
    action_space = 4
    observation_space = 13
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zhengyuz/Projects/IsaacLab/source/isaaclab_assets/data/Robots/NTNU/quad.usd"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={}
    )

    # custom variables
    thrust_to_torque_ratio = 0.01
    allocation_matrix = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [-0.13, -0.13, 0.13, 0.13], [-0.13, 0.13, 0.13, -0.13], [-0.01, 0.01, -0.01, 0.01]]
    application_mask = [5, 6, 7, 8]
    force_application_level = 'motor_link'
    motor_directions = [1, -1, 1, -1]
    
    #drag_variables
    body_vel_linear_damping_coefficient = [0.0, 0.0, 0.0]
    body_vel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]
    angvel_linear_damping_coefficient = [0.0, 0.0, 0.0]
    angvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]
    
    # disturbance
    enable_disturbance = False
    max_force_and_torque_disturbance = [0.75, 0.75, 0.75, 0.004, 0.004, 0.004]
    disturbance_probability = 0.02