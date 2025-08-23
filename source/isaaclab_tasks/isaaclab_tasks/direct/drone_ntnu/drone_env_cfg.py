import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class DroneEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 1
    action_scale = 0.5
    action_space = 12
    observation_space = 48
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="https://isaac-dev.ov.nvidia.com/omni/web3/omniverse://isaac-dev.ov.nvidia.com/Users/zhengyuz@nvidia.com/Robots/NTNU/Quad/quad.usd"
        )
    )

    
    # custom variables
    thrust_to_torque_ratio = 0.01
    motor_directions = [1, -1, 1, -1]
    allocation_matrix = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [-0.13, -0.13, 0.13, 0.13], [-0.13, 0.13, 0.13, -0.13], [-0.01, 0.01, -0.01, 0.01]]
    
    #drag_variables
    body_vel_linear_damping_coefficient = [0.0, 0.0, 0.0]
    body_vel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]
    angvel_linear_damping_coefficient = [0.0, 0.0, 0.0]
    angvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]
    
    # disturbance
    enable_disturbance = False
    max_force_and_torque_disturbance = [0.75, 0.75, 0.75, 0.004, 0.004, 0.004]
    disturbance_probability = 0.02