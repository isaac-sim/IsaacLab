from __future__ import annotations

from gymnasium import spaces

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


def get_tiled_camera_cfg(data_type: str, width: int = 100, height: int = 100) -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=[data_type],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=width,
        height=height,
    )


@configclass
class CartpoleCameraBaseEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # change viewer settings
    viewer = ViewerCfg(eye=(20.0, 20.0, 20.0))

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=20.0, replicate_physics=True)

    # control
    max_effort = 100.0  # [N]

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.125, 0.125]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005


###
# Observation space as Box
###


@configclass
class BoxBoxEnvCfg(CartpoleCameraBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Box`` with shape (height, width, 3))

        ===  ===
        Idx  Observation
        ===  ===
        -    RGB image
        ===  ===

    * Action space (``~gymnasium.spaces.Box`` with shape (1,))

        ===  ===
        Idx  Action
        ===  ===
        0    Cart DOF effort scale: [-1, 1]
        ===  ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Box(
        low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)
    )  # or for simplicity: [height, width, 3]
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class BoxDiscreteEnvCfg(CartpoleCameraBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Box`` with shape (height, width, 3))

        ===  ===
        Idx  Observation
        ===  ===
        -    RGB image
        ===  ===

    * Action space (``~gymnasium.spaces.Discrete`` with 3 elements)

        ===  ===
        N    Action
        ===  ===
        0    Zero cart DOF effort
        1    Negative maximum cart DOF effort
        2    Positive maximum cart DOF effort
        ===  ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Box(
        low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)
    )  # or for simplicity: [height, width, 3]
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class BoxMultiDiscreteEnvCfg(CartpoleCameraBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Box`` with shape (height, width, 3))

        ===  ===
        Idx  Observation
        ===  ===
        -    RGB image
        ===  ===

    * Action space (``~gymnasium.spaces.MultiDiscrete`` with 2 discrete spaces)

        ===  ===
        N    Action (Discrete 0)
        ===  ===
        0    Zero cart DOF effort
        1    Half of maximum cart DOF effort
        2    Maximum cart DOF effort
        ===  ===

        ===  ===
        N    Action (Discrete 1)
        ===  ===
        0    Negative effort (one side)
        1    Positive effort (other side)
        ===  ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Box(
        low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)
    )  # or for simplicity: [height, width, 3]
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]


###
# Observation space as Dict
###


@configclass
class DictBoxEnvCfg(CartpoleCameraBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Dict`` with 2 constituent spaces)

        ================  ===
        Key               Observation
        ================  ===
        joint-velocities  DOF velocities
        camera            RGB image
        ================  ===

    * Action space (``~gymnasium.spaces.Box`` with shape (1,))

        ===  ===
        Idx  Action
        ===  ===
        0    Cart DOF effort scale: [-1, 1]
        ===  ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Dict({
        "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        "camera": spaces.Box(low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)),
    })  # or for simplicity: {"joint-velocities": 2, "camera": [height, width, 3]}
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class DictDiscreteEnvCfg(CartpoleCameraBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Dict`` with 2 constituent spaces)

        ================  ===
        Key               Observation
        ================  ===
        joint-velocities  DOF velocities
        camera            RGB image
        ================  ===

    * Action space (``~gymnasium.spaces.Discrete`` with 3 elements)

        ===  ===
        N    Action
        ===  ===
        0    Zero cart DOF effort
        1    Negative maximum cart DOF effort
        2    Positive maximum cart DOF effort
        ===  ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Dict({
        "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        "camera": spaces.Box(low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)),
    })  # or for simplicity: {"joint-velocities": 2, "camera": [height, width, 3]}
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class DictMultiDiscreteEnvCfg(CartpoleCameraBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Dict`` with 2 constituent spaces)

        ================  ===
        Key               Observation
        ================  ===
        joint-velocities  DOF velocities
        camera            RGB image
        ================  ===

    * Action space (``~gymnasium.spaces.MultiDiscrete`` with 2 discrete spaces)

        ===  ===
        N    Action (Discrete 0)
        ===  ===
        0    Zero cart DOF effort
        1    Half of maximum cart DOF effort
        2    Maximum cart DOF effort
        ===  ===

        ===  ===
        N    Action (Discrete 1)
        ===  ===
        0    Negative effort (one side)
        1    Positive effort (other side)
        ===  ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Dict({
        "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        "camera": spaces.Box(low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)),
    })  # or for simplicity: {"joint-velocities": 2, "camera": [height, width, 3]}
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]
