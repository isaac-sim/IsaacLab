# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.cartpole.cartpole_camera_env import CartpoleRGBCameraEnvCfg as CartpoleCameraEnvCfg


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


###
# Observation space as Box
###


@configclass
class BoxBoxEnvCfg(CartpoleCameraEnvCfg):
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
class BoxDiscreteEnvCfg(CartpoleCameraEnvCfg):
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
class BoxMultiDiscreteEnvCfg(CartpoleCameraEnvCfg):
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
class DictBoxEnvCfg(CartpoleCameraEnvCfg):
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
class DictDiscreteEnvCfg(CartpoleCameraEnvCfg):
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
class DictMultiDiscreteEnvCfg(CartpoleCameraEnvCfg):
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


###
# Observation space as Tuple
###


@configclass
class TupleBoxEnvCfg(CartpoleCameraEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Tuple`` with 2 constituent spaces)

        ===  ===
        Idx  Observation
        ===  ===
        0    RGB image
        1    DOF velocities
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
    observation_space = spaces.Tuple((
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)),
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    ))  # or for simplicity: ([height, width, 3], 2)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class TupleDiscreteEnvCfg(CartpoleCameraEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Tuple`` with 2 constituent spaces)

        ===  ===
        Idx  Observation
        ===  ===
        0    RGB image
        1    DOF velocities
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
    observation_space = spaces.Tuple((
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)),
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    ))  # or for simplicity: ([height, width, 3], 2)
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class TupleMultiDiscreteEnvCfg(CartpoleCameraEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Tuple`` with 2 constituent spaces)

        ===  ===
        Idx  Observation
        ===  ===
        0    RGB image
        1    DOF velocities
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
    observation_space = spaces.Tuple((
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)),
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    ))  # or for simplicity: ([height, width, 3], 2)
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]
