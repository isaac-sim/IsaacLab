from __future__ import annotations

from gymnasium import spaces

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class CartpoleBaseEnvCfg(DirectRLEnvCfg):
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # control
    max_effort = 100.0  # [N]

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]

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
class BoxBoxEnvCfg(CartpoleBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Box`` with shape (4,))

        ===  ===
        Idx  Observation
        ===  ===
        0    Pole DOF position
        1    Pole DOF velocity
        2    Cart DOF position
        3    Cart DOF velocity
        ===  ===

    * Action space (``~gymnasium.spaces.Box`` with shape (1,))

        ===  ===
        Idx  Action
        ===  ===
        0    Cart DOF effort scale: [-1, 1]
        ===  ===
    """

    observation_space = spaces.Box(low=float("-inf"), high=float("inf"), shape=(4,))  # or for simplicity: 4 or [4]
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class BoxDiscreteEnvCfg(CartpoleBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Box`` with shape (4,))

        ===  ===
        Idx  Observation
        ===  ===
        0    Pole DOF position
        1    Pole DOF velocity
        2    Cart DOF position
        3    Cart DOF velocity
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

    observation_space = spaces.Box(low=float("-inf"), high=float("inf"), shape=(4,))  # or for simplicity: 4 or [4]
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class BoxMultiDiscreteEnvCfg(CartpoleBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Box`` with shape (4,))

        ===  ===
        Idx  Observation
        ===  ===
        0    Pole DOF position
        1    Pole DOF velocity
        2    Cart DOF position
        3    Cart DOF velocity
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

    observation_space = spaces.Box(low=float("-inf"), high=float("inf"), shape=(4,))  # or for simplicity: 4 or [4]
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]


###
# Observation space as Discrete
###


@configclass
class DiscreteBoxEnvCfg(CartpoleBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Discrete`` with 16 elements)

        ===  ===
        N    Observation (Value signs: pole position, cart position, pole velocity, cart velocity)
        ===  ===
        0    - - - -
        1    - - - +
        2    - - + -
        3    - - + +
        4    - + - -
        5    - + - +
        6    - + + -
        7    - + + +
        8    + - - -
        9    + - - +
        10   + - + -
        11   + - + +
        12   + + - -
        13   + + - +
        14   + + + -
        15   + + + +
        ===  ===

    * Action space (``~gymnasium.spaces.Box`` with shape (1,))

        ===  ===
        Idx  Action
        ===  ===
        0    Cart DOF effort scale: [-1, 1]
        ===  ===
    """

    observation_space = spaces.Discrete(16)  # or for simplicity: {16}
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class DiscreteDiscreteEnvCfg(CartpoleBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Discrete`` with 16 elements)

        ===  ===
        N    Observation (Value signs: pole position, cart position, pole velocity, cart velocity)
        ===  ===
        0    - - - -
        1    - - - +
        2    - - + -
        3    - - + +
        4    - + - -
        5    - + - +
        6    - + + -
        7    - + + +
        8    + - - -
        9    + - - +
        10   + - + -
        11   + - + +
        12   + + - -
        13   + + - +
        14   + + + -
        15   + + + +
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

    observation_space = spaces.Discrete(16)  # or for simplicity: {16}
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class DiscreteMultiDiscreteEnvCfg(CartpoleBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Discrete`` with 16 elements)

        ===  ===
        N    Observation (Value signs: pole position, cart position, pole velocity, cart velocity)
        ===  ===
        0    - - - -
        1    - - - +
        2    - - + -
        3    - - + +
        4    - + - -
        5    - + - +
        6    - + + -
        7    - + + +
        8    + - - -
        9    + - - +
        10   + - + -
        11   + - + +
        12   + + - -
        13   + + - +
        14   + + + -
        15   + + + +
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

    observation_space = spaces.Discrete(16)  # or for simplicity: {16}
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]


###
# Observation space as MultiDiscrete
###


@configclass
class MultiDiscreteBoxEnvCfg(CartpoleBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.MultiDiscrete`` with 4 discrete spaces)

        ===  ===
        N    Observation (Discrete 0)
        ===  ===
        0    Negative pole position (-)
        1    Zero or positive pole position (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 1)
        ===  ===
        0    Negative cart position (-)
        1    Zero or positive cart position (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 2)
        ===  ===
        0    Negative pole velocity (-)
        1    Zero or positive pole velocity (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 3)
        ===  ===
        0    Negative cart velocity (-)
        1    Zero or positive cart velocity (+)
        ===  ===

    * Action space (``~gymnasium.spaces.Box`` with shape (1,))

        ===  ===
        Idx  Action
        ===  ===
        0    Cart DOF effort scale: [-1, 1]
        ===  ===
    """

    observation_space = spaces.MultiDiscrete([2, 2, 2, 2])  # or for simplicity: [{2}, {2}, {2}, {2}]
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class MultiDiscreteDiscreteEnvCfg(CartpoleBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.MultiDiscrete`` with 4 discrete spaces)

        ===  ===
        N    Observation (Discrete 0)
        ===  ===
        0    Negative pole position (-)
        1    Zero or positive pole position (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 1)
        ===  ===
        0    Negative cart position (-)
        1    Zero or positive cart position (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 2)
        ===  ===
        0    Negative pole velocity (-)
        1    Zero or positive pole velocity (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 3)
        ===  ===
        0    Negative cart velocity (-)
        1    Zero or positive cart velocity (+)
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

    observation_space = spaces.MultiDiscrete([2, 2, 2, 2])  # or for simplicity: [{2}, {2}, {2}, {2}]
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class MultiDiscreteMultiDiscreteEnvCfg(CartpoleBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.MultiDiscrete`` with 4 discrete spaces)

        ===  ===
        N    Observation (Discrete 0)
        ===  ===
        0    Negative pole position (-)
        1    Zero or positive pole position (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 1)
        ===  ===
        0    Negative cart position (-)
        1    Zero or positive cart position (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 2)
        ===  ===
        0    Negative pole velocity (-)
        1    Zero or positive pole velocity (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 3)
        ===  ===
        0    Negative cart velocity (-)
        1    Zero or positive cart velocity (+)
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

    observation_space = spaces.MultiDiscrete([2, 2, 2, 2])  # or for simplicity: [{2}, {2}, {2}, {2}]
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]


###
# Observation space as Dict
###


@configclass
class DictBoxEnvCfg(CartpoleBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Dict`` with 2 constituent spaces)

        ================  ===
        Key               Observation
        ================  ===
        joint-positions   DOF positions
        joint-velocities  DOF velocities
        ================  ===

    * Action space (``~gymnasium.spaces.Box`` with shape (1,))

        ===  ===
        Idx  Action
        ===  ===
        0    Cart DOF effort scale: [-1, 1]
        ===  ===
    """

    observation_space = spaces.Dict({
        "joint-positions": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    })  # or for simplicity: {"joint-positions": 2, "joint-velocities": 2}
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class DictDiscreteEnvCfg(CartpoleBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Dict`` with 2 constituent spaces)

        ================  ===
        Key               Observation
        ================  ===
        joint-positions   DOF positions
        joint-velocities  DOF velocities
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

    observation_space = spaces.Dict({
        "joint-positions": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    })  # or for simplicity: {"joint-positions": 2, "joint-velocities": 2}
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class DictMultiDiscreteEnvCfg(CartpoleBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Dict`` with 2 constituent spaces)

        ================  ===
        Key               Observation
        ================  ===
        joint-positions   DOF positions
        joint-velocities  DOF velocities
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

    observation_space = spaces.Dict({
        "joint-positions": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    })  # or for simplicity: {"joint-positions": 2, "joint-velocities": 2}
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]


###
# Observation space as Tuple
###


@configclass
class TupleBoxEnvCfg(CartpoleBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Tuple`` with 2 constituent spaces)

        ===  ===
        Idx  Observation
        ===  ===
        0    DOF positions
        1    DOF velocities
        ===  ===

    * Action space (``~gymnasium.spaces.Box`` with shape (1,))

        ===  ===
        Idx  Action
        ===  ===
        0    Cart DOF effort scale: [-1, 1]
        ===  ===
    """

    observation_space = spaces.Tuple((
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    ))  # or for simplicity: (2, 2)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class TupleDiscreteEnvCfg(CartpoleBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Tuple`` with 2 constituent spaces)

        ===  ===
        Idx  Observation
        ===  ===
        0    DOF positions
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

    observation_space = spaces.Tuple((
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    ))  # or for simplicity: (2, 2)
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class TupleMultiDiscreteEnvCfg(CartpoleBaseEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Tuple`` with 2 constituent spaces)

        ===  ===
        Idx  Observation
        ===  ===
        0    DOF positions
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

    observation_space = spaces.Tuple((
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    ))  # or for simplicity: (2, 2)
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]
