# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gymnasium import spaces

from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg
from isaaclab_tasks.utils import PresetCfg

###
# Observation space as Box
###


@dataclass
class BoxBoxEnvCfg(CartpoleEnvCfg):
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

    observation_space: Any = field(
        default_factory=lambda: spaces.Box(low=float("-inf"), high=float("inf"), shape=(4,))
    )  # or for simplicity: 4 or [4]
    action_space: Any = field(
        default_factory=lambda: spaces.Box(low=-1.0, high=1.0, shape=(1,))
    )  # or for simplicity: 1 or [1]


@dataclass
class BoxDiscreteEnvCfg(CartpoleEnvCfg):
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

    observation_space: Any = field(
        default_factory=lambda: spaces.Box(low=float("-inf"), high=float("inf"), shape=(4,))
    )  # or for simplicity: 4 or [4]
    action_space: Any = field(default_factory=lambda: spaces.Discrete(3))  # or for simplicity: {3}


@dataclass
class BoxMultiDiscreteEnvCfg(CartpoleEnvCfg):
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

    observation_space: Any = field(
        default_factory=lambda: spaces.Box(low=float("-inf"), high=float("inf"), shape=(4,))
    )  # or for simplicity: 4 or [4]
    action_space: Any = field(default_factory=lambda: spaces.MultiDiscrete([3, 2]))  # or for simplicity: [{3}, {2}]


###
# Observation space as Discrete
###


@dataclass
class DiscreteBoxEnvCfg(CartpoleEnvCfg):
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

    observation_space: Any = field(default_factory=lambda: spaces.Discrete(16))  # or for simplicity: {16}
    action_space: Any = field(
        default_factory=lambda: spaces.Box(low=-1.0, high=1.0, shape=(1,))
    )  # or for simplicity: 1 or [1]


@dataclass
class DiscreteDiscreteEnvCfg(CartpoleEnvCfg):
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

    observation_space: Any = field(default_factory=lambda: spaces.Discrete(16))  # or for simplicity: {16}
    action_space: Any = field(default_factory=lambda: spaces.Discrete(3))  # or for simplicity: {3}


@dataclass
class DiscreteMultiDiscreteEnvCfg(CartpoleEnvCfg):
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

    observation_space: Any = field(default_factory=lambda: spaces.Discrete(16))  # or for simplicity: {16}
    action_space: Any = field(default_factory=lambda: spaces.MultiDiscrete([3, 2]))  # or for simplicity: [{3}, {2}]


###
# Observation space as MultiDiscrete
###


@dataclass
class MultiDiscreteBoxEnvCfg(CartpoleEnvCfg):
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

    observation_space: Any = field(
        default_factory=lambda: spaces.MultiDiscrete([2, 2, 2, 2])
    )  # or for simplicity: [{2}, {2}, {2}, {2}]
    action_space: Any = field(
        default_factory=lambda: spaces.Box(low=-1.0, high=1.0, shape=(1,))
    )  # or for simplicity: 1 or [1]


@dataclass
class MultiDiscreteDiscreteEnvCfg(CartpoleEnvCfg):
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

    observation_space: Any = field(
        default_factory=lambda: spaces.MultiDiscrete([2, 2, 2, 2])
    )  # or for simplicity: [{2}, {2}, {2}, {2}]
    action_space: Any = field(default_factory=lambda: spaces.Discrete(3))  # or for simplicity: {3}


@dataclass
class MultiDiscreteMultiDiscreteEnvCfg(CartpoleEnvCfg):
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

    observation_space: Any = field(
        default_factory=lambda: spaces.MultiDiscrete([2, 2, 2, 2])
    )  # or for simplicity: [{2}, {2}, {2}, {2}]
    action_space: Any = field(default_factory=lambda: spaces.MultiDiscrete([3, 2]))  # or for simplicity: [{3}, {2}]


###
# Observation space as Dict
###


@dataclass
class DictBoxEnvCfg(CartpoleEnvCfg):
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

    observation_space: Any = field(
        default_factory=lambda: spaces.Dict(
            {
                "joint-positions": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
                "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
            }
        )
    )  # or for simplicity: {"joint-positions": 2, "joint-velocities": 2}
    action_space: Any = field(
        default_factory=lambda: spaces.Box(low=-1.0, high=1.0, shape=(1,))
    )  # or for simplicity: 1 or [1]


@dataclass
class DictDiscreteEnvCfg(CartpoleEnvCfg):
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

    observation_space: Any = field(
        default_factory=lambda: spaces.Dict(
            {
                "joint-positions": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
                "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
            }
        )
    )  # or for simplicity: {"joint-positions": 2, "joint-velocities": 2}
    action_space: Any = field(default_factory=lambda: spaces.Discrete(3))  # or for simplicity: {3}


@dataclass
class DictMultiDiscreteEnvCfg(CartpoleEnvCfg):
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

    observation_space: Any = field(
        default_factory=lambda: spaces.Dict(
            {
                "joint-positions": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
                "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
            }
        )
    )  # or for simplicity: {"joint-positions": 2, "joint-velocities": 2}
    action_space: Any = field(default_factory=lambda: spaces.MultiDiscrete([3, 2]))  # or for simplicity: [{3}, {2}]


###
# Observation space as Tuple
###


@dataclass
class TupleBoxEnvCfg(CartpoleEnvCfg):
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

    observation_space: Any = field(
        default_factory=lambda: spaces.Tuple(
            (
                spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
                spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
            )
        )
    )  # or for simplicity: (2, 2)
    action_space: Any = field(
        default_factory=lambda: spaces.Box(low=-1.0, high=1.0, shape=(1,))
    )  # or for simplicity: 1 or [1]


@dataclass
class TupleDiscreteEnvCfg(CartpoleEnvCfg):
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

    observation_space: Any = field(
        default_factory=lambda: spaces.Tuple(
            (
                spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
                spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
            )
        )
    )  # or for simplicity: (2, 2)
    action_space: Any = field(default_factory=lambda: spaces.Discrete(3))  # or for simplicity: {3}


@dataclass
class TupleMultiDiscreteEnvCfg(CartpoleEnvCfg):
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

    observation_space: Any = field(
        default_factory=lambda: spaces.Tuple(
            (
                spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
                spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
            )
        )
    )  # or for simplicity: (2, 2)
    action_space: Any = field(default_factory=lambda: spaces.MultiDiscrete([3, 2]))  # or for simplicity: [{3}, {2}]


##
# Consolidated PresetCfg
##


@dataclass
class CartpoleShowcasePresetsEnvCfg(PresetCfg):
    """Proprioceptive cartpole showcase with selectable observation/action space.

    Each variant attribute is an instance of an existing per-shape cfg class
    declared above. The hydra resolver picks one based on the
    ``presets=<name>`` CLI token; the default is ``box_box`` (matching the
    canonical cartpole shape). The retired per-shape task IDs are registered
    in the sibling ``__init__.py`` with a ``deprecated={"alias": ...}`` kwarg
    so ``parse_cfg.load_cfg_from_registry`` emits a ``FutureWarning`` when
    one of them is loaded.
    """

    box_box: Any = field(default_factory=BoxBoxEnvCfg)
    box_discrete: Any = field(default_factory=BoxDiscreteEnvCfg)
    box_multidiscrete: Any = field(default_factory=BoxMultiDiscreteEnvCfg)
    discrete_box: Any = field(default_factory=DiscreteBoxEnvCfg)
    discrete_discrete: Any = field(default_factory=DiscreteDiscreteEnvCfg)
    discrete_multidiscrete: Any = field(default_factory=DiscreteMultiDiscreteEnvCfg)
    multidiscrete_box: Any = field(default_factory=MultiDiscreteBoxEnvCfg)
    multidiscrete_discrete: Any = field(default_factory=MultiDiscreteDiscreteEnvCfg)
    multidiscrete_multidiscrete: Any = field(default_factory=MultiDiscreteMultiDiscreteEnvCfg)
    dict_box: Any = field(default_factory=DictBoxEnvCfg)
    dict_discrete: Any = field(default_factory=DictDiscreteEnvCfg)
    dict_multidiscrete: Any = field(default_factory=DictMultiDiscreteEnvCfg)
    tuple_box: Any = field(default_factory=TupleBoxEnvCfg)
    tuple_discrete: Any = field(default_factory=TupleDiscreteEnvCfg)
    tuple_multidiscrete: Any = field(default_factory=TupleMultiDiscreteEnvCfg)
    default: Any = field(default_factory=BoxBoxEnvCfg)  # canonical Cartpole shape: Box obs, Box action.
