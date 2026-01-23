# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets.articulation.articulation_data import ArticulationData


class MultirotorData(ArticulationData):
    """Data container for a multirotor articulation.

    This class extends the base :class:`~isaaclab.assets.ArticulationData` container to include
    multirotor-specific data such as thruster states, thrust targets, and computed forces.
    It provides access to all the state information needed for monitoring and controlling
    multirotor vehicles.

    The data container is automatically created and managed by the :class:`~isaaclab_contrib.assets.Multirotor`
    class. Users typically access this data through the :attr:`Multirotor.data` property.

    Note:
        All tensor attributes have shape ``(num_instances, num_thrusters)`` where
        ``num_instances`` is the number of environment instances and ``num_thrusters``
        is the total number of thrusters per multirotor.

    .. seealso::
        - :class:`~isaaclab.assets.ArticulationData`: Base articulation data container
        - :class:`~isaaclab_contrib.assets.Multirotor`: Multirotor asset class
    """

    thruster_names: list[str] = None
    """List of thruster names in the multirotor.

    This list contains the ordered names of all thrusters, matching the order used
    for indexing in the thrust tensors. The names correspond to the USD body prim names
    matched by the thruster name expressions in the actuator configuration.

    Example:
        ``["rotor_0", "rotor_1", "rotor_2", "rotor_3"]`` for a quadcopter
    """

    default_thruster_rps: torch.Tensor = None
    """Default thruster RPS (revolutions per second) state of all thrusters. Shape is (num_instances, num_thrusters).

    This quantity is configured through the :attr:`MultirotorCfg.init_state.rps` parameter
    and represents the baseline/hover RPS for each thruster. It is used to initialize
    thruster states during reset operations.

    For a hovering multirotor, these values should produce enough collective thrust
    to counteract gravity.

    Example:
        For a 1kg quadcopter with 4 thrusters, if each thruster produces 2.5N at 110 RPS,
        the default might be ``[[110.0, 110.0, 110.0, 110.0]]`` for hover.
    """

    thrust_target: torch.Tensor = None
    """Thrust targets commanded by the user or controller. Shape is ``(num_instances, num_thrusters)``

    This quantity contains the target thrust values set through the
    :meth:`~isaaclab_contrib.assets.Multirotor.set_thrust_target` method or by
    action terms in RL environments. These targets are processed by the thruster
    actuator models to compute actual applied thrusts.

    The units depend on the actuator model configuration (typically Newtons for
    force or RPS for rotational speed).
    """

    ##
    # Thruster commands
    ##

    computed_thrust: torch.Tensor = None
    """Computed thrust from the actuator model before clipping. Shape is (num_instances, num_thrusters).

    This quantity contains the thrust values computed by the thruster actuator models
    before any clipping or saturation is applied. It represents the "desired" thrust
    based on the actuator dynamics (rise/fall times) but may exceed physical limits.

    The difference between :attr:`computed_thrust` and :attr:`applied_thrust` indicates
    when the actuator is saturating at its limits.

    Example Use:
        Monitor actuator saturation by comparing computed vs applied thrust:

        .. code-block:: python

            saturation = multirotor.data.computed_thrust - multirotor.data.applied_thrust
            is_saturated = saturation.abs() > 1e-6
    """

    applied_thrust: torch.Tensor = None
    """Applied thrust from the actuator model after clipping. Shape is (num_instances, num_thrusters).

    This quantity contains the final thrust values that are actually applied to the
    simulation after all actuator model processing, including:

    - Dynamic response (rise/fall time constants)
    - Clipping to thrust range limits
    - Any other actuator model constraints

    This is the "ground truth" thrust that affects the multirotor's motion in the
    physics simulation.
    """
