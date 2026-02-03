# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import thrust_actions


@configclass
class ThrustActionCfg(ActionTermCfg):
    """Configuration for the thrust action term.

    This configuration class specifies how policy actions are transformed into thruster
    commands for multirotor control. It provides extensive customization of the action
    processing pipeline including scaling, offsetting, and clipping.

    The action term is designed to work with :class:`~isaaclab_contrib.assets.Multirotor`
    assets and uses their thruster configuration to determine which thrusters to control.

    Key Configuration Options:
        - **scale**: Multiplies raw actions to adjust command magnitude
        - **offset**: Adds a baseline value (e.g., hover thrust) to actions
        - **clip**: Constrains actions to safe operational ranges
        - **use_default_offset**: Automatically uses hover thrust as offset

    Example Configurations:
        **Normalized thrust control around hover**:

        .. code-block:: python

            thrust_action = ThrustActionCfg(
                asset_name="robot",
                scale=2.0,  # Actions in [-1,1] become [-2,2] N
                use_default_offset=True,  # Add hover thrust (e.g., 5N)
                clip={".*": (0.0, 10.0)},  # Final thrust in [0, 10] N
            )

        **Direct thrust control with per-thruster scaling**:

        .. code-block:: python

            thrust_action = ThrustActionCfg(
                asset_name="robot",
                scale={
                    "rotor_[0-1]": 8.0,  # Front rotors: stronger
                    "rotor_[2-3]": 7.0,  # Rear rotors: weaker
                },
                offset=0.0,
                use_default_offset=False,
            )

        **Differential thrust control**:

        .. code-block:: python

            thrust_action = ThrustActionCfg(
                asset_name="robot",
                scale=3.0,
                use_default_offset=True,  # Center around hover
                clip={".*": (-2.0, 8.0)},  # Allow +/-2N deviation
            )

    .. seealso::
        - :class:`~isaaclab_contrib.mdp.actions.ThrustAction`: Implementation of this action term
        - :class:`~isaaclab.managers.ActionTermCfg`: Base action term configuration
    """

    class_type: type[ActionTerm] = thrust_actions.ThrustAction

    asset_name: str = MISSING
    """Name or regex expression of the asset that the action will be mapped to.

    This should match the name given to the multirotor asset in the scene configuration.
    For example, if the robot is defined as ``robot = MultirotorCfg(...)``, then
    ``asset_name`` should be ``"robot"``.
    """

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action. Default is ``1.0``, which means no scaling.

    This multiplies the raw action values to adjust the command magnitude. It can be:

    - A float: uniform scaling for all thrusters (e.g., ``2.0``)
    - A dict: per-thruster scaling using regex patterns (e.g., ``{"rotor_.*": 2.5}``)

    For normalized actions in [-1, 1], the scale determines the maximum deviation
    from the offset value.

    Example:
        .. code-block:: python

            # Uniform scaling
            scale = 5.0  # Actions of ±1 become ±5N

            # Per-thruster scaling
            scale = {
                "rotor_[0-1]": 8.0,   # Front rotors
                "rotor_[2-3]": 6.0,   # Rear rotors
            }
    """

    offset: float | dict[str, float] = 0.0
    """Offset factor for the action. Default is ``0.0``, which means no offset.

    This value is added to the scaled actions to establish a baseline thrust.
    It can be:

    - A float: uniform offset for all thrusters (e.g., ``5.0`` for 5N hover thrust)
    - A dict: per-thruster offset using regex patterns

    If :attr:`use_default_offset` is ``True``, this value is overwritten by the
    default thruster RPS from the multirotor configuration.

    Example:
        .. code-block:: python

            # Uniform offset (5N baseline thrust)
            offset = 5.0

            # Per-thruster offset
            offset = {
                "rotor_0": 5.2,
                "rotor_1": 4.8,
            }
    """

    clip: dict[str, tuple[float, float]] | None = None
    """Clipping ranges for processed actions. Default is ``None``, which means no clipping.

    This constrains the final thrust commands to safe operational ranges after
    scaling and offset are applied. It must be specified as a dictionary mapping
    regex patterns to (min, max) tuples.

    Example:
        .. code-block:: python

            # Clip all thrusters to [0, 10] N
            clip = {".*": (0.0, 10.0)}

            # Different limits for different thrusters
            clip = {
                "rotor_[0-1]": (0.0, 12.0),  # Front rotors
                "rotor_[2-3]": (0.0, 8.0),   # Rear rotors
            }

    """

    preserve_order: bool = False
    """Whether to preserve the order of the asset names in the action output. Default is ``False``.

    If ``True``, the thruster ordering matches the regex pattern order exactly.
    If ``False``, ordering is determined by the USD scene traversal order.
    """

    use_default_offset: bool = True
    """Whether to use default thrust configured in the multirotor asset as offset. Default is ``True``.

    If ``True``, the :attr:`offset` value is overwritten with the default thruster
    RPS values from :attr:`MultirotorCfg.init_state.rps`. This is useful for
    controlling thrust as deviations from the hover state.

    If ``False``, the manually specified :attr:`offset` value is used.
    """
