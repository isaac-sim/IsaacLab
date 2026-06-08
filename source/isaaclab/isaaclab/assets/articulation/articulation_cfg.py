# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from isaaclab.actuators import ActuatorBaseCfg
from isaaclab.utils.configclass import configclass

from ..asset_base_cfg import AssetBaseCfg

if TYPE_CHECKING:
    from .articulation import Articulation


@configclass
class ArticulationCfg(AssetBaseCfg):
    """Configuration parameters for an articulation."""

    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        """Initial state of the articulation."""

        # root velocity
        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Linear velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

        # joint state
        joint_pos: dict[str, float] = {".*": 0.0}
        """Joint positions of the joints. Defaults to 0.0 for all joints."""
        joint_vel: dict[str, float] = {".*": 0.0}
        """Joint velocities of the joints. Defaults to 0.0 for all joints."""

    ##
    # Initialize configurations.
    ##

    class_type: type[Articulation] | str = "{DIR}.articulation:Articulation"

    articulation_root_prim_path: str | None = None
    """Path to the articulation root prim under the :attr:`prim_path`. Defaults to None, in which case the class
    will search for a prim with the USD ArticulationRootAPI on it.

    This path should be relative to the :attr:`prim_path` of the asset. If the asset is loaded from a USD file,
    this path should be relative to the root of the USD stage. For instance, if the loaded USD file at :attr:`prim_path`
    contains two articulations, one at `/robot1` and another at `/robot2`, and you want to use `robot2`,
    then you should set this to `/robot2`.

    The path must start with a slash (`/`).
    """

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the articulated object. Defaults to identity pose with zero velocity and zero joint state."""

    soft_joint_pos_limit_factor: float = 1.0
    """Fraction specifying the range of joint position limits (parsed from the asset) to use. Defaults to 1.0.

    The soft joint position limits are scaled by this factor to specify a safety region within the simulated
    joint position limits. This isn't used by the simulation, but is useful for learning agents to prevent the joint
    positions from violating the limits, such as for termination conditions.

    The soft joint position limits are accessible through the :attr:`ArticulationData.soft_joint_pos_limits` attribute.
    """

    actuators: dict[str, ActuatorBaseCfg] = MISSING
    """Actuators for the robot with corresponding joint names."""

    actuator_value_resolution_debug_print = False
    """Print the resolution of actuator final value when input cfg is different from USD value, Defaults to False
    """

    def _post_spawn(self, stage: Any) -> None:
        """Author ``NewtonActuator`` USD prims from :attr:`actuators` after spawn.

        Invoked by :class:`~isaaclab.assets.AssetBase` once the articulation's prims
        exist on the stage. Delegates to
        :func:`~isaaclab.sim.schemas.define_actuator_properties`, which gates itself
        on ``sim_cfg.use_newton_actuators`` and silently no-ops when the simulation
        is not configured for Newton-native actuators.
        """
        if self.actuators is MISSING:
            return
        from isaaclab.sim.schemas.schemas_actuators import define_actuator_properties  # noqa: PLC0415

        # In InteractiveScene, articulated assets are often spawned first under
        # a template path (for example ``/World/template/Robot``) and cloned
        # into ``{ENV_REGEX_NS}`` later. Author NewtonActuator prims on the
        # actual spawned source prim so clones inherit them.
        author_prim_path = (
            self.spawn.spawn_path if self.spawn is not None and self.spawn.spawn_path is not None else self.prim_path
        )
        define_actuator_properties(author_prim_path, self.actuators, stage=stage)
