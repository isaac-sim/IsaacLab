# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Dict, Optional, Sequence, Tuple

from omni.isaac.orbit.utils import configclass


@configclass
class ArticulatedObjectCfg:
    """Configuration parameters for an articulated object."""

    @configclass
    class MetaInfoCfg:
        """Meta-information about the manipulator."""

        usd_path: str = MISSING
        """USD file to spawn asset from."""
        scale: Tuple[float] = (1.0, 1.0, 1.0)
        """Scale of the object. Default to (1.0, 1.0, 1.0)."""
        sites_names: Optional[Sequence[str]] = None
        """Name of the sites to track (added to :obj:`data`). Defaults to :obj:`None`."""

    @configclass
    class RigidBodyPropertiesCfg:
        """Properties to apply to all rigid bodies in the articulation."""

        linear_damping: Optional[float] = None
        """Linear damping coefficient."""
        angular_damping: Optional[float] = None
        """Angular damping coefficient."""
        max_linear_velocity: Optional[float] = 1000.0
        """Maximum linear velocity for rigid bodies (in m/s). Defaults to 1000.0."""
        max_angular_velocity: Optional[float] = 1000.0
        """Maximum angular velocity for rigid bodies (in rad/s). Defaults to 1000.0."""
        max_depenetration_velocity: Optional[float] = 10.0
        """Maximum depenetration velocity permitted to be introduced by the solver (in m/s).
        Defaults to 10.0."""
        disable_gravity: Optional[bool] = False
        """Disable gravity for the actor. Defaults to False."""
        retain_accelerations: Optional[bool] = None
        """Carries over forces/accelerations over sub-steps."""

    @configclass
    class CollisionPropertiesCfg:
        """Properties to apply to all collisions in the articulation."""

        collision_enabled: Optional[bool] = None
        """Whether to enable or disable collisions."""
        contact_offset: Optional[float] = None
        """Contact offset for the collision shape."""
        rest_offset: Optional[float] = None
        """Rest offset for the collision shape."""
        torsional_patch_radius: Optional[float] = None
        """Radius of the contact patch for applying torsional friction."""
        min_torsional_patch_radius: Optional[float] = None
        """Minimum radius of the contact patch for applying torsional friction."""

    @configclass
    class ArticulationRootPropertiesCfg:
        """Properties to apply to articulation."""

        enable_self_collisions: Optional[bool] = None
        """Whether to enable or disable self-collisions."""
        solver_position_iteration_count: Optional[int] = None
        """Solver position iteration counts for the body."""
        solver_velocity_iteration_count: Optional[int] = None
        """Solver position iteration counts for the body."""

    @configclass
    class InitialStateCfg:
        """Initial state of the robot."""

        # root state
        pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Position of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` of the root in simulation world frame.
        Defaults to (1.0, 0.0, 0.0, 0.0).
        """
        lin_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Linear velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        ang_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        # dof state
        dof_pos: Dict[str, float] = MISSING
        """DOF positions of all joints."""
        dof_vel: Dict[str, float] = MISSING
        """DOF velocities of all joints."""

    ##
    # Initialize configurations.
    ##

    meta_info: MetaInfoCfg = MetaInfoCfg()
    """Meta-information about the articulated object."""
    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the articulated object."""
    rigid_props: RigidBodyPropertiesCfg = RigidBodyPropertiesCfg()
    """Properties to apply to all rigid bodies in the articulation."""
    collision_props: CollisionPropertiesCfg = CollisionPropertiesCfg()
    """Properties to apply to all collisions in the articulation."""
    articulation_props: ArticulationRootPropertiesCfg = ArticulationRootPropertiesCfg()
    """Properties to apply to articulation."""
