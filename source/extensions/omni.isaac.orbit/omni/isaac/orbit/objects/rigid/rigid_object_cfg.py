# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Optional, Tuple

from omni.isaac.orbit.utils import configclass


@configclass
class RigidObjectCfg:
    """Configuration parameters for a robot."""

    @configclass
    class MetaInfoCfg:
        """Meta-information about the manipulator."""

        usd_path: str = MISSING
        """USD file to spawn asset from."""
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        """Scale to spawn the object with. Defaults to (1.0, 1.0, 1.0)."""

    @configclass
    class RigidBodyPropertiesCfg:
        """Properties to apply to the rigid body."""

        solver_position_iteration_count: Optional[int] = None
        """Solver position iteration counts for the body."""
        solver_velocity_iteration_count: Optional[int] = None
        """Solver position iteration counts for the body."""
        max_linear_velocity: Optional[float] = 1000.0
        """Maximum linear velocity for rigid bodies (in m/s). Defaults to 1000.0."""
        max_angular_velocity: Optional[float] = 1000.0
        """Maximum angular velocity for rigid bodies (in rad/s). Defaults to 1000.0."""
        max_depenetration_velocity: Optional[float] = 10.0
        """Maximum depenetration velocity permitted to be introduced by the solver (in m/s).
        Defaults to 10.0."""
        disable_gravity: Optional[bool] = False
        """Disable gravity for the actor. Defaults to False."""

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
    class PhysicsMaterialCfg:
        """Physics material applied to the rigid object."""

        prim_path: str = "/World/Materials/rigidMaterial"
        """Path to the physics material prim. Defaults to /World/Materials/rigidMaterial.

        Note:
            If the prim path is not absolute, it will be resolved relative to the path specified when spawning
            the object.
        """
        static_friction: float = 0.5
        """Static friction coefficient. Defaults to 0.5."""
        dynamic_friction: float = 0.5
        """Dynamic friction coefficient. Defaults to 0.5."""
        restitution: float = 0.0
        """Restitution coefficient. Defaults to 0.0."""

    @configclass
    class InitialStateCfg:
        """Initial state of the rigid body."""

        # root position
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

    ##
    # Initialize configurations.
    ##

    meta_info: MetaInfoCfg = MetaInfoCfg()
    """Meta-information about the rigid object."""
    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the rigid object."""
    rigid_props: RigidBodyPropertiesCfg = RigidBodyPropertiesCfg()
    """Properties to apply to all rigid bodies in the object."""
    collision_props: CollisionPropertiesCfg = CollisionPropertiesCfg()
    """Properties to apply to all collisions in the articulation."""
    physics_material: Optional[PhysicsMaterialCfg] = PhysicsMaterialCfg()
    """Settings for the physics material to apply to the rigid object.

    If set to None, no physics material will be created and applied.
    """
