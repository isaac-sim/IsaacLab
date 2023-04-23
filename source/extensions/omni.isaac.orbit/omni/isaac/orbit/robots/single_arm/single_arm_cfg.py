# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Optional, Sequence, Tuple

from omni.isaac.orbit.utils import configclass

from ..robot_base_cfg import RobotBaseCfg


@configclass
class SingleArmManipulatorCfg(RobotBaseCfg):
    """Properties for a single arm manipulator."""

    @configclass
    class MetaInfoCfg(RobotBaseCfg.MetaInfoCfg):
        """Meta-information about the manipulator."""

        arm_num_dof: int = MISSING
        """Number of degrees of freedom of arm."""
        tool_num_dof: int = MISSING
        """Number of degrees of freedom of tool."""
        tool_sites_names: Optional[Sequence[str]] = None
        """Name of the tool sites to track (added to :obj:`data`). Defaults to :obj:`None`.

        If :obj:`None`, then no tool sites are tracked. The returned tensor for tool sites state
        is in the same order as that of the provided list.
        """

    @configclass
    class EndEffectorFrameCfg:
        """Information about the end-effector frame location."""

        body_name: str = MISSING
        """Name of the body corresponding to the end-effector."""
        pos_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Additional position offset from the body frame. Defaults to (0, 0, 0)."""
        rot_offset: Tuple[float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Additional rotation offset ``(w, x, y, z)`` from the body frame. Defaults to (1, 0, 0, 0)."""

    @configclass
    class DataInfoCfg:
        """Information about what all data to read from simulator.

        Note: Setting all of them to true leads to an overhead of 10-15%.
        """

        enable_jacobian: bool = False
        """Fill in jacobian for the end-effector into data buffers. Defaults to False."""
        enable_mass_matrix: bool = False
        """Fill in mass matrix into data buffers. Defaults to False."""
        enable_coriolis: bool = False
        """Fill in coriolis and centrifugal forces into data buffers. Defaults to False."""
        enable_gravity: bool = False
        """Fill in generalized gravity forces into data buffers. Defaults to False."""

    @configclass
    class PhysicsMaterialCfg:
        """Physics material applied to the tool sites of the robot."""

        prim_path = "/World/Materials/toolMaterial"
        """Path to the physics material prim. Defaults to /World/Materials/toolMaterial.

        Note:
            If the prim path is not absolute, it will be resolved relative to the path specified when spawning
            the object.
        """
        static_friction: float = 1.0
        """Static friction coefficient. Defaults to 1.0."""
        dynamic_friction: float = 1.0
        """Dynamic friction coefficient. Defaults to 1.0."""
        restitution: float = 0.0
        """Restitution coefficient. Defaults to 0.0."""

    ##
    # Initialize configurations.
    ##

    meta_info: MetaInfoCfg = MetaInfoCfg()
    """Meta-information about the manipulator."""
    ee_info: EndEffectorFrameCfg = EndEffectorFrameCfg()
    """Information about the end-effector frame location."""
    data_info: DataInfoCfg = DataInfoCfg()
    """Information about what all data to read from simulator."""
    physics_material: PhysicsMaterialCfg = PhysicsMaterialCfg()
    """Physics material applied to the tool sites of the robot."""
