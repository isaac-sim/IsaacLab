# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.sim.converters.asset_converter_base_cfg import AssetConverterBaseCfg
from isaaclab.utils import configclass


@configclass
class UrdfConverterCfg(AssetConverterBaseCfg):
    """The configuration class for UrdfConverter."""

    @configclass
    class JointDriveCfg:

        @configclass
        class PDGainsCfg:
            """Configuration for the PD gains of the drive."""

            stiffness: dict[str, float] | float = MISSING
            """The stiffness of the joint drive in Nm/rad or N/rad.

            If None, the stiffness is set to the value parsed from the URDF file.
            If :attr:`~UrdfConverterCfg.JointDriveCfg.target_type` is set to ``"velocity"``, this value determines
            the drive strength in joint velocity space.
            """

            damping: dict[str, float] | float | None = None
            """The damping of the joint drive in Nm/(rad/s) or N/(rad/s). Defaults to None.

            If None, the damping is set to the value parsed from the URDF file or 0.0 if no value is found in the URDF.
            If :attr:`~UrdfConverterCfg.JointDriveCfg.target_type` is set to ``"velocity"``, this attribute is set to
            0.0 and :attr:`stiffness` serves as the drive's strength in joint velocity space.
            """

        @configclass
        class NaturalFrequencyGainsCfg:
            r"""Configuration for the natural frequency gains of the drive.

            Computes the joint drive stiffness and damping based on the desired natural frequency using the formula:

            :math:`P = m \cdot f^2`, :math:`D = 2 \cdot r \cdot f \cdot m`

            where :math:`f` is the natural frequency, :math:`r` is the damping ratio, and :math:`m` is the total
            equivalent inertia at the joint. The damping ratio is such that:

            * :math:`r = 1.0` is a critically damped system,
            * :math:`r < 1.0` is underdamped,
            * :math:`r > 1.0` is overdamped.
            """

            natural_frequency: dict[str, float] | float = MISSING
            """The natural frequency of the joint drive.

            If :attr:`~UrdfConverterCfg.JointDriveCfg.target_type` is set to ``"velocity"``, this value determines the
            drive's natural frequency in joint velocity space.
            """

            damping_ratio: dict[str, float] | float = 0.005
            """The damping ratio of the joint drive. Defaults to 0.005.

            If :attr:`~UrdfConverterCfg.JointDriveCfg.target_type` is set to ``"velocity"``, this value is ignored and
            only :attr:`natural_frequency` is used.
            """

        drive_type: dict[str, Literal["acceleration", "force"]] | Literal["acceleration", "force"] = "force"
        """The drive type used for the joint. Defaults to ``"force"``.

        * ``"acceleration"``: The joint drive normalizes the inertia before applying the joint effort so it's invariant
          to inertia and mass changes (equivalent to ideal damped oscillator).
        * ``"force"``: Applies effort through forces, so is subject to variations on the body inertia.
        """

        target_type: dict[str, Literal["none", "position", "velocity"]] | Literal["none", "position", "velocity"] = (
            "position"
        )
        """The drive target type used for the joint. Defaults to ``"position"``.

        If the target type is set to ``"none"``, the joint stiffness and damping are set to 0.0.
        """

        gains: PDGainsCfg | NaturalFrequencyGainsCfg = PDGainsCfg()
        """The drive gains configuration."""

    fix_base: bool = MISSING
    """Create a fix joint to the root/base link."""

    root_link_name: str | None = None
    """The name of the root link. Defaults to None.

    If None, the root link will be set by PhysX.
    """

    link_density: float = 0.0
    """Default density in ``kg/m^3`` for links whose ``"inertial"`` properties are missing in the URDF. Defaults to 0.0."""

    merge_fixed_joints: bool = True
    """Consolidate links that are connected by fixed joints. Defaults to True."""

    convert_mimic_joints_to_normal_joints: bool = False
    """Convert mimic joints to normal joints. Defaults to False."""

    joint_drive: JointDriveCfg | None = JointDriveCfg()
    """The joint drive settings.

    None can be used for URDFs without joints.
    """

    collision_from_visuals = False
    """Create collision geometry from visual geometry."""

    collider_type: Literal["convex_hull", "convex_decomposition"] = "convex_hull"
    """The collision shape simplification. Defaults to ``"convex_hull"``.

    * ``"convex_hull"``: The collision shape is simplified to a convex hull.
    * ``"convex_decomposition"``: The collision shape is decomposed into smaller convex shapes for a closer fit.
    """

    self_collision: bool = False
    """Activate self-collisions between links of the articulation. Defaults to False."""

    replace_cylinders_with_capsules: bool = False
    """Replace cylinder shapes with capsule shapes. Defaults to False."""
