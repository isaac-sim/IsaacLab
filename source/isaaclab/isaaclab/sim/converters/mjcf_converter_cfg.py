# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Literal

from isaaclab.sim.converters.asset_converter_base_cfg import AssetConverterBaseCfg
from isaaclab.utils.configclass import configclass


@configclass
class MjcfConverterCfg(AssetConverterBaseCfg):
    """The configuration class for MjcfConverter.

    Maps to :class:`~isaacsim.asset.importer.mjcf.MJCFImporterConfig` from the Isaac Sim
    MJCF importer. All post-import USD edits (fix-base, density override, actuator gain
    overrides, self-collision, mesh merging, asset transformer profile) are performed by
    the Isaac Sim importer — this config just forwards the user's choices.

    .. note::
        From Isaac Sim 5.0 onwards, the MJCF importer was rewritten to use the
        ``mujoco-usd-converter`` library. The :attr:`AssetConverterBaseCfg.make_instanceable`
        setting from the base class is not supported by the new MJCF importer and is ignored.
    """

    merge_mesh: bool = False
    """Merge meshes where possible to optimize the model. Defaults to False."""

    collision_from_visuals: bool = False
    """Generate collision geometry from visual geometries. Defaults to False."""

    collision_type: Literal["Convex Hull", "Convex Decomposition", "Bounding Sphere", "Bounding Cube"] = "Convex Hull"
    """Type of collision geometry to use. Defaults to ``"Convex Hull"``.

    Supported values match the ``collision_type`` field of
    :class:`~isaacsim.asset.importer.mjcf.MJCFImporterConfig`.
    """

    self_collision: bool = False
    """Activate self-collisions between links of the articulation. Defaults to False."""

    import_physics_scene: bool = False
    """Import the physics scene (time step per second, gravity, etc.) from the MJCF file. Defaults to False."""

    fix_base: bool = False
    """Add a fixed joint from the world to the root rigid-body link. Defaults to False.

    When enabled, :class:`~isaacsim.asset.importer.mjcf.MJCFImporter` inserts a ``FixedJoint``
    between the world and the articulation root and relocates ``ArticulationRootAPI`` onto the
    appropriate ancestor prim so PhysX treats the articulation as fixed-base.
    """

    link_density: float = 0.0
    """Default density in ``kg/m^3`` for links whose ``"inertial"`` properties are missing.
    Defaults to 0.0.

    A value of ``0.0`` leaves density unchanged.
    """

    robot_type: str = "Default"
    """Robot type applied by the USD robot schema. Defaults to ``"Default"``.

    Supported types are: ``Default``, ``End Effector``, ``Manipulator``, ``Humanoid``, ``Wheeled``,
    ``Holonomic``, ``Quadruped``, ``Mobile Manipulators``, ``Aerial``.
    Forwarded to :class:`~isaacsim.asset.importer.mjcf.MJCFImporterConfig`.
    """

    override_gain_type: str | None = None
    """MuJoCo actuator gain type override (e.g. ``"fixed"``). Defaults to ``None``.

    ``None`` leaves the value parsed from the MJCF file unchanged. See
    :func:`isaacsim.asset.importer.utils.impl.asset_utils.apply_mjc_actuator_gains` for
    the supported encodings.
    """

    override_bias_type: str | None = None
    """MuJoCo actuator bias type override (e.g. ``"affine"``). Defaults to ``None``.

    ``None`` leaves the value parsed from the MJCF file unchanged.
    """

    override_gain_prm: list[float] | None = None
    """MuJoCo actuator gain parameter array override. Defaults to ``None``.

    Mujoco models actuators using an affine transformation, which is a linear combination of the
    gain parameters, control, and bias.

    The affine transformation is defined as:
    tau = gain @ control + bias

    ``None`` leaves the value parsed from the MJCF file unchanged. Example for position
    control: ``[kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]``.
    """

    override_bias_prm: list[float] | None = None
    """MuJoCo actuator bias parameter array override. Defaults to ``None``.

    ``None`` leaves the value parsed from the MJCF file unchanged. Example for position
    control: ``[0, -kp, -kd, 0, 0, 0, 0, 0, 0, 0]``.
    """

    run_asset_transformer: bool = True
    """Run the asset transformation profile to convert the flattened USD into a layered USD asset. Defaults to True.

    After running this profile, the USD asset will be a layered USD asset with the following structure:
    - robot_name.usda (interface usd)
    - payloads/base.usda (base usd with links, meshes, and materials)
    - payloads/instances.usda (usd with visual and collision geometry)
    - payloads/geometry.usd (binary usd with meshes)
    - payloads/materials.usda (materials)
    - payloads/Physics/physics.usda (neutral physics format)
    - payloads/Physics/physX.usda (PhysX attributes)
    - payloads/Physics/mujoco.usda (MuJoCo attributes)


    """

    run_multi_physics_conversion: bool = True
    """Enable to convert compatible MuJoCo attributes to PhysX attributes, such as actuator gains. Defaults to True."""

    debug_mode: bool = False
    """Enable debug mode in the underlying MJCF importer. Defaults to False.

    When enabled, the importer writes intermediate conversion artifacts next to the output
    USD for inspection instead of using a temporary scratch directory.
    """
