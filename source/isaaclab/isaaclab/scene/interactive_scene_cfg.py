# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass


@configclass
class InteractiveSceneCfg:
    """Configuration for the interactive scene.

    The users can inherit from this class to add entities to their scene. This is then parsed by the
    :class:`InteractiveScene` class to create the scene.

    .. note::
        The adding of entities to the scene is sensitive to the order of the attributes in the configuration.
        Please make sure to add the entities in the order you want them to be added to the scene.
        The recommended order of specification is terrain, physics-related assets (articulations and rigid bodies),
        sensors and non-physics-related assets (lights).

    For example, to add a robot to the scene, the user can create a configuration class as follows:

    .. code-block:: python

        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors.ray_caster import GridPatternCfg, RayCasterCfg
        from isaaclab.utils import configclass

        from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

        @configclass
        class MySceneCfg(InteractiveSceneCfg):

            # terrain - flat terrain plane
            terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
            )

            # articulation - robot 1
            robot_1 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
            # articulation - robot 2
            robot_2 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
            robot_2.init_state.pos = (0.0, 1.0, 0.6)

            # sensor - ray caster attached to the base of robot 1 that scans the ground
            height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot_1/base",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                attach_yaw_only=True,
                pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
                debug_vis=True,
                mesh_prim_paths=["/World/ground"],
            )

            # extras - light
            light = AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
            )

    """

    num_envs: int = MISSING
    """Number of environment instances handled by the scene."""

    env_spacing: float = MISSING
    """Spacing between environments.

    This is the default distance between environment origins in the scene. Used only when the
    number of environments is greater than one.
    """

    lazy_sensor_update: bool = True
    """Whether to update sensors only when they are accessed. Default is True.

    If true, the sensor data is only updated when their attribute ``data`` is accessed. Otherwise, the sensor
    data is updated every time sensors are updated.
    """

    replicate_physics: bool = True
    """Enable/disable replication of physics schemas when using the Cloner APIs. Default is True.

    If True, the simulation will have the same asset instances (USD prims) in all the cloned environments.
    Internally, this ensures optimization in setting up the scene and parsing it via the physics stage parser.

    If False, the simulation allows having separate asset instances (USD prims) in each environment.
    This flexibility comes at a cost of slowdowns in setting up and parsing the scene.

    .. note::
        Optimized parsing of certain prim types (such as deformable objects) is not currently supported
        by the physics engine. In these cases, this flag needs to be set to False.
    """

    filter_collisions: bool = True
    """Enable/disable collision filtering between cloned environments. Default is True.

    If True, collisions will not occur between cloned environments.

    If False, the simulation will generate collisions between environments.

    .. note::
        Collisions can only be filtered automatically in direct workflows when physics replication is enabled.
        If ``replicated_physics=False`` and collision filtering is desired, make sure to call ``scene.filter_collisions()``.
    """
