Added
^^^^^
* Added :attr:`~isaaclab.sim.converters.UrdfConverterCfg.ros_package_paths`,
  :attr:`~isaaclab.sim.converters.UrdfConverterCfg.robot_type`,
  :attr:`~isaaclab.sim.converters.UrdfConverterCfg.run_asset_transformer`,
  :attr:`~isaaclab.sim.converters.UrdfConverterCfg.run_multi_physics_conversion`, and
  :attr:`~isaaclab.sim.converters.UrdfConverterCfg.debug_mode` config fields that mirror the
  new :class:`isaacsim.asset.importer.urdf.URDFImporterConfig` options.
* Extended :attr:`~isaaclab.sim.converters.UrdfConverterCfg.collision_type` to accept
  ``"Bounding Sphere"`` and ``"Bounding Cube"`` in addition to the existing ``"Convex Hull"``
  and ``"Convex Decomposition"`` values.
* Added :attr:`~isaaclab.sim.converters.MjcfConverterCfg.fix_base`,
  :attr:`~isaaclab.sim.converters.MjcfConverterCfg.link_density`,
  :attr:`~isaaclab.sim.converters.MjcfConverterCfg.robot_type`,
  :attr:`~isaaclab.sim.converters.MjcfConverterCfg.override_gain_type`,
  :attr:`~isaaclab.sim.converters.MjcfConverterCfg.override_bias_type`,
  :attr:`~isaaclab.sim.converters.MjcfConverterCfg.override_gain_prm`,
  :attr:`~isaaclab.sim.converters.MjcfConverterCfg.override_bias_prm`,
  :attr:`~isaaclab.sim.converters.MjcfConverterCfg.run_asset_transformer`,
  :attr:`~isaaclab.sim.converters.MjcfConverterCfg.run_multi_physics_conversion`, and
  :attr:`~isaaclab.sim.converters.MjcfConverterCfg.debug_mode` config fields that mirror the
  new :class:`isaacsim.asset.importer.mjcf.MJCFImporterConfig` options.

Changed
^^^^^^^
* Refactored :class:`~isaaclab.sim.converters.UrdfConverter` to delegate the full conversion
  pipeline to :class:`isaacsim.asset.importer.urdf.URDFImporter` /
  :class:`isaacsim.asset.importer.urdf.URDFImporterConfig`. The duplicated IsaacLab
  implementations of ``_apply_fix_base``, ``_apply_link_density``, ``_apply_joint_drives``,
  ``_set_drive_type_on_joints``, ``_set_target_type_on_joints``, ``_set_drive_gains_on_joints``,
  and ``_fix_articulation_root_for_fixed_base`` have been removed and replaced with a thin
  translation layer that maps :class:`~isaaclab.sim.converters.UrdfConverterCfg` onto the
  Isaac Sim importer config. All behaviour is preserved.
* Updated :class:`~isaaclab.sim.converters.MjcfConverter` to forward the full set of
  :class:`isaacsim.asset.importer.mjcf.MJCFImporterConfig` options to the Isaac Sim importer.

Removed
^^^^^^^
* Removed :func:`~isaaclab.sim.converters.urdf_utils.merge_fixed_joints` as it is now handled by the Isaac Sim URDF importer.
