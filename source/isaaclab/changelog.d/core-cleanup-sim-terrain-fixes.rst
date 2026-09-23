Fixed
^^^^^

* Fixed the built-in :class:`~isaaclab.terrains.MeshRepeatedBoxesTerrainCfg`,
  :class:`~isaaclab.terrains.MeshRepeatedCylindersTerrainCfg`, and
  :class:`~isaaclab.terrains.MeshRepeatedPyramidsTerrainCfg` raising ``ValueError`` with their default
  ``object_type``. The resolvable ``"module:function"`` default was looked up as ``make_<object_type>``
  instead of being called.
* Fixed :func:`~isaaclab.sim.schemas.modify_articulation_root_properties` looking up the existing fixed
  joint on the current stage instead of the ``stage`` argument when ``fix_root_link`` is set.
* Fixed :func:`~isaaclab.sim.utils.queries.find_global_fixed_joint_prim` raising ``AttributeError`` for an
  :class:`pxr.Sdf.Path` argument.
* Fixed :class:`~isaaclab.sim.converters.MeshConverter` raising ``ValueError`` for mesh file names with
  more than one dot.
