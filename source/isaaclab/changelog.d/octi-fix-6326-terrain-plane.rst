Fixed
^^^^^

* Fixed :func:`~isaaclab.sim.spawners.from_files.spawn_ground_plane` raising ``ValueError`` for
  ground-plane USD files that do not follow the layout of the default grid asset from Isaac Sim
  (issue `#6326 <https://github.com/isaac-sim/IsaacLab/issues/6326>`__). The physics material is now
  bound to all collision-enabled prims in the spawned asset instead of requiring a ``Plane``-typed
  prim, and the color override is skipped with a warning when the asset does not contain the grid
  shader.
* Fixed :meth:`~isaaclab.terrains.TerrainImporter.import_ground_plane` overriding the ground plane's
  color with black when :attr:`~isaaclab.terrains.TerrainImporterCfg.visual_material` is set to None.
  The spawned asset's authored color is now kept unchanged in this case, matching the documented
  semantics of :attr:`~isaaclab.sim.spawners.from_files.GroundPlaneCfg.color`.
