Changed
^^^^^^^

* Changed the default ``size`` argument of :meth:`~isaaclab.terrains.TerrainImporter.import_ground_plane`
  from ``(2.0e6, 2.0e6)`` to ``(2.0e3, 2.0e3)`` to avoid numerical precision issues. The issue was
  originally reported against `newton-physics/newton#3977
  <https://github.com/newton-physics/newton/issues/3977>`_, but was isolated to Isaac Lab's default
  ground plane size.
