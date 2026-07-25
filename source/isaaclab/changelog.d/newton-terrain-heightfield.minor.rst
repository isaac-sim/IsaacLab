Added
^^^^^

* Added a ``newton:heightfield:resolution`` attribute on generated terrain colliders in
  :class:`~isaaclab.terrains.TerrainImporter`, recording the terrain's horizontal grid spacing so
  backends that support heightfield collision can replace the terrain collision mesh with an
  equivalent heightfield. The attribute is inert for backends that do not consume it.
