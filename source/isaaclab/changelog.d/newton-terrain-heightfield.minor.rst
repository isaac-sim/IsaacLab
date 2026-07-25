Added
^^^^^

* Added a ``newton:heightfield:resolution`` attribute on generated terrain colliders in
  :class:`~isaaclab.terrains.TerrainImporter`, recording the terrain's horizontal grid spacing so
  backends that support heightfield collision can replace the terrain collision mesh with an
  equivalent heightfield. The attribute is inert for backends that do not consume it.
* Added :attr:`~isaaclab.terrains.SubTerrainBaseCfg.convert_to_heightfield` to control whether a
  sub-terrain can be collided against as a heightfield. Height field sub-terrains
  (:class:`~isaaclab.terrains.height_field.HfTerrainBaseCfg`) default it to True since the conversion
  is exact for them, while mesh sub-terrains default it to False since the conversion is lossy. The
  terrain is only converted when every sub-terrain enables the flag.
