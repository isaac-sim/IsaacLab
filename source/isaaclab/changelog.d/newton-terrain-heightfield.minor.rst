Added
^^^^^

* Added a ``newton:heightfield:resolution`` attribute on generated terrain colliders in
  :class:`~isaaclab.terrains.TerrainImporter`, recording the terrain's horizontal grid spacing so
  backends that support heightfield collision can replace the terrain collision mesh with an
  equivalent heightfield. The attribute is inert for backends that do not consume it.
* Added :attr:`~isaaclab.terrains.TerrainGeneratorCfg.convert_to_heightfield` to opt terrains that
  contain mesh sub-terrains into heightfield collision. Terrains built only from height field
  sub-terrains are converted automatically, since the conversion is exact for them. The flag defaults
  to False because the conversion is lossy for mesh sub-terrains.
