Added
^^^^^

* Added a ``newton:heightfield:resolution`` attribute on generated terrain colliders in
  :class:`~isaaclab.terrains.TerrainImporter`, recording the terrain's horizontal grid spacing so
  backends that support heightfield collision can replace the terrain collision mesh with an
  equivalent heightfield. The attribute is inert for backends that do not consume it.
* Added :attr:`~isaaclab.terrains.SubTerrainBaseCfg.convert_to_heightfield` to control whether a
  sub-terrain is converted to a heightfield. Height field sub-terrains
  (:class:`~isaaclab.terrains.height_field.HfTerrainBaseCfg`) default it to True since the conversion
  is exact for them, while mesh sub-terrains default it to False since the conversion is lossy. The
  terrain is only converted when every sub-terrain enables the flag.

Changed
^^^^^^^

* Changed the mesh sub-terrains in ``ROUGH_TERRAINS_CFG`` to enable
  :attr:`~isaaclab.terrains.SubTerrainBaseCfg.convert_to_heightfield`, so the rough locomotion terrain
  is collided against as a heightfield on backends that support it. Set the flag back to False on the
  ``pyramid_stairs``, ``pyramid_stairs_inv``, or ``boxes`` sub-terrains to collide against the original
  mesh instead.
