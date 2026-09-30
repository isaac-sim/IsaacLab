Changed
^^^^^^^

* **Breaking:** Height-field sub-terrains now retain their own ``horizontal_scale``, ``vertical_scale``,
  and ``slope_threshold`` values. Moved these fields from ``TerrainGeneratorCfg`` to each
  ``HfTerrainBaseCfg`` when configuring a terrain generator. Newton users can set
  ``TerrainImporterCfg.heightfield_collider_resolution`` to control the resolution of the combined collider.
