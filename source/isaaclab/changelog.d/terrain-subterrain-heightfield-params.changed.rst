* **Breaking:** Height-field sub-terrains now retain non-None ``horizontal_scale``, ``vertical_scale``,
  and ``slope_threshold`` values instead of being overwritten by ``TerrainGeneratorCfg``.
  Set child fields to None to inherit the corresponding generator value. Child scale defaults
  remained 0.1 m horizontally and 0.005 m vertically. To disable slope correction for inheriting
  children, set the generator's ``slope_threshold`` to None.
