Changed
^^^^^^^

* Changed the default ground plane to a warm-white hosted asset with NVIDIA-green 1 m grid lines.
  Its metric UVs now follow the requested plane size, and plane terrains bound the visual mesh to the
  environment grid while retaining an infinite collision plane.
  Set :attr:`isaaclab.sim.spawners.from_files.from_files_cfg.GroundPlaneCfg.color` to tint its
  diffuse component, or provide a custom USD path to replace the authored appearance. Existing
  generated terrains retain their dark material by default; explicitly set
  :attr:`isaaclab.terrains.terrain_importer_cfg.TerrainImporterCfg.visual_material` to ``None`` to
  leave a generated terrain without a bound visual material.
