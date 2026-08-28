Changed
^^^^^^^

* Changed the default ground plane to a bundled warm-white asset with NVIDIA-green 1 m grid lines.
  Its metric UVs now follow the requested plane size, and plane terrains bound the visual mesh to the
  environment grid while retaining an infinite collision plane.
  Set :attr:`isaaclab.sim.spawners.from_files.from_files_cfg.GroundPlaneCfg.color` or provide a
  custom USD path to override its authored appearance.
