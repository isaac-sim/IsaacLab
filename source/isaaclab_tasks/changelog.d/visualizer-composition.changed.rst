* **Breaking:** Moved task visualizer presets into ``VisualizerCfg.backend``. Configure common view
  settings once with ``VisualizerCfg(..., backend=MultiBackendVisualizerCfg())`` instead of backend-specific
  viewer declarations or ``SimulationCfg.default_visualizer_cfg``. Unselected visualizer presets resolved to ``[]``.
* Composed standalone visualizer declarations before calling the core launcher, preserving declared backend options.
