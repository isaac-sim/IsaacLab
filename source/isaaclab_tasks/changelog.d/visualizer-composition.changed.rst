* **Breaking:** Moved task visualizer presets into ``VisualizerCfg.backend``. Configure common view
  settings once with ``VisualizerCfg(..., backend=MultiBackendVisualizerCfg())`` instead of backend-specific
  viewer declarations or ``SimulationCfg.default_visualizer_cfg``.
* Added ``isaaclab_tasks.utils.launch_simulation`` for standalone scripts declaring presets. It composed
  typed launcher selections before core launch, preserving custom backend settings without per-script resolution calls.
