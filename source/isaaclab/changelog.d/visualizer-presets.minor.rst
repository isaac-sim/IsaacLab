Changed
^^^^^^^

* Selected visualizers through ``SimulationCfg.visualizer_cfgs`` presets before launch, removing
  runtime viewer-name factories and settings-based selection. Used ``visualizer=NAME`` or its
  ``--viz`` / ``--visualizer`` aliases; declared custom alternatives with ``PresetCfg``.
* Moved config-only ``PresetCfg``, ``preset``, and ``resolve_presets`` into ``isaaclab.utils``;
  retained the existing ``isaaclab_tasks.utils`` imports.

Fixed
^^^^^

* Applied distributed and runtime-selected devices to standalone ``SimulationCfg`` inputs as well
  as environment configs in ``launch_simulation``.
