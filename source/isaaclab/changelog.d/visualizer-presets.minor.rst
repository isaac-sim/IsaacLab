Changed
^^^^^^^

* Selected visualizers through ``SimulationCfg.visualizer_cfgs`` presets before launch, removing
  runtime viewer-name factories and settings-based selection. Used ``visualizer=NAME`` or its
  ``--viz`` / ``--visualizer`` aliases; declared custom alternatives with ``PresetCfg``.
* Moved config-only ``PresetCfg``, ``preset``, and ``resolve_presets`` into ``isaaclab.utils``;
  retained the existing ``isaaclab_tasks.utils`` imports.

Fixed
^^^^^

* Applied distributed and runtime-selected devices to the ``SimulationCfg`` found in the config tree,
  including standalone configs and configs nested in dictionaries or lists, in ``launch_simulation``.
