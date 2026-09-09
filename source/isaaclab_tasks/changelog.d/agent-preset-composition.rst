Changed
^^^^^^^

* Changed environment-coupled agent variants to participate directly in the shared preset resolution pass through
  each library's canonical ``<library>_cfg_entry_point``. Commands selecting Cartpole camera features or showcase
  observation/action spaces now need only ``presets=<name>`` and no preset-specific ``--agent`` value.

Removed
^^^^^^^

* Removed registry-side agent/preset compatibility metadata and agent auto-selection. Removed ``agent_library`` from
  :func:`~isaaclab_tasks.utils.setup_preset_cli`; agent configuration families should declare matching root-level
  :class:`~isaaclab_tasks.utils.PresetCfg` alternatives instead.
