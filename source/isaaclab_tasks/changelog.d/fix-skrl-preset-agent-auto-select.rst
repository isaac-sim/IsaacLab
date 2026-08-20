Fixed
^^^^^

* :func:`~isaaclab_tasks.utils.setup_preset_cli` now automatically resolves the
  correct agent entry point from ``agent_preset_compatibility`` when a preset token
  (e.g. ``presets=box_discrete``) is active and no explicit ``--agent`` flag is
  given. Previously, skrl training on tasks with multiple space presets (such as
  ``IsaacContrib-Cartpole-Showcase-Direct``) always loaded the default
  ``skrl_cfg_entry_point`` regardless of the active preset, causing a shape mismatch
  crash when a non-box action space was selected.
