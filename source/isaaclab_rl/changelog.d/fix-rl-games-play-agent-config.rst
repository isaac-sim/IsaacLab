Fixed
^^^^^

* Rejected incompatible agent configurations in RL-Games playback before launching simulation,
  with guidance to select a matching RL library instead of failing with an opaque ``TypeError``.
  Camera feature presets use ``--rl_library rsl_rl --agent rsl_rl_cfg_entry_point`` with
  ``presets=resnet18`` or ``presets=theia_tiny``.
