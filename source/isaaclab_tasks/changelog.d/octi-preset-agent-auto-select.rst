Fixed
^^^^^

* Fixed :func:`~isaaclab_tasks.utils.setup_preset_cli` ignoring the agent metadata
  tasks already declare in the Gym registry. When ``--agent`` is left at its
  default, the entry point is now resolved from ``agent_preset_compatibility``
  for the active ``presets=`` token, and from the sole registered entry point
  when a task does not register ``<library>_cfg_entry_point`` at all. Previously
  ``presets=box_discrete`` on ``IsaacContrib-Cartpole-Showcase-Direct`` loaded the
  Gaussian-policy default and crashed on a shape mismatch, and the
  ``IsaacContrib-Humanoid-AMP-*`` tasks crashed on the unregistered
  ``skrl_cfg_entry_point``. Selection is skipped whenever the metadata is
  ambiguous or ``--agent`` was passed explicitly, so no existing command line
  changes meaning.
* Fixed the default-absent rule being unreachable whenever any ``presets=`` token
  was present. Benchmark sweeps broadcast the physics backend through that same
  token, so ``presets=newton_mjwarp`` left the AMP tasks crashing as before.
* Fixed agent auto-selection being unreachable for ``rsl_rl``, ``rl_games``, and
  ``sb3``, whose ``--agent`` defaults to ``<library>_cfg_entry_point`` rather than
  to ``None``. ``Isaac-Cartpole-Camera`` now honors the feature-extractor configs
  it declares for ``presets=resnet18`` and ``presets=theia_tiny``.
