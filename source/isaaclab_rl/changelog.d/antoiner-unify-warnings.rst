Changed
^^^^^^^

* Changed the RSL-RL configuration migration in :func:`~isaaclab_rl.rsl_rl.handle_deprecated_rsl_rl_cfg`
  to emit ``FutureWarning`` for deprecated configuration fields and ``logger.warning`` for settings the
  installed rsl-rl version does not support, instead of printing ``[WARNING]`` lines.
