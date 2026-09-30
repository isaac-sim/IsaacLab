Changed
^^^^^^^

* Changed the RSL-RL configuration migration in :func:`~isaaclab_rl.rsl_rl.handle_deprecated_rsl_rl_cfg`
  to emit ``FutureWarning`` for deprecated configuration fields and ``logger.warning`` for settings the
  installed rsl-rl version does not support, instead of printing ``[WARNING]`` lines.
* Changed ``[INFO]`` messages printed by the training, playback, export, and simple-agent entry points
  and the Weights & Biases helpers to ``logger.info``. They still print as ``[INFO]: <message>`` by default.
