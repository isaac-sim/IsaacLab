* Fixed :class:`~isaaclab.utils.noise.NoiseModelWithAdditiveBias` accumulating the bias across resets when
  ``bias_noise_cfg.operation`` is ``"add"`` (the default). Each reset now samples a fresh bias as documented, which
  matches the ``"abs"`` operation; configurations that use ``"abs"`` or ``"scale"`` are unchanged.
