Added
^^^^^

* Added :class:`~isaaclab_tasks.direct.cartpole.cartpole_camera_presets_env.CartpoleCameraPresetsEnv`,
  a subclass of :class:`~isaaclab_tasks.direct.cartpole.cartpole_camera_env.CartpoleCameraEnv` that
  wires :class:`~isaaclab.utils.buffers.CircularBuffer` into the ``Isaac-Cartpole-Camera-Presets-Direct-v0``
  task. ``frame_stack`` defaults to ``2`` for the Newton + Warp combo and ``1`` otherwise;
  CLI overrides via ``env.frame_stack=N`` are respected.
