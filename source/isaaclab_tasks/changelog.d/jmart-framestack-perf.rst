Changed
^^^^^^^

* Changed :class:`~isaaclab_tasks.direct.cartpole.cartpole_camera_env.CartpoleCameraEnv`
  and its presets subclass to route image normalization through
  :func:`isaaclab.utils.images.normalize_camera_image` and defer the normalize past the
  frame-stack buffer for RGB-like data types, improving cartpole-camera frame-stacking
  throughput.
