Changed
^^^^^^^

* Changed :class:`~isaaclab_tasks.core.cartpole.cartpole_direct_camera_env.CartpoleCameraEnv`
  to route image normalization through
  :func:`isaaclab.utils.images.normalize_camera_image` and defer the normalize past the
  frame-stack buffer for RGB-like data types, improving cartpole-camera frame-stacking
  throughput.
