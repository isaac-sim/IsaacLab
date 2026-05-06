Added
^^^^^

* Added ``frame_stack`` ring buffer to :class:`~isaaclab.sensors.camera.Camera` so
  backends without implicit damping can supply explicit temporal information via
  stacked frames in the channel dimension.
* Added :meth:`~isaaclab.renderers.RendererCfg.provides_temporal_camera_data`
  capability flag for renderers to declare whether their pipeline supplies
  inter-frame temporal information.
* Added :attr:`~isaaclab.physics.PhysicsCfg.requires_temporal_camera_data` for
  physics backends to declare whether they need temporal camera data.

Changed
^^^^^^^

* Changed :class:`~isaaclab.envs.DirectRLEnv` to raise ``ValueError`` when
  multiple cameras carry conflicting ``frame_stack`` values for a 3-tuple
  ``observation_space``, instead of silently picking the first.
