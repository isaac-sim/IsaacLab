Added
^^^^^

* Added the :meth:`~isaaclab.physics.physics_manager_cfg.PhysicsCfg.provides_implicit_damping` and
  :meth:`~isaaclab.renderers.renderer_cfg.RendererCfg.provides_temporal_camera_data` capability
  methods, so physics and renderer backends can declare whether a camera observation carries the
  temporal information a policy needs to infer velocity (used to decide frame stacking). Base
  defaults: physics has implicit damping (``True``); a renderer provides no temporal data (``False``).
