Added
^^^^^

* Added the :meth:`~isaaclab.physics.physics_manager.PhysicsManager.provides_implicit_damping` and
  :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.provides_temporal_camera_data` capability
  classmethods on the runtime backend bases, so physics and renderer backends declare whether a
  camera observation carries the temporal information a policy needs to infer velocity (used to
  decide frame stacking). Base defaults: physics has implicit damping (``True``); a renderer
  provides no temporal data (``False``).
* Added :meth:`~isaaclab.renderers.Renderer.resolve_class` to resolve a renderer's implementation
  class from its configuration without instantiating it (so tasks can query the above classmethod
  before a simulation exists).
