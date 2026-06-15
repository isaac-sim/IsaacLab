Fixed
^^^^^

* Fixed ``AttributeError: 'NoneType' object has no attribute 'get_clone_plan'``
  in :meth:`NewtonManager._ensure_visualization_model` on the PhysX backend
  (regression from #6119). The renderer's ``PHYSICS_READY`` callback can fire
  before ``PhysicsManager.initialize()`` sets ``_sim``, and
  :meth:`SimulationContext.get_clone_plan` may also legitimately return ``None``
  for scenes that have not yet replicated. ``build_visualization_builder_from_stage_envs``
  now accepts ``clone_plan=None`` and falls back to the env_0 prototype-replicate
  path that pre-existed PR #6119, restoring the ``physx,newton_renderer,*``
  benchmark presets in omniperf-benchmark.
