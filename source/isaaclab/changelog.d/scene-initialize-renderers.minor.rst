Added
^^^^^

* Added :meth:`~isaaclab.scene.InteractiveScene.initialize_renderers` to
  pre-create renderer backends for all scene sensors with a
  ``renderer_cfg`` against the shared
  :class:`~isaaclab.renderers.render_context.RenderContext`. The method is
  idempotent and is now invoked from
  :class:`~isaaclab.envs.DirectRLEnv`,
  :class:`~isaaclab.envs.DirectMARLEnv`,
  :class:`~isaaclab.envs.ManagerBasedEnv`, and
  :class:`~isaaclab.envs.LeappDeploymentEnv` after scene construction so
  that renderer backend creation order is deterministic and front-loaded
  before the first :meth:`~isaaclab.sim.SimulationContext.reset`.
* Added :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.initialize`
  post-physics lifecycle hook (default no-op) that runs once per backend
  after :meth:`~isaaclab.sim.SimulationContext.reset` builds physics
  models. ``__init__`` now defines the pre-physics phase (eagerly invoked
  by :meth:`~isaaclab.scene.InteractiveScene.initialize_renderers`) and
  ``initialize`` defines the post-physics phase, letting backends whose
  setup needs scene data (e.g. a built Newton model) defer that work
  cleanly. Driven by
  :meth:`~isaaclab.renderers.render_context.RenderContext.ensure_initialize`,
  registered on
  :class:`~isaaclab.physics.physics_manager.PhysicsEvent` ``PHYSICS_READY``
  by :class:`~isaaclab.sim.SimulationContext` at ``order=5`` so it fires
  before sensor/asset callbacks (``order=10``). This decouples renderer
  post-physics setup from camera initialization. Backends created lazily
  after PHYSICS_READY are eagerly initialized at
  :meth:`~isaaclab.renderers.render_context.RenderContext.get_renderer`
  time.
