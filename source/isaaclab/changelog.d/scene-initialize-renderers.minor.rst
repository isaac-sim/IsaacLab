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
