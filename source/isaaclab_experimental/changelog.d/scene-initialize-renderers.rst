Changed
^^^^^^^

* Pre-create renderer backends in
  :class:`~isaaclab_experimental.envs.ManagerBasedEnvWarp` and
  :class:`~isaaclab_experimental.envs.DirectRLEnvWarp` by invoking
  :meth:`~isaaclab.scene.InteractiveScene.initialize_renderers` after scene
  construction so that renderer backend creation order is deterministic and
  front-loaded before the first
  :meth:`~isaaclab.sim.SimulationContext.reset`.
