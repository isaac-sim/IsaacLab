* Added :attr:`~isaaclab.envs.ManagerBasedRLEnvCfg.capture_step` and :attr:`~isaaclab.envs.DirectRLEnvCfg.capture_step`:
  on Newton the environment step replays from CUDA graphs, with non-graphable operations as eager islands. See
  :class:`~isaaclab.envs.utils.step_capture.EnvStepCapture`, :class:`~isaaclab.utils.StepGraph`,
  :meth:`~isaaclab.physics.PhysicsManager.step_ops` and the ``supports_graph_capture`` attribute of managers, terms,
  actuators and sensors.
* Added ``env_mask`` to the ``reset`` of managers, the scene, assets and actuators, and the helpers
  :func:`~isaaclab.utils.env_mask_from_ids`, :func:`~isaaclab.utils.env_ids_from_mask`,
  :func:`~isaaclab.utils.takes_env_mask` and :func:`~isaaclab.utils.env_selection_kwargs`.
* Added a ``physics_manager`` argument to :class:`~isaaclab.sim.SimulationContext` and
  :class:`~isaaclab.envs.ManagerBasedEnv`, and :attr:`~isaaclab.utils.wrench_composer.WrenchComposer.used`.
