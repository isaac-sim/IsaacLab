* Added :meth:`PhysicsManager.bind_control <isaaclab.physics.PhysicsManager.bind_control>`, which let a physics
  backend run the environment's actions and asset writes inside its step. :class:`~isaaclab.envs.ManagerBasedEnv`
  called it after loading its managers and skipped the actions and asset writes that the step ran.
* Added ``supports_graph_capture`` to :class:`~isaaclab.managers.ManagerTermBase`,
  :class:`~isaaclab.managers.ActionManager`, :class:`~isaaclab.assets.AssetBase`,
  :class:`~isaaclab.actuators.ActuatorCollection` and :class:`~isaaclab.scene.InteractiveScene`, and
  ``apply_every_physics_step`` to :class:`~isaaclab.managers.ActionTerm` and :class:`~isaaclab.managers.ActionManager`.
  Joint, joint-to-limits and binary joint action terms set them.
* Added :attr:`WrenchComposer.pending <isaaclab.utils.wrench_composer.WrenchComposer.pending>` and the
  ``on_first_use`` argument of :class:`~isaaclab.utils.wrench_composer.WrenchComposer`. Once forces were written,
  wrench work recorded into a CUDA graph no longer depended on host flags.
