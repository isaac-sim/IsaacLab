* Added :meth:`PhysicsManager.handle_actions <isaaclab.physics.PhysicsManager.handle_actions>`, which let a physics
  backend run the environment's actions and asset writes inside its step. :class:`~isaaclab.envs.ManagerBasedEnv`
  called it on its first step, unless its render interval was not a multiple of the decimation, a recorder term
  recorded after every physics step, or a sensor sampled at times other than whole environment steps, and skipped the
  actions and asset writes that the step ran.
* Added :attr:`RecorderManager.records_physics_steps <isaaclab.managers.RecorderManager.records_physics_steps>`.
* Added ``is_graphable`` to :class:`~isaaclab.managers.ActionTerm`,
  :class:`~isaaclab.managers.ActionManager`, :class:`~isaaclab.assets.AssetBase`,
  :class:`~isaaclab.actuators.ActuatorCollection` and :class:`~isaaclab.scene.InteractiveScene`. Joint, joint-to-limits
  and binary joint action terms set it. A subclass that overrode :meth:`~isaaclab.managers.ActionTerm.process_actions`
  or :meth:`~isaaclab.managers.ActionTerm.apply_actions` did not inherit it. Setting
  :attr:`ActionManager.is_graphable <isaaclab.managers.ActionManager.is_graphable>` to False before the first step
  kept the actions in the environment's loop, as LEAPP export did for its wrapped ``apply_action``.
* Added :attr:`WrenchComposer.pending <isaaclab.utils.wrench_composer.WrenchComposer.pending>` and the
  ``on_first_use`` argument of :class:`~isaaclab.utils.wrench_composer.WrenchComposer`, called at the first force
  write or :meth:`~isaaclab.utils.wrench_composer.WrenchComposer.add_raw_buffers_from` merge. Once forces were written,
  wrench work recorded into a CUDA graph no longer depended on host flags.
