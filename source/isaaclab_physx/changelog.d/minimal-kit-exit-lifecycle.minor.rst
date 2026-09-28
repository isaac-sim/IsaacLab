Added
^^^^^

* Added :class:`~isaaclab_physx.app.KitLauncher`, the Isaac Sim / Kit launcher formerly
  ``isaaclab.app.AppLauncher``. :func:`~isaaclab.app.launch_simulation` constructs it; scripts do not.
* Added ``launcher_type`` to :class:`~isaaclab_physx.physics.PhysxCfg`, naming the launcher its runtime needs.
* Added :class:`~isaaclab_physx.app.KitStageBackendCfg`, the simulation backend that attaches the stage to
  Kit's USD context and closes it with the simulation.
* Added :func:`~isaaclab_physx.app.show_stage_in_viewport`, which replaces ``isaaclab.sim.utils.show_stage_in_viewport``.
* Added ``set_gravity`` to :class:`~isaaclab_physx.physics.PhysxManager` to set the scene-wide gravity.

Changed
^^^^^^^

* Changed the Isaac RTX renderer to enable ``omni.replicator.core`` itself, so scripts no longer load it.
* Changed the ``randomize_visual_color`` and ``randomize_visual_texture_material`` Replicator event terms
  to seed Replicator with ``env.cfg.seed`` when set, since the environments' ``seed()`` no longer does.
