Added
^^^^^

* Added :class:`~isaaclab.app.SimulationLauncher` for backend runtimes and
  :func:`isaaclab.test.utils.launch_test_simulation` to start Kit in test modules.
* Added ``class_type`` to :class:`~isaaclab.envs.ManagerBasedEnvCfg` and :class:`~isaaclab.envs.ManagerBasedRLEnvCfg`,
  so ``instantiate(env_cfg)`` constructs the environment without importing its class.

Changed
^^^^^^^

* Changed :func:`~isaaclab.app.launch_simulation` to start each runtime through the launcher its resolved
  config names in ``launcher_type``, and Isaac Sim / Kit for Kit needs no config names, such as ``--viz kit``.
* Changed Kit exit handling: an unhandled exception exits with 1 and ``SIGINT`` raises
  :class:`KeyboardInterrupt`; ``SIGTERM``, ``SIGABRT``, and ``SIGSEGV`` keep their default actions.
* Changed ``seed()`` of the environments to no longer seed Replicator, so core modules no longer import
  ``omni.replicator``. The Replicator event terms seed it with ``env.cfg.seed`` instead; call
  ``omni.replicator.core.set_global_seed`` directly to reseed Replicator after the environment is created.
* Changed :class:`~isaaclab.sim.SimulationContext` to attach its stage to Kit's USD context through the
  :class:`~isaaclab_physx.app.KitStageBackendCfg` backend, which closes it with the simulation.
  :func:`~isaaclab.sim.utils.clear_stage` and :func:`~isaaclab.sim.utils.close_stage` no longer run Kit app
  updates or close Kit's USD context themselves.
* Changed :meth:`~isaaclab.sim.SimulationContext.clear_instance` to close native backends newest first, so a
  backend closes before the backends it was created on top of.

Deprecated
^^^^^^^^^^

* Deprecated ``isaaclab.app.AppLauncher``; it now starts Isaac Sim / Kit through
  :func:`~isaaclab.app.launch_simulation`. Migrate to :func:`~isaaclab.app.add_launcher_args` plus
  ``with launch_simulation(cfg, args_cli):``.

Fixed
^^^^^

* Fixed ``{DIR}`` in an inherited ``class_type`` resolving against the module of a config subclass that is
  missing ``@configclass``, which made ``instantiate(cfg)`` fail with ``ModuleNotFoundError``.

Removed
^^^^^^^

* **Breaking:** Removed ``SceneDataProvider.usd_stage`` and ``SceneDataProvider.get_usd_stage()``; use
  ``SimulationContext.instance().stage``.
* **Breaking:** Removed ``isaaclab.sim.utils.is_current_stage_in_memory``; compare the current stage against
  :attr:`~isaaclab.sim.SimulationContext.stage` instead.
* **Breaking:** Removed ``isaaclab.sim.utils.update_stage``. The simulation's reset, step, and render process
  stage changes; remove the calls.
* **Breaking:** Removed ``isaaclab.sim.utils.show_stage_in_viewport``; use
  :func:`isaaclab_physx.app.show_stage_in_viewport`, which requires a running Kit app.
* **Breaking:** Removed ``SettingsManager.initialize_carb_settings``, the module-level
  ``initialize_carb_settings``, and ``SettingsManager.is_omniverse_mode``. The Kit launcher sets the
  ``carb.settings`` backend through :meth:`~isaaclab.app.SettingsManager.set_backend`; use
  :func:`~isaaclab.utils.version.has_kit` to check for Kit.
* **Breaking:** Removed the hidden ``--cpu`` launcher argument; use ``--device cpu``.
