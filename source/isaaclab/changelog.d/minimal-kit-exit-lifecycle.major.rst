Changed
^^^^^^^

* Changed :func:`~isaaclab.app.launch_simulation` to start each runtime through the launcher its resolved
  config names in ``launcher_type``.
* Deprecated ``isaaclab.app.AppLauncher``; it now starts Isaac Sim / Kit through
  :func:`~isaaclab.app.launch_simulation`. Migrate to :func:`~isaaclab.app.add_launcher_args` plus
  ``with launch_simulation(cfg, args_cli):``.
* Changed Kit exit handling: an unhandled exception exits with 1 and ``SIGINT`` raises
  :class:`KeyboardInterrupt`; ``SIGTERM``, ``SIGABRT``, and ``SIGSEGV`` keep their default actions.
* **Breaking:** Removed ``SettingsManager.initialize_carb_settings``, the module-level
  ``initialize_carb_settings``, and ``SettingsManager.is_omniverse_mode``. The Kit launcher now passes
  ``carb.settings`` to :meth:`~isaaclab.app.SettingsManager.set_backend`.
* **Breaking:** Removed ``isaaclab.sim.utils.is_current_stage_in_memory`` and the hidden ``--cpu`` launcher
  argument. Use ``--device cpu`` instead of ``--cpu``.
* Changed the environments to seed Replicator through a hook the Kit launcher registers with
  :func:`~isaaclab.utils.seed.register_seed_hook`, so core modules no longer import ``omni.replicator``.

Added
^^^^^

* Added :class:`~isaaclab.app.SimulationLauncher` for backend runtimes and
  :func:`isaaclab.test.utils.launch_test_simulation` to start Kit in test modules.
* Added :func:`~isaaclab.utils.seed.register_seed_hook` so a runtime can seed its own random number generators.
* Added ``class_type`` to :class:`~isaaclab.envs.ManagerBasedEnvCfg` and :class:`~isaaclab.envs.ManagerBasedRLEnvCfg`,
  so ``env_cfg.class_type(env_cfg)`` constructs the environment without importing its class.

Fixed
^^^^^

* Fixed ``{DIR}`` in an inherited ``class_type`` resolving against the module of a config subclass that is
  missing ``@configclass``, which made ``cfg.class_type(cfg)`` fail with ``ModuleNotFoundError``.
