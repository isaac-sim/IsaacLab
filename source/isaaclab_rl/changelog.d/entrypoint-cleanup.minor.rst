Changed
^^^^^^^

* Restructured the ``isaaclab_rl.entrypoints`` backends so that every train, play, and export module exposes the
  same ``run(argv)`` function with module-level imports; the play backends are no longer module-level scripts
  executed through ``runpy``. Shared playback pieces moved to ``isaaclab_rl.entrypoints.common``
  (``add_common_play_args``, ``run_playback``, ``resolve_published_checkpoint``, ``resolve_seed``,
  ``normalize_task_name``) and the LEAPP exporters share ``run_export``, ``prepare_export_env``, and
  ``leapp_capture`` from ``export_common``. Downstream code that imported the play modules as scripts should
  call their ``run(argv)`` function instead.
* Added :func:`~isaaclab_rl.rsl_rl.check_rsl_rl_version` and :func:`~isaaclab_rl.rsl_rl.create_rsl_rl_runner`,
  :func:`~isaaclab_rl.skrl.check_skrl_version` and :func:`~isaaclab_rl.skrl.import_skrl_runner`, and
  :meth:`~isaaclab_rl.rl_games.RlGamesVecEnvWrapper.from_agent_cfg` with
  :func:`~isaaclab_rl.rl_games.register_rl_games_env`, which the entrypoints now share. The RSL-RL play and
  export backends require ``rsl-rl-lib`` 5.0.1 or newer like training already did; the legacy
  ``export_policy_as_jit``/``export_policy_as_onnx`` path for older releases was dropped from playback.
* The framework wrappers validate their environment through :func:`isaaclab_rl.utils.env_types.check_env_type`
  instead of repeating the type check.
* ``apply_env_overrides`` now also applies ``--disable_fabric`` and ``--export_io_descriptors``, so the
  ``--disable_fabric`` flag of the play entrypoints is functional and IO descriptors are only enabled when the
  flag is passed. The zero and random agents honor ``--deterministic``.
* Population-Based Training restarts re-execute :data:`sys.executable` instead of the legacy
  ``_isaac_sim/python.sh`` launcher.

Removed
^^^^^^^

* Removed the unused ``dispatch_library_entrypoint``, ``import_local_module``, ``configure_io_descriptors``, and
  ``wrap_training_capture`` helpers from ``isaaclab_rl.entrypoints.common``. Use the unified
  ``isaaclab_rl.entrypoints.run_train_cli``/``run_play_cli`` dispatchers, ``apply_env_overrides``, and
  ``wrap_sensor_capture`` instead. ``pre_launch_video_config`` takes ``(env_cfg, args_cli)``; its unused
  ``log_dir`` parameter was dropped.
