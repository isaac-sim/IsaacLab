Changed
^^^^^^^

* **Breaking:** Changed ``--viz none`` to parse to an empty list instead of ``None``, so ``args_cli.visualizer is None``
  means no ``--viz`` was passed and ``[]`` means all visualizers are disabled. The ``visualizer_explicit`` and
  ``visualizer_disable_all`` launcher arguments are removed; check ``args_cli.visualizer`` instead.
* Changed :func:`~isaaclab.app.launch_simulation` to also detect a ``newton_rtx`` visualizer listed in
  ``visualizer_cfgs``, so the OVRTX runtime starts and the Kit conflict check applies.

Added
^^^^^

* Added ``launcher_type`` to :class:`~isaaclab.physics.PhysicsCfg` and :class:`~isaaclab.renderers.RendererCfg`,
  defaulting to ``None`` for configs whose runtime needs no launcher.

Removed
^^^^^^^

* **Breaking:** Removed ``SettingsManager.instance()``. Use :func:`~isaaclab.app.get_settings_manager`.
* **Breaking:** Removed ``SettingsManager.set_bool``, ``set_int``, ``set_float``, and ``set_string``. Use
  :meth:`~isaaclab.app.SettingsManager.set`, which dispatches on the value type.
* **Breaking:** Removed ``isaaclab.app.make_physics_cfg``. Select the backend with the ``physics`` launcher
  argument of :func:`~isaaclab.app.launch_simulation`, or construct the physics config directly.
* **Breaking:** Removed ``isaaclab.app.logging_utils.resolve_python_logging_level``.
  :func:`~isaaclab.app.launch_simulation` applies the ``--verbose`` / ``--info`` level itself.
