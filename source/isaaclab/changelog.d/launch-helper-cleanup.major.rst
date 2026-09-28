Added
^^^^^

* Added :func:`~isaaclab.app.fuse_kit_args` to the :mod:`isaaclab.app` exports. It moved to the lightweight
  ``isaaclab.app.argv`` module; import it from :mod:`isaaclab.app` instead of ``isaaclab.app.sim_launcher``.

Removed
^^^^^^^

* **Breaking:** Removed ``SettingsManager.instance()``. Use :func:`~isaaclab.app.get_settings_manager`.
* **Breaking:** Removed ``SettingsManager.set_bool``, ``set_int``, ``set_float``, and ``set_string``. Use
  :meth:`~isaaclab.app.SettingsManager.set`, which dispatches on the value type.
* **Breaking:** Removed ``isaaclab.app.make_physics_cfg``. Select the backend with the ``physics`` launcher
  argument of :func:`~isaaclab.app.launch_simulation`, or construct the physics config directly.
* **Breaking:** Removed ``isaaclab.app.logging_utils.resolve_python_logging_level``.
  :func:`~isaaclab.app.launch_simulation` applies the ``--verbose`` / ``--info`` level itself.
