Fixed
^^^^^

* Fixed docker installs deleting distributions from Isaac Sim's ``pip_prebundle``
  directories: pip operations now run through the concrete kit interpreter instead
  of the ``python.sh`` launcher (which re-injects prebundle paths onto
  ``PYTHONPATH``), so pip can no longer uninstall prebundled packages such as
  ``packaging`` and break extension startup.
* Added a fail-loud post-install check that aborts installation when any
  distribution disappeared from an Isaac Sim prebundle during pip operations.
* Fixed the ``isaacsim.robot_motion.pink`` extension failing to load after
  installation by moving the ``pin-pink`` pin from ``3.1.0`` to ``3.3.0``, which
  provides ``pink.exceptions.NoSolutionFound`` while staying below the pink 3.4
  task-API break. Environments installed manually should update with
  ``pip install pin-pink==3.3.0``.
