Fixed
^^^^^

* Added a fail-loud post-install check that aborts installation when any
  distribution disappeared from an Isaac Sim ``pip_prebundle`` directory during
  pip operations. In docker installs a forced downgrade could previously delete
  prebundled packages (e.g. ``packaging``) that Isaac Sim extensions share via
  symlinks, silently breaking extension startup.
* Fixed the ``isaacsim.robot_motion.pink`` extension failing to load after
  installation by moving the ``pin-pink`` pin from ``3.1.0`` to ``3.3.0``, which
  provides ``pink.exceptions.NoSolutionFound`` while staying below the pink 3.4
  task-API break. Environments installed manually should update with
  ``pip install pin-pink==3.3.0``.
