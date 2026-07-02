Fixed
^^^^^

* Added a fail-loud post-install check that aborts installation when pip
  operations leave new dangling symlinks in Isaac Sim's ``pip_prebundle``
  directories. In docker installs a forced downgrade could previously delete a
  prebundled package (e.g. ``packaging``) that other Isaac Sim extensions share
  via symlink farms, silently breaking extension startup.
* Fixed the ``isaacsim.robot_motion.pink`` extension failing to load after
  installation by moving the ``pin-pink`` pin from ``3.1.0`` to ``3.3.0``, which
  provides ``pink.exceptions.NoSolutionFound`` while staying below the pink 3.4
  task-API break. Environments installed manually should update with
  ``pip install pin-pink==3.3.0``.
