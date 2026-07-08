Fixed
^^^^^

* Fixed docker installs deleting ``packaging`` from Isaac Sim's
  ``omni.isaac.core_archive`` prebundle by removing the ``packaging<24`` bound
  (no consumer requires it). The deletion dangled the symlink farm that
  ``omni.services.pip_archive`` shares with it and broke 13 extensions at
  startup.
* Added a fail-loud post-install check that aborts installation when pip
  operations leave a prebundled Isaac Sim package with a dangling
  ``__init__.py`` (other new dangling symlinks are reported as warnings).
* Fixed the ``isaacsim.robot_motion.pink`` extension failing to load after
  installation by moving the ``pin-pink`` pin from ``3.1.0`` to ``3.3.0``, which
  provides ``pink.exceptions.NoSolutionFound`` while staying below the pink 3.4
  task-API break. Environments installed manually should update with
  ``pip install pin-pink==3.3.0``.

Changed
^^^^^^^

* Changed the Pink IK force-install stack in the install CLI to derive from the
  root ``pyproject.toml`` pins instead of a hardcoded list, so version bumps are
  made in one place.
