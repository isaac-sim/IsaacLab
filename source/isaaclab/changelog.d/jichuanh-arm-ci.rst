Fixed
^^^^^

* Added a defensive fallback in :class:`isaaclab.app.AppLauncher` so it derives
  ``EXP_PATH`` from the installed ``isaacsim`` package when the env var is not
  set. ``isaacsim.bootstrap_kernel`` normally sets ``EXP_PATH`` on first import,
  but the early-return path in its bootstrap (triggered under some pip install
  layouts on aarch64) skips the env-var setup. Previously this caused
  ``KeyError: 'EXP_PATH'`` deep inside ``_resolve_experience_file``; now
  AppLauncher resolves the path from ``isaacsim.__file__`` and stores it back
  into the environment so subsequent code can rely on it.
