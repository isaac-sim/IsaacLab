Added
^^^^^

* Added physics-backend-agnostic ``runtime.py`` and ``startup.py`` benchmark
  entry points. They emit :class:`~isaaclab.test.benchmark.RuntimeBundle` and
  :class:`~isaaclab.test.benchmark.StartupBundle` outputs and select physics and
  rendering backends with ``presets=`` Hydra tokens.

Fixed
^^^^^

* Fixed the runtime benchmark to honor ``--device`` for Kitless physics
  backends.
* Fixed runtime and startup bundle metadata to record resolved task-default
  physics and rendering backends.
