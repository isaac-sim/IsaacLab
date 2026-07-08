Added
^^^^^

* Added physics-backend-agnostic ``runtime.py`` and ``startup.py`` benchmark
  entry points. They emit :class:`~isaaclab.test.benchmark.RuntimeBundle` and
  :class:`~isaaclab.test.benchmark.StartupBundle` outputs and select physics and
  rendering backends with ``presets=`` Hydra tokens.
* Added the ``uv run isaaclab benchmark`` entry point for runtime and startup benchmarks.

Fixed
^^^^^

* Fixed benchmark recorder imports in uv environments by declaring the
  ``psutil`` dependency.
* Fixed the runtime benchmark to honor ``--device`` for Kitless physics
  backends.
* Fixed runtime and startup bundle metadata to record resolved task-default
  physics and rendering backends.
* Fixed runtime benchmark output to record Python-import and task-configuration
  startup timings.
* Fixed the startup benchmark total duration to end at the first synchronized
  environment step instead of including profile-report generation.
