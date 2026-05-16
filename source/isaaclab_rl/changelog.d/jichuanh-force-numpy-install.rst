Fixed
^^^^^

* Excluded the broken ``numpy 2.3.5`` release from the package's install requirements.
  ``numpy 2.3.5``'s vendored OpenBLAS registers a ``pthread_atfork`` handler that
  crashes Kit's ``libomni.platforminfo`` ``fork()`` during ``SimulationApp`` startup.
  See ``source/isaaclab/setup.py`` and ``isaaclab.cli.commands.install`` for the
  authoritative install-time defense.
