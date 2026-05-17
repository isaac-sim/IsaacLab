Changed
^^^^^^^

* Moved ``robomimic`` from an opt-in extra (``isaaclab_mimic[robomimic]``) to a
  required dependency of :mod:`isaaclab_mimic` on Linux (via a ``sys_platform``
  marker). ``robomimic`` is now installed automatically whenever
  ``isaaclab_mimic`` is installed on Linux; no extra selector is needed.
