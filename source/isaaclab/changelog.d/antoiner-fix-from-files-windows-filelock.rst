Fixed
^^^^^

* Fixed :mod:`isaaclab.sim.spawners.from_files` failing to import on Windows
  due to an unconditional ``import fcntl`` (Unix-only). The distributed-rank
  USD spawn lock now uses :class:`filelock.FileLock`, which works on both
  Windows and POSIX.

Changed
^^^^^^^

* Added :mod:`filelock` to ``isaaclab`` install requirements.
