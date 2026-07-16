Added
^^^^^

* Added :mod:`isaaclab.benchmark` as the public benchmark framework and added
  typed Python requests for runtime, startup, training, and play workflows.

Deprecated
^^^^^^^^^^

* Deprecated :mod:`isaaclab.test.benchmark` and the standalone runtime, startup,
  training, and play benchmark implementations. Import :mod:`isaaclab.benchmark`
  and use ``isaaclab benchmark`` or :func:`~isaaclab.benchmark.run_benchmark`
  instead.
