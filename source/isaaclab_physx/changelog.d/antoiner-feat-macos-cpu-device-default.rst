Changed
^^^^^^^

* Changed the ``--device`` default to ``cpu`` on macOS, which has no CUDA, so commands run without
  passing ``--device cpu``. Other platforms still default to ``cuda:0``.
