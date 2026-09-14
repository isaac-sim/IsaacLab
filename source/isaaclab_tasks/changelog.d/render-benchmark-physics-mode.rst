Added
^^^^^

* Added a ``benchmark_mode`` option to the ``Isaac-RenderBenchmark-Franka-Cabinet`` task, read from the
  ``BENCHMARK_MODE`` environment variable. The default ``"render"`` mode poses the articulations by writing joint
  positions straight into the simulation so the renderer can be timed on a scene nothing had to be actuated to
  produce, while ``"physics_render"`` drives the same poses through the actuators so the solver does an ordinary
  task's tracking work. ``scripts/benchmarks/benchmark_renderer.py`` now enables the physics timer for every run,
  so each step's cost is recorded in the run log.
