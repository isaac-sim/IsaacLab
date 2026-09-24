Added
^^^^^

* Added a ``benchmark_mode`` option to the ``Isaac-RenderBenchmark-Franka-Cabinet`` task, read from the
  ``BENCHMARK_MODE`` environment variable. The default ``"render"`` mode wrote analytic joint poses after
  physics and required ``scene.lazy_sensor_update=True`` so rendering followed the pose write.
  Isaac RTX direct posing also rejected visualizers that pumped the Kit app; use ``--visualizer none``.
  Set ``BENCHMARK_MODE=physics_render`` to preserve actuator-driven animation. Both modes still stepped physics.
  The renderer sweep enabled physics and render timers and reported per-frame and combined timings
  from the profiling JSON file.
