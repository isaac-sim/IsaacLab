Fixed
^^^^^

* Added the missing ``ovphysx`` physics preset to ``Isaac-RenderBenchmark-Franka-Cabinet``
  and enabled automatic ``physx`` selection to use OvPhysX when running without Kit.

Changed
^^^^^^^

* **Breaking:** Changed the canonical render task's unset ``BENCHMARK_MODE`` default to
  ``None``, disabling benchmark animation and scope profiling. Set ``BENCHMARK_MODE=render``
  to retain direct posing or ``BENCHMARK_MODE=physics_render`` for actuator tracking;
  scope timings remained opt-in through their profiling flags. Renderer sweep users must
  also set an explicit benchmark mode to collect timings.
