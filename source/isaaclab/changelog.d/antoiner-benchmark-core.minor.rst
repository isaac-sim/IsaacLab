Added
^^^^^

* Added a backend-agnostic benchmark core under :mod:`isaaclab.test.benchmark` — the
  ``capture``, ``metrics``, ``builders``, ``stepping``, ``profiling``, and
  ``backend_descriptor`` submodules — for assembling and emitting the schema-v1 benchmark
  bundles (``RuntimeBundle`` / ``TrainingBundle`` / ``StartupBundle``).
* Added a ``schema`` output backend that serializes a benchmark bundle through the
  :class:`~isaaclab.test.benchmark.BaseIsaacLabBenchmark` metrics-backend system, and taught
  ``BaseIsaacLabBenchmark`` to emit several backends in one run via a comma-separated
  ``--benchmark_backend`` and a new ``attach_bundle`` hook.
