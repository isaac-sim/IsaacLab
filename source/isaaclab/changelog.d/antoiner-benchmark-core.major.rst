Added
^^^^^

* Added a backend-agnostic benchmark core under :mod:`isaaclab.test.benchmark`,
  including the ``capture``, ``metrics``, ``builders``, ``stepping``,
  ``profiling``, and ``rllib_descriptor`` submodules, for assembling and
  emitting schema-v1 benchmark bundles (``RuntimeBundle`` / ``TrainingBundle`` /
  ``StartupBundle``).
* Added a ``schema`` output formatter that serializes a benchmark bundle through
  :class:`~isaaclab.test.benchmark.BaseIsaacLabBenchmark`, and taught
  ``BaseIsaacLabBenchmark`` to emit several formatters in one run from a
  comma-separated formatter selection and a new ``attach_bundle`` hook.

Changed
^^^^^^^

* **Breaking:** Renamed the benchmark metrics-formatter module
  ``isaaclab.test.benchmark.backends`` to ``isaaclab.test.benchmark.formatters``, and the
  ``MetricsBackend`` / ``MetricsBackendInterface`` classes to ``MetricsFormatter`` /
  ``MetricsFormatterInterface``. The output formatter classes (``JSONFileMetrics``,
  ``SummaryMetrics``, ``OsmoKPIFile``, ``OmniPerfKPIFile``) are unchanged but now live in the
  ``formatters`` module — update imports from ``isaaclab.test.benchmark.backends`` to
  ``isaaclab.test.benchmark.formatters``. The
  :class:`~isaaclab.test.benchmark.BaseIsaacLabBenchmark` constructor keeps ``backend_type`` as
  an alias for the new ``formatter_type`` argument, so callers that pass ``backend_type=``
  continue to work unchanged.
