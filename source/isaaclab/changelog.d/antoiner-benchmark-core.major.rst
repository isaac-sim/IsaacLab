Added
^^^^^

* Added a backend-agnostic benchmark core under :mod:`isaaclab.test.benchmark`,
  including the ``capture``, ``metrics``, ``builders``, ``stepping``,
  and ``profiling`` submodules, for assembling and emitting schema-v1 benchmark
  bundles (``RuntimeBundle`` / ``TrainingBundle`` /
  ``StartupBundle``).
* Added a ``schema`` output formatter that serializes a benchmark bundle through
  :class:`~isaaclab.test.benchmark.BaseIsaacLabBenchmark`, and taught
  ``BaseIsaacLabBenchmark`` to emit several formatters in one run from a
  comma-separated formatter selection and a new ``attach_bundle`` hook.
* Added runtime and package version metadata to schema benchmark bundles,
  including IsaacLab extensions, OVRTX, OVPhysX, MuJoCo, CUDA bindings, and
  USD Core.

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

Fixed
^^^^^

* Fixed multi-phase :class:`~isaaclab.test.benchmark.OsmoKPIFile` output
  overwriting earlier phases by writing one phase-suffixed JSON file per phase.
* Fixed benchmark run metadata to use resolved task defaults for physics and
  rendering backends.
* Fixed simulation launch failures being reported with a zero process exit
  status during Kit fast shutdown.
* Fixed benchmark metadata so Kit-full runs now report Kit and Isaac Sim
  versions while Kitless runs report null.
* Fixed benchmark metadata to report the installed OVPhysX runtime version.
* Fixed benchmark metadata to preserve null values for unavailable OVRTX and
  OVPhysX runtimes.
