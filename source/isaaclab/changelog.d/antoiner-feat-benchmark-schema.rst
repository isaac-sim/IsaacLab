Added
^^^^^

* Added :mod:`isaaclab.test.benchmark.schema`, the public v1.0 JSON schema for
  benchmark bundles produced by the standalone scripts under
  ``scripts/benchmarks/``. Exposes :class:`~isaaclab.test.benchmark.schema.TrainingBundle`
  and :class:`~isaaclab.test.benchmark.schema.StartupBundle` plus the supporting
  :class:`~isaaclab.test.benchmark.schema.Versions`, :class:`~isaaclab.test.benchmark.schema.Hardware`,
  :class:`~isaaclab.test.benchmark.schema.Runtime`, :class:`~isaaclab.test.benchmark.schema.Resources`,
  and :class:`~isaaclab.test.benchmark.schema.Learning` records, along with
  :func:`~isaaclab.test.benchmark.schema.write_bundle_file` for emitting
  schema-compliant JSON. The package root re-exports the same surface so
  ``from isaaclab.test.benchmark import TrainingBundle`` works.

Changed
^^^^^^^

* Extended :class:`~isaaclab.test.benchmark.recorders.GPUInfoRecorder` and the
  system memory recorder to also report per-device **peak** memory and
  utilisation alongside the existing mean/std rows. Existing rows are
  unchanged; new rows are ``"Memory Used peak"``, ``"Utilization peak"``,
  ``"System Memory RSS peak"``, ``"System Memory VMS peak"``, and
  ``"System Memory USS peak"``. The peak rows are always emitted (initialised
  to ``0.0``) so downstream consumers see consistent keys regardless of
  whether any sample was recorded.
