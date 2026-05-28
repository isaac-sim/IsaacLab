Added
^^^^^

* Added :mod:`isaaclab.benchmark.schema`, the public v1.0 JSON schema for
  benchmark bundles produced by the standalone scripts under
  ``scripts/benchmarks/``. Exposes :class:`~isaaclab.benchmark.schema.TrainingBundle`
  and :class:`~isaaclab.benchmark.schema.StartupBundle` plus the supporting
  :class:`~isaaclab.benchmark.schema.Versions`, :class:`~isaaclab.benchmark.schema.Hardware`,
  :class:`~isaaclab.benchmark.schema.Runtime`, :class:`~isaaclab.benchmark.schema.Resources`,
  and :class:`~isaaclab.benchmark.schema.Learning` records, along with
  :func:`~isaaclab.benchmark.schema.write_bundle_file` for emitting
  schema-compliant JSON. The package root re-exports the same surface so
  ``from isaaclab.benchmark import TrainingBundle`` works.

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
