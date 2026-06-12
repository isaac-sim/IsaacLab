Added
^^^^^

* Added the public v1.0 benchmark schema under :mod:`isaaclab.test.benchmark`
  (``schema`` and ``serialize`` submodules): the JSON contract for benchmark
  bundles produced by the standalone scripts under ``scripts/benchmarks/``.
  Exposes :class:`~isaaclab.test.benchmark.schema.RuntimeBundle`,
  :class:`~isaaclab.test.benchmark.schema.TrainingBundle`, and
  :class:`~isaaclab.test.benchmark.schema.StartupBundle` plus the supporting
  :class:`~isaaclab.test.benchmark.schema.Versions`,
  :class:`~isaaclab.test.benchmark.schema.Hardware`,
  :class:`~isaaclab.test.benchmark.schema.RunConfig`,
  :class:`~isaaclab.test.benchmark.schema.Runtime`,
  :class:`~isaaclab.test.benchmark.schema.Resources`, and
  :class:`~isaaclab.test.benchmark.schema.Learning` records, along with
  :func:`~isaaclab.test.benchmark.serialize.write_bundle_file` for emitting
  schema-compliant JSON atomically.
  Each bundle also carries an optional ``extra`` mapping of free-form scalar
  values for producer-specific data outside the stable contract.

Changed
^^^^^^^

* Extended :class:`~isaaclab.test.benchmark.recorders.GPUInfoRecorder` and the
  system memory recorder to also report per-device **peak** memory alongside
  the existing mean/std rows. New rows are ``"GPU Memory Used peak"``
  (``"GPU {i} Memory Used peak"`` for multi-GPU), ``"System Memory RSS peak"``,
  ``"System Memory VMS peak"``, and ``"System Memory USS peak"``. These peak
  rows are always emitted (initialised to ``0.0``) so downstream consumers see
  a consistent key set regardless of whether any sample was recorded.
