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
