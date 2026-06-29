Added
^^^^^

* Added unified, physics-backend-agnostic benchmark entry points under ``scripts/benchmarks/``:
  ``runtime.py`` (steps an environment with random actions and emits a ``RuntimeBundle``) and
  ``startup.py`` (``cProfile`` startup-phase profiling, emits a ``StartupBundle``). Physics and
  rendering backends are selected with ``presets=`` Hydra tokens, matching ``train.py``.
