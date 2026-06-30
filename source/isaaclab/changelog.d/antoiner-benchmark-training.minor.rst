Added
^^^^^

* Added a unified ``training.py`` benchmark dispatcher under ``scripts/benchmarks/`` that runs
  real training for the library selected with ``--rl_library`` (``rsl_rl``, ``rl_games``,
  ``skrl``, or ``sb3``) and emits :class:`~isaaclab.test.benchmark.TrainingBundle` output through
  ``--benchmark_formatter``. The RSL-RL and RL-Games adapters support optional success-metric
  early stopping; physics and rendering backends are selected with ``presets=`` Hydra tokens.
