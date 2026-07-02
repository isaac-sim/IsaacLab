Added
^^^^^

* Added a unified ``training.py`` benchmark dispatcher under ``scripts/benchmarks/`` that runs
  real training for the library selected with ``--rl_library`` (``rsl_rl``, ``rl_games``,
  ``skrl``, or ``sb3``) and emits :class:`~isaaclab.test.benchmark.TrainingBundle` output through
  ``--benchmark_formatter``. The RSL-RL and RL-Games adapters support optional success-metric
  early stopping; physics and rendering backends are selected with ``presets=`` Hydra tokens.

Fixed
^^^^^

* Fixed SKRL and Stable-Baselines3 training bundles omitting task success rates
  reported by the environment.
* Fixed training bundles omitting Python-import and task-configuration startup timings.
* Fixed RSL-RL training benchmarks failing in environments without Git LFS.
  Benchmark runs now skip RSL-RL source-state archiving while retaining
  TensorBoard metric logging.
