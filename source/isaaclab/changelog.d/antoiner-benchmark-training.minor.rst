Added
^^^^^

* Added ``isaaclab benchmark training`` to run real training with the library selected by
  ``--rl_library`` (``rsl_rl``, ``rl_games``, ``skrl``, or ``sb3``) and emit
  :class:`~isaaclab.test.benchmark.TrainingBundle` output through ``--benchmark_formatter``.
  The RSL-RL and RL-Games adapters support optional success-metric
  early stopping; physics and rendering backends are selected with ``presets=`` Hydra tokens.
* Added ``run.json`` manifests to benchmark training logs, enabling compatible
  runs to be selected through ``--checkpoint latest`` or ``--checkpoint best``.

Fixed
^^^^^

* Fixed SKRL and Stable-Baselines3 training bundles omitting task success rates
  reported by the environment.
* Fixed training bundles omitting Python-import and task-configuration startup timings.
* Fixed benchmark training help and RL-library adapter dispatch through
  ``isaaclab benchmark training``.
* Fixed RSL-RL training benchmarks failing in environments without Git LFS.
  Benchmark runs now skip RSL-RL source-state archiving while retaining
  TensorBoard metric logging.
