Added
^^^^^

* Added a unified ``training.py`` benchmark dispatcher under ``scripts/benchmarks/`` that runs
  real training for the RL library selected with ``--rl_library`` (``rsl_rl``, ``rl_games``,
  ``skrl``, or ``sb3``) and emits a ``TrainingBundle``. Each library has a thin adapter and an
  optional success-metric early stop; physics/rendering backends are selected with ``presets=``
  Hydra tokens, matching ``train.py``.
