Added
^^^^^

* Added a ``PlayBundle`` schema type and a unified ``play.py`` benchmark dispatcher under
  ``scripts/benchmarks/`` that loads a trained checkpoint and benchmarks policy inference for the
  RL library selected with ``--rl_library`` (``rsl_rl``, ``rl_games``, ``skrl``, ``sb3``), emitting
  the policy's inference throughput plus reward / episode-length / success. The checkpoint is taken
  from ``--checkpoint`` (local path or Nucleus URI) or, failing that, the published Nucleus
  checkpoint for the task.
