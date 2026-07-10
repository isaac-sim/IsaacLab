Added
^^^^^

* Added a ``PlayBundle`` schema type and ``isaaclab benchmark play`` to load a trained checkpoint
  and benchmark policy inference for the RL library selected with ``--rl_library``
  (``rsl_rl``, ``rl_games``, ``skrl``, ``sb3``), emitting the policy's inference throughput plus
  reward / episode-length / success. The checkpoint is taken
  from ``--checkpoint`` (a local path, Nucleus URI, or ``latest``/``best`` selector)
  or, failing that, the published Nucleus checkpoint for the task.
