Changed
^^^^^^^

* Changed ``isaaclab benchmark training_multigpu`` to accept ``--check_success`` for RSL-RL and RL-Games.
  The success metric is now summed across ranks before each convergence check, so every rank stops at
  the same iteration.
