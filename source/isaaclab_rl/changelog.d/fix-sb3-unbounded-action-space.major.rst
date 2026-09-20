Changed
^^^^^^^

* **Breaking:** Removed the artificial ``[-100, 100]`` fallback for unbounded continuous action spaces in
  ``Sb3VecEnvWrapper``. Environments must now declare finite raw policy action bounds for Stable-Baselines3.
  Existing SB3 checkpoints that stored the legacy bounds must be retrained or used with an environment that
  declares matching finite action bounds.
* Rejected Gymnasium ``TimeLimit`` wrappers around Isaac Lab vectorized environments. Configure
  ``episode_length_s`` on the environment instead so timeouts remain independent for every sub-environment.
