Changed
^^^^^^^

* **Breaking:** Changed ``Sb3VecEnvWrapper`` to expose normalized ``[-1, 1]`` bounds instead of the artificial
  ``[-100, 100]`` fallback for unbounded continuous action spaces. Pass ``action_bounds=(-100, 100)``
  to preserve the previous action space when loading an existing Stable-Baselines3 checkpoint.
