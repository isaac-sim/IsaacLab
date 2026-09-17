Fixed
^^^^^

* Added configurable finite bounds for unbounded Stable-Baselines3 action spaces,
  validation for invalid bounds, and a warning when falling back to ``[-100, 100]``.
  Explicit normalized bounds are recommended for continuous-control algorithms such
  as SAC and TD3.
