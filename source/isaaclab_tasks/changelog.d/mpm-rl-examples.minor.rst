Added
^^^^^

* Added Newton MPM Franka pouring and UR10 particle-pushing reinforcement
  learning tasks with reset-safe particle randomization and rigid-particle
  coupling. The pouring task uses compact current-state observations and an
  outcome-aware reset curriculum with per-region progress metrics.
* Added bounded sparse MPM configurations with CUDA graph capture and
  fixed-payload resets for both tasks.
* Added randomized pile footprint, shape, and lateral placement to the UR10
  particle-pushing task, with partial-progress and split-pile resets for
  multi-pass manipulation.
