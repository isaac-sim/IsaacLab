Added
^^^^^

* Added Newton MPM Franka pouring and UR10 particle-pushing reinforcement
  learning tasks with reset-safe particle randomization and rigid-particle
  coupling. The pouring task uses compact current-state observations and an
  outcome-aware reset curriculum with per-region progress metrics.
* Added bounded sparse MPM configurations with CUDA graph capture and
  fixed-payload resets for both tasks.
* Added randomized pile shape and table placement, paired collision-screened
  robot starts, and a competence-based reset curriculum to the UR10
  particle-pushing task. Every level keeps the same single-pile, single-sweep
  objective.
