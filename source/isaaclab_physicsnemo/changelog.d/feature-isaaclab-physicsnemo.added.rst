* Added the ``isaaclab_physicsnemo`` extension, which records Isaac Lab Newton episodes as
  PhysicsNeMo Robotics trajectory datasets, trains NeRD dynamics artifacts on them, and runs those
  artifacts as the Newton solver of the same environment.
* Added :class:`~isaaclab_physicsnemo.physics.PhysicsNeMoSolverCfg` and
  :class:`~isaaclab_physicsnemo.physics.NewtonPhysicsNeMoManager`, an Isaac Lab Newton solver manager
  that validates an artifact against the live scene and advances it as the environment's physics.
* Added :func:`~isaaclab_physicsnemo.evaluate_rollout_nrmse`, the prefix-averaged recursive-rollout
  error of the FlashNeRD evaluation, which training reports on held-out trajectories.
* Added Cartpole and ANYmal-D flat-terrain teacher configurations with recording, training, and
  deployment examples, task-independent Cartpole data collection, and a Cartpole PPO example that
  trains in learned physics and evaluates in MuJoCo Warp.
