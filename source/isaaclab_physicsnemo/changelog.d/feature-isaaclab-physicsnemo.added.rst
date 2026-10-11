* Added the ``isaaclab_physicsnemo`` extension, which records Isaac Lab Newton episodes as
  PhysicsNeMo Robotics trajectory datasets, trains NeRD dynamics artifacts on them, and runs those
  artifacts as the Newton solver of the same environment.
* Added :class:`~isaaclab_physicsnemo.physics.PhysicsNeMoSolverCfg` and
  :class:`~isaaclab_physicsnemo.physics.NewtonPhysicsNeMoManager`, an Isaac Lab Newton solver manager
  that validates an artifact against the live scene and advances it as the environment's physics.
* Added :func:`~isaaclab_physicsnemo.evaluate_rollout_nrmse`, the prefix-averaged recursive-rollout
  error of the FlashNeRD evaluation, which training reports on held-out trajectories.
* Added :class:`~isaaclab_physicsnemo.TrajectoryDatasetRecorder`, which records joint efforts or, with
  ``control_targets``, the joint-position targets of a control period, and
  :func:`~isaaclab_physicsnemo.train_dynamics_artifact`, which trains NeRD or ParallelNeRD on
  device-resident windows with FP16 autocast, data-parallel ranks, and resumable checkpoints.
  Recording can write bounded dataset shards, loading reads them with worker processes and an optional
  cache, and deployment can run the network's projections in FP16.
* Added Cartpole and ANYmal-C examples with their scenes, recording, training, deployment, and PPO
  scripts that train in learned physics and evaluate in MuJoCo Warp. The ANYmal-C example follows the
  FlashNeRD locomotion experiment, and a CartPole benchmark repeats the protocol of the FlashNeRD
  main-results study.
