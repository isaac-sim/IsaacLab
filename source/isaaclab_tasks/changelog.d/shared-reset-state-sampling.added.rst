* Added ``ResetStateSampler`` to share success-guided row assignment and completed-episode
  bookkeeping between cached and procedural reset catalogs, using the existing success monitor.
* Moved reusable weighted sampling strategies to ``isaaclab_tasks.utils.sampling`` and retained
  the previous NIST import paths for existing configurations.
* Added a sparse Franka cube-pickup reference task that generates validated reset states
  through physical approach and grasp rollouts and samples them using recent success rates.
