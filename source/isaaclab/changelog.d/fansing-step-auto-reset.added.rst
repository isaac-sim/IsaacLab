* Added the ``auto_reset`` argument to :meth:`~isaaclab.envs.ManagerBasedRLEnv.step`. When False, terminated
  environments are not reset within the step, so callers that execute action chunks can reset them with
  :meth:`~isaaclab.envs.ManagerBasedRLEnv.reset` at the chunk boundary.
