* Added the ``auto_reset`` argument to :meth:`~isaaclab.envs.ManagerBasedRLEnv.step`. When False, the environments
  whose episodes ended are not reset within the step and stay in
  :attr:`~isaaclab.envs.ManagerBasedRLEnv.pending_reset_buf`, so callers that execute action chunks can reset them
  at the chunk boundary with a step with ``auto_reset`` or with :meth:`~isaaclab.envs.ManagerBasedRLEnv.reset`.
