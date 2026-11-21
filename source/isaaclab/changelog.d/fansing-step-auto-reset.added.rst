* Added the ``auto_reset`` argument to :meth:`~isaaclab.envs.ManagerBasedRLEnv.step`. When False, the environments
  whose episodes ended are not reset within the step and keep reporting ``terminated`` or ``truncated`` through
  :attr:`~isaaclab.envs.ManagerBasedRLEnv.pending_terminated_buf` and
  :attr:`~isaaclab.envs.ManagerBasedRLEnv.pending_time_outs_buf`, so callers that execute action chunks can reset
  them at the chunk boundary with a step with ``auto_reset`` or with :meth:`~isaaclab.envs.ManagerBasedRLEnv.reset`.
