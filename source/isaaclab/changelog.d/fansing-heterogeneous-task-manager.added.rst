* Added :attr:`~isaaclab.envs.ManagerBasedRLEnvCfg.tasks` and :class:`~isaaclab.managers.HeterogeneousTaskManager`
  so that each environment runs one of several tasks and switches tasks on reset, where its task changed. The tasks
  contribute their rigid objects, termination terms and episode lengths; a task leaves a slot empty with
  :meth:`~isaaclab.managers.HeterogeneousTaskManager.placeholder`, kinematic in the slots of kinematic objects.
