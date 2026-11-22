* Added :class:`~isaaclab.managers.HeterogeneousTaskManager` so that each environment runs one of several tasks and
  switches tasks on reset, where its task changed. :meth:`~isaaclab.managers.HeterogeneousTaskManager.compose_tasks`
  adds the tasks' rigid objects, termination terms and episode lengths to an environment configuration; a task leaves
  a slot empty with :meth:`~isaaclab.managers.HeterogeneousTaskManager.placeholder`, kinematic in the slots of
  kinematic objects. The objects are selected as reset-time mesh variants, which need the Newton backend with the
  XPBD solver, or with the MJWarp solver and ``use_mujoco_contacts=False``, and of the camera renderers only
  :class:`~isaaclab_newton.renderers.NewtonWarpRendererCfg` shows the selected objects.
