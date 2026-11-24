* Added :class:`~isaaclab.managers.HeterogeneousTaskCfg` records and
  :class:`~isaaclab.managers.HeterogeneousTaskManager` to run tasks over one flat scene. Records
  declared an inclusion set, initial-state overrides, instruction, duration and termination terms;
  composition retained the source catalog once and compiled fixed runtime slots for reset-time mesh
  switching. Shared candidates carried multiple task memberships without mutating source definitions.
  The default ``initial_task`` inclusion retained one initial task world. Switching required Newton
  with XPBD or MJWarp external contacts and the Newton Warp renderer for switched camera geometry.
