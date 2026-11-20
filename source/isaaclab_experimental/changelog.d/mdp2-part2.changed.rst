* **Breaking:** Changed :attr:`~isaaclab_experimental.envs.MDP2EnvCfg.task` to one task: a task that splits the worlds between
  tasks is :func:`~isaaclab_experimental.mdp2.split`, which replaces the list of tasks and ``task_worlds``. Migrate
  ``task=[WALK, STAND], task_worlds=[3072, 1024]`` to
  ``task=mdp2.split(walk=WALK, stand=STAND, worlds={"walk": 3072, "stand": 1024})``.
* **Breaking:** Changed the observation columns of MDP2 world configs: observations of the same name share their columns, as wide
  as the widest, and each name has its own, instead of each world config's observations starting at the first column.
* **Breaking:** Changed the action columns of MDP2 world configs: actions of the same name share their columns, as wide
  as the widest, as observations do, and the last actions hold zero in the columns a world does not take, which
  :meth:`~isaaclab_experimental.mdp2.MDP.action_mask` gives. This replaces the narrow copies of the last actions and
  ``MDP.narrow_action_dims``.
* Changed the order of an MDP2 reset: reset events run before the progress of the tasks restarts, so that they read
  the progress of the episode that ended, such as a curriculum over milestones. ``MDP.interval_writes`` is now
  :attr:`~isaaclab_experimental.mdp2.MDP.step_writes`, which also holds the writes of start events.
