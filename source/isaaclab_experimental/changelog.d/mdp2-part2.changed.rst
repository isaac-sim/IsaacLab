* **Breaking:** Changed :attr:`~isaaclab_experimental.envs.MDP2EnvCfg.task` to one task: a task that splits the worlds between
  tasks is :func:`~isaaclab_experimental.mdp2.split`, which replaces the list of tasks and ``task_worlds``. Migrate
  ``task=[WALK, STAND], task_worlds=[3072, 1024]`` to
  ``task=mdp2.split(walk=WALK, stand=STAND, worlds={"walk": 3072, "stand": 1024})``.
* **Breaking:** Changed the observation columns of MDP2 world configs: observations of the same name share their columns, as wide
  as the widest, and each name has its own, instead of each world config's observations starting at the first column.
