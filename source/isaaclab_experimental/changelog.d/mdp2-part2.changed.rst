* **Breaking:** Changed :attr:`~isaaclab_experimental.envs.MDP2EnvCfg.task` to one task: a task that splits the
  worlds between tasks is :func:`~isaaclab_experimental.mdp2.split`, which replaces the list of tasks and
  ``task_worlds``. Migrate
  ``task=[WALK, STAND], task_worlds=[3072, 1024]`` to
  ``task=mdp2.split(walk=WALK, stand=STAND, worlds={"walk": 3072, "stand": 1024})``.
* **Breaking:** Changed :meth:`~isaaclab_experimental.mdp2.MDPBuilder.add_task` to take one task per builder:
  combine tasks with :func:`~isaaclab_experimental.mdp2.compose` and add the result once.
* **Breaking:** Changed the observation columns of MDP2 world configs: observations of the same name share their
  columns, as wide as the widest, and each name has its own, instead of each world config's observations starting at
  the first column.
* **Breaking:** Changed the action columns of MDP2 world configs: actions of the same name share their columns, as wide
  as the widest, as observations do, and the last actions hold zero in the columns a world does not take, which
  :meth:`~isaaclab_experimental.mdp2.MDP.action_mask` gives. This replaces the narrow copies of the last actions and
  ``MDP.narrow_action_dims``.
* Changed the order of an MDP2 reset: reset events run before the progress of the tasks restarts, so that they read
  the progress of the episode that ended, such as a curriculum over milestones. ``MDP.interval_writes`` is now
  :attr:`~isaaclab_experimental.mdp2.MDP.step_writes`, which also holds the writes of start events.
* **Breaking:** Changed ``Quantity(log=True)`` to ``Quantity(record=True)``: a recorded quantity is reduced per episode,
  logged as ``Metrics/<name>`` and recorded in the episode records of an episode budget. With an episode budget,
  ``done`` stays whether an episode ended, and the environment resets only the worlds that restart.
* Changed the actuators that MDP2 environments accept: any capturable Isaac Lab actuator
  (:attr:`~isaaclab.actuators.ActuatorBase.capturable`) and any actuator that Newton runs, instead of a fixed list of
  actuator types. Where Newton cannot capture one, such as a network loaded from a TorchScript checkpoint, the physics
  and the scene reset run between the graphs.
* Changed :class:`~isaaclab_experimental.envs.MDP2Env` to capture its steps through the physics backend
  (:meth:`~isaaclab.physics.PhysicsManager.capture`) instead of a Newton function: a backend that cannot capture its
  step steps between the MDP's graphs. Replayed steps now count their physics steps and advance the simulation time.
* Changed :class:`~isaaclab_experimental.envs.MDP2Env` to raise a ``ValueError`` on a physics backend other than
  Newton, instead of capturing writes that the backend synchronizes or reading stale state.
* Changed ``MDP2EnvCfg.verify_capture`` on physics without a reproducible mode to check the MDP alone, with the physics
  step replaced by nothing, bitwise, instead of comparing to a relative tolerance.
* Changed :class:`~isaaclab_experimental.envs.MDP2Env` to render only the cameras that some term of the task reads.
* Changed :class:`~isaaclab_experimental.envs.MDP2RLEnv` to keep its episode statistics in one generated kernel
  inside the captured step, instead of about 15 Torch operations per step, and to sum them once per step for
  ``extras["log"]``: about 15 to 35 µs less per step at 4,096 worlds. Without a time limit, rewards are logged per
  second of each episode rather than as zero, and a choice's tasks are averaged over the finished episodes that drew
  them.
