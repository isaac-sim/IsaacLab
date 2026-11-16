* Added :mod:`isaaclab_experimental.mdp2`, an experimental MDP that compiles declarative task configs into an
  immutable :class:`~isaaclab_experimental.mdp2.MDP` with :class:`~isaaclab_experimental.mdp2.MDPBuilder`, and runs
  it through the functional :func:`~isaaclab_experimental.mdp2.startup`,
  :func:`~isaaclab_experimental.mdp2.apply_actions`, :func:`~isaaclab_experimental.mdp2.evaluate`,
  :func:`~isaaclab_experimental.mdp2.reset`, and :func:`~isaaclab_experimental.mdp2.observe`. A
  :class:`~isaaclab_experimental.mdp2.TaskCfg` binds terms to scene fields and to the MDP's own fields
  (:attr:`~isaaclab_experimental.mdp2.commands`, :attr:`~isaaclab_experimental.mdp2.quantities`,
  :attr:`~isaaclab_experimental.mdp2.last_action`, :func:`~isaaclab_experimental.mdp2.term_state`) as actions,
  observations, rewards, terminations, commands, startup, reset, and interval events, and
  :class:`~isaaclab_experimental.mdp2.Quantity` uses that terms share. The builder can describe, validate, and
  reorder the schedule within each phase, and reports every error at once.
* Added term backends: batched Torch hooks, and Warp hooks that are either ``wp.func`` s for one world, fused into a
  few generated kernels per runtime function, or functions that launch kernels themselves. A Warp MDP runs the Torch
  hooks of terms without Warp ones between its kernels. Terms may keep per-world or shared
  :class:`~isaaclab_experimental.mdp2.State`, filled once by an ``init`` hook and optionally persistent across
  episodes, and random terms draw from per-world streams that are bitwise identical on both backends. Eager terms,
  ``Term(eager=fn)``, are Python functions with direct access to the scene for work that cannot run in a CUDA graph.
* Added :meth:`~isaaclab_experimental.mdp2.MDPBuilder.replicate`, and task lists in
  :class:`~isaaclab_experimental.envs.MDP2EnvCfg` with ``task_worlds``, to run different task configs in different
  worlds of one scene. Uses the configs share run once, the others only in their worlds, reward weights are per
  world, uses that the configs define differently share their state, and actions and observations are padded to the
  widest config.
* Added :class:`~isaaclab_experimental.envs.MDP2Env` and :class:`~isaaclab_experimental.envs.MDP2RLEnv`, which run an
  MDP2 task on a Newton interactive scene with the Gym interface of the manager-based environments. Resets are masks
  over the worlds, so every step after the first replays as CUDA graphs, physics included, on both backends: one
  graph without eager terms, and the eager terms between graphs otherwise. Reset events write scene fields in place,
  including fixed-base root poses and gravity, and ``Event(commit=True)`` makes the simulation see earlier writes.
  ``extras["log"]`` holds per-term episode statistics and logged quantities. Articulations must have actuators
  without state (implicit, ideal PD, and DC motor).
* Added :mod:`isaaclab_experimental.mdp2.mdp`, built-in terms with the names of their MDP1 counterparts: joint, root,
  and command observations with noise, clipping, scaling, and history; velocity tracking, regularization, contact,
  and air-time rewards; terminations; velocity, heading, and pose commands; joint and root reset and push events;
  startup randomization of mass, center of mass, inertia, material, actuator gains, and joint parameters; and the
  :data:`~isaaclab_experimental.mdp2.mdp.difficulty_scheduler` curriculum.
