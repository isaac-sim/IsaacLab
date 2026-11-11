* Added :mod:`isaaclab_experimental.mdp2`, an experimental MDP that compiles declarative task configs
  into an immutable :class:`~isaaclab_experimental.mdp2.MDP` built with
  :class:`~isaaclab_experimental.mdp2.MDPBuilder`, and runs it through the functional
  :func:`~isaaclab_experimental.mdp2.apply_actions`, :func:`~isaaclab_experimental.mdp2.evaluate`,
  :func:`~isaaclab_experimental.mdp2.reset`, and :func:`~isaaclab_experimental.mdp2.observe`. Terms provide
  batched Torch implementations and Warp implementations, either ``wp.func`` s for one world, which are fused
  into a few generated kernels per step, or functions that launch kernels themselves. Terms may keep per-world
  :class:`~isaaclab_experimental.mdp2.State` that advances once per step. Tasks may have
  :class:`~isaaclab_experimental.mdp2.Command` and :class:`~isaaclab_experimental.mdp2.Event` uses, and
  random terms draw from per-world streams that are bitwise identical on both backends. Before finalizing,
  the builder can describe, inspect, validate, and reorder the schedule within each phase, checking that
  every use runs after the uses of its phase that write the fields it reads. Terms can also read the MDP's own
  fields (:attr:`~isaaclab_experimental.mdp2.commands`, :attr:`~isaaclab_experimental.mdp2.last_action`,
  :attr:`~isaaclab_experimental.mdp2.episode_length`) and index fields with subsets of joints or bodies selected
  by name. Reward weights live in :class:`~isaaclab_experimental.mdp2.MDPState` and may change while the MDP
  runs, and ``finalize(diagnostics=True)`` also reports the reward and termination of every term. Terms that cannot
  run inside a CUDA graph are eager terms, ``Term(eager=fn)``, or set ``capturable=False``;
  :class:`~isaaclab_experimental.mdp2.CapturedMDP` replays the runtime functions as CUDA graphs and runs those
  terms between the graphs. With
  :meth:`~isaaclab_experimental.mdp2.MDPBuilder.replicate`, one MDP runs different task configs in different
  worlds of one scene: uses the configs share run once, the others only in their worlds, and actions and
  observations are padded to the widest config.
* Added :class:`~isaaclab_experimental.envs.MDP2Env` and :class:`~isaaclab_experimental.envs.MDP2RLEnv`, which run
  an MDP2 task on a Newton interactive scene with the Gym interface of the manager-based environments. Resets are
  masks over the worlds, so with the Warp backend each control step, physics included, replays as one CUDA graph.
* Added :mod:`isaaclab_experimental.mdp2.mdp`, built-in MDP2 terms with the names of their MDP1 functions: joint and
  root observations, rewards, terminations, contact rewards and terminations, velocity and pose commands, and joint
  and root reset and push events, on both backends. Observations take uniform noise, clipping, scaling, and history,
  and the scene data of :class:`~isaaclab_experimental.envs.MDP2Env` covers rigid objects, the root and body state
  of articulations, and contact sensors.
* Added per-term episode statistics to ``extras["log"]`` of :class:`~isaaclab_experimental.envs.MDP2RLEnv`, and the
  ``obs_buf``, ``log_dir``, and ``video_recorders`` members that the training scripts use.
* Added startup events, ``Event(mode=EventMode.STARTUP)``, which :class:`~isaaclab_experimental.envs.MDP2Env` runs
  once through :func:`~isaaclab_experimental.mdp2.startup`, and the eager terms
  :data:`~isaaclab_experimental.mdp2.mdp.randomize_rigid_body_mass`,
  :data:`~isaaclab_experimental.mdp2.mdp.randomize_rigid_body_com`, and
  :data:`~isaaclab_experimental.mdp2.mdp.randomize_rigid_body_material`, which change the physics model of a scene
  entity that an input selects through :meth:`~isaaclab_experimental.mdp2.TermContext.entity`.
* Added state to commands and events. Commands resample like events: their ``compute`` hook writes the command,
  passed as ``command``, and its state in place, and an ``update`` hook may change the command every step. Hooks take
  only the arguments they name.
* Added :data:`~isaaclab_experimental.mdp2.mdp.heading_velocity_command`, the velocity command with heading control,
  the velocity tracking, joint torque, joint acceleration, and base height rewards, and field offsets of
  :class:`~isaaclab_experimental.mdp2.DirectAction`, such as the default joint positions. The scene data provides
  joint position targets, joint accelerations, and the applied efforts.
* Added per-world reward weights: :attr:`~isaaclab_experimental.mdp2.MDPState.reward_weights` has one row per world,
  so world configs whose uses differ only in their weights share those uses. The ``task`` of
  :class:`~isaaclab_experimental.envs.MDP2EnvCfg` may be a list of tasks, which split the worlds evenly, and
  ``extras["log"]`` averages each term over the worlds where it runs. Function ``update`` hooks may take ``mask``,
  the worlds where their use runs, so that a command's update changes only its own worlds.
* Added ``init`` hooks to :class:`~isaaclab_experimental.mdp2.Term`, eager functions that
  :func:`~isaaclab_experimental.mdp2.startup` runs once to fill a term's state, and ``persistent``
  :class:`~isaaclab_experimental.mdp2.State` that outlives episodes. Int params can size the dimensions of a term's
  output and state, and inputs of different entities may give a dimension name different sizes.
* Added :data:`~isaaclab_experimental.mdp2.mdp.joint_vel_out_of_limit` and the startup events
  :data:`~isaaclab_experimental.mdp2.mdp.randomize_rigid_body_inertia`,
  :data:`~isaaclab_experimental.mdp2.mdp.randomize_actuator_gains`, and
  :data:`~isaaclab_experimental.mdp2.mdp.randomize_joint_parameters`.
  :data:`~isaaclab_experimental.mdp2.mdp.randomize_rigid_body_material` also randomizes rigid objects. The scene data
  of :class:`~isaaclab_experimental.envs.MDP2Env` provides the filtered contact forces ``normal_force_matrix_w``, and
  the environment applies the actions before every physics step, as MDP1 does.
* Added :func:`~isaaclab_experimental.mdp2.term_state`, which lets a term read the state of another term use, the
  ``scale`` of :class:`~isaaclab_experimental.mdp2.UniformNoise`, a field that multiplies the noise, and the
  curriculum term :data:`~isaaclab_experimental.mdp2.mdp.difficulty_scheduler`, whose persistent progress other terms
  read. It runs inside CUDA graphs: :class:`~isaaclab_experimental.mdp2.State` whose shape has no ``"worlds"`` is one
  buffer for every world, into which a hook that launches its own kernels can reduce. Eager terms keep host objects
  across calls in ``ctx.cache``.
* Added :class:`~isaaclab_experimental.mdp2.Quantity` uses, values per world that are computed once per step and
  that other terms read as ``mdp2.quantities.<name>``, so that terms share a calculation. Quantities may keep
  state, keep their value through a reset for the reset events to read, and are logged by
  :class:`~isaaclab_experimental.envs.MDP2RLEnv` as ``Metrics/<name>`` with ``log=True``. The terms
  :data:`~isaaclab_experimental.mdp2.mdp.value` and :data:`~isaaclab_experimental.mdp2.mdp.is_positive` reward and
  end episodes on a quantity.
* Added :meth:`~isaaclab_experimental.mdp2.SceneData.commit`, with which the environment makes the simulation see
  the fields that reset events wrote, and ``Event(commit=True)``, a reset event before which the scene commits the
  writes of the earlier ones, so that it reads the body poses they gave. The ``gravity_w`` field of the scene data of
  :class:`~isaaclab_experimental.envs.MDP2Env` is the gravity of the physics model, which events may write.
