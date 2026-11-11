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
