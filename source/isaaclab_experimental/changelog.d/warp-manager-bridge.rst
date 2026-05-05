Added
^^^^^

* Added :class:`~isaaclab_experimental.envs.warp_frontend.WarpFrontend`, a
  runtime adapter that lets any stable manager-based RL task config run on the
  experimental warp runtime (:class:`~isaaclab_experimental.envs.ManagerBasedRLEnvWarp`)
  without a parallel ``-Warp-v0`` registration. The adapter is built on a
  pluggable :class:`~isaaclab_experimental.envs.warp_frontend.CompatRule`
  pipeline; new incompatibilities (sensor types, term-cfg fields, action
  classes) are added by writing a small rule subclass instead of editing the
  dispatcher. The default rules cover physics-preset resolution, dropping
  unsupported sensors, in-place :class:`SceneEntityCfg` promotion, mdp
  function swaps, and action-class swaps. The frontend also dispatches
  direct envs by verifying their registered entry-point class lives under
  ``isaaclab_experimental`` / ``isaaclab_tasks_experimental`` and routing
  through :func:`gym.make` unchanged.

* Added a ``--frontend={stable,warp}`` flag to ``rsl_rl/train.py``. When set
  to ``warp`` the script auto-injects ``presets=newton`` (so Hydra picks the
  Newton physics preset before the adapter runs), warns on conflicting
  ``presets=`` overrides, and dispatches the env through ``WarpFrontend``
  instead of :func:`gym.make`. ``render_mode`` is forwarded so ``--video``
  keeps working under the warp frontend.

Fixed
^^^^^

* Fixed a regression in :class:`~isaaclab_experimental.envs.ManagerBasedRLEnvWarp`
  introduced when the ``SimulationContext.get_setting`` API was reshaped:
  the warp env now mirrors the stable env and probes
  :meth:`~isaaclab.sim.SimulationContext.has_active_visualizers` instead of
  splitting a string setting that no longer exists.
