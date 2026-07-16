Added
^^^^^

* Added ``--frontend {torch,warp}`` to the shared reinforcement-learning
  training CLI (all supported RL libraries) for selecting the environment
  runtime; default ``torch`` is unchanged.
* Added :mod:`isaaclab_experimental.envs.frontend` runtime selector and
  :meth:`isaaclab_experimental.managers.SceneEntityCfg.from_stable` used by
  ``--frontend=warp`` to adapt stable cfgs onto the warp runtime.

Changed
^^^^^^^

* Changed :class:`~isaaclab_experimental.envs.ManagerBasedRLEnvWarp` to adapt
  its configuration in ``__init__`` via
  :func:`~isaaclab_experimental.envs.frontend.adapt_cfg_for_warp`, so
  registered warp task variants can derive from stable configurations instead
  of duplicating them.

Fixed
^^^^^

* Fixed :class:`~isaaclab_experimental.envs.ManagerBasedRLEnvWarp` to probe
  :meth:`~isaaclab.sim.SimulationContext.has_active_visualizers` after the
  ``get_setting`` API was reshaped.
